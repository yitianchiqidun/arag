"""Builds the assistant with real services from a config: models, the TiDB docs index, the
local cluster (optional), a SQLite checkpointer for sessions, and per-turn trace files.

Traces: every event of a turn, and the steps of each document search, are appended to
``<trace_dir>/<session>.jsonl`` (one JSON object per line, with the turn number), so a
conversation can be replayed after the fact.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from research_agent.assistant.graph import Assistant, Services
from research_agent.assistant.retrieval import DocAnswer, DocRetriever
from research_agent.assistant.routing import Channel, ChatClient, channel_for
from research_agent.config import LLMConfig, RunConfig
from research_agent.llm.client import client_from_config
from research_agent.llm.pricing import PricingTable
from research_agent.tidb.db import TiDBConfig


class AssistantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_config: str = "configs/tidb_docs_qwen37_plus.yaml"  # A-RAG / Naive RAG over the docs
    router: LLMConfig  # condense + intent (small, fast)
    agent: LLMConfig  # diagnosis loop
    # Intents answered by the fast channel (Naive RAG, asked to cite); the rest use A-RAG.
    # Empty: everything goes to A-RAG (stage 6). Stage 7 decides the default (D32).
    fast_intents: list[str] = []
    tidb: TiDBConfig = TiDBConfig()
    use_cluster: bool = True
    checkpoint_db: str = "data/assistant/sessions.sqlite"
    trace_dir: str = "data/assistant/traces"
    audit_file: str = "data/assistant/writes.jsonl"
    max_steps: int = 12
    turn_timeout_s: float | None = 240.0  # the diagnosis loop stops after this (stage 9)
    # Overrides the run config's embedding device, e.g. "cpu" in the Docker image (stage 9).
    embedding_device: str | None = Field(
        default_factory=lambda: os.environ.get("EMBEDDING_DEVICE") or None
    )
    # Tools (stages 10-11): where they run, and the harness around every call (D37, D38).
    tool_source: Literal["inprocess", "mcp"] = "inprocess"
    # MCP mode: the server is started over stdio with this config (default: the same file).
    mcp_server_config: str = Field(
        default_factory=lambda: os.environ.get("ASSISTANT_CONFIG", "configs/assistant.yaml")
    )
    log_dir: str = "~/.tiup/data/arag"  # the tiup playground's component logs
    report_dir: str = "data/assistant/reports"
    tool_audit_file: str = "data/assistant/tool_calls.jsonl"
    max_tool_calls: int = 16  # per turn
    max_doc_searches: int = 2  # per turn

    @classmethod
    def from_yaml(cls, path: str | Path) -> AssistantConfig:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))


class TracedDocs:
    """Wraps the retriever to keep each answer's steps for the trace."""

    def __init__(self, retriever: DocRetriever) -> None:
        self.retriever = retriever
        self.last: list[DocAnswer] = []

    def answer(self, question: str, channel: Channel = "arag") -> DocAnswer:
        doc = self.retriever.answer(question, channel)
        self.last.append(doc)
        return doc


def attach_tools(services: Any, cfg: AssistantConfig) -> None:
    """The tools' own services on ``services``: cluster reads and proposals (when TiDB is
    reachable), logs, configuration, reports. Shared by the agent and the MCP server."""
    from research_agent.assistant.report import ReportWriter
    from research_agent.tidb.logs import LogSearch

    services.logs = LogSearch(cfg.log_dir)
    services.reports = ReportWriter(cfg.report_dir)
    if not cfg.use_cluster:
        return
    from research_agent.tidb import cluster
    from research_agent.tidb.config_reader import ConfigReader
    from research_agent.tidb.db import config_database, readonly_database
    from research_agent.tidb.readonly import ReadOnlyTools
    from research_agent.tidb.write import WriteTools

    if cluster.port_open(cfg.tidb.host, cfg.tidb.port):
        reader = ReadOnlyTools(readonly_database(cfg.tidb), cfg.tidb.max_rows)
        services.reader = reader
        services.writes = WriteTools(reader)
        services.configs = ConfigReader(config_database(cfg.tidb), cfg.tidb.max_rows)


def mcp_command(cfg: AssistantConfig) -> list[str]:
    return [
        sys.executable,
        "-m",
        "research_agent.toolkit.mcp_server",
        "--config",
        cfg.mcp_server_config,
    ]


def build_services(cfg: AssistantConfig) -> tuple[Services, TracedDocs]:
    from research_agent.toolkit.harness import ToolHarness, TurnBudget
    from research_agent.toolkit.local import InProcessTools, ToolBackend

    run = RunConfig.from_yaml(cfg.run_config)
    if cfg.embedding_device:
        embedding = run.embedding.model_copy(update={"device": cfg.embedding_device})
        run = run.model_copy(update={"embedding": embedding})
    pricing = PricingTable.from_yaml(run.pricing_file)
    clients: dict[str, LLMConfig] = {"router": cfg.router, "agent": cfg.agent}

    def llm(role: str) -> ChatClient:
        return client_from_config(clients[role], pricing)

    docs = TracedDocs(DocRetriever(run))
    services = Services(
        llm=llm,
        docs=docs,
        max_steps=cfg.max_steps,
        turn_timeout_s=cfg.turn_timeout_s,
        choose_channel=lambda question, intent: channel_for(intent, cfg.fast_intents),
    )
    attach_tools(services, cfg)
    if services.reader is not None:
        from research_agent.tidb.db import admin_database
        from research_agent.tidb.write import WriteExecutor

        # Confirmed writes always run here, never behind a tool (D37).
        services.executor = WriteExecutor(admin_database(cfg.tidb), services.reader, cfg.audit_file)
    backend: ToolBackend
    if cfg.tool_source == "mcp":
        from research_agent.toolkit.mcp_client import McpTools

        backend = McpTools(mcp_command(cfg))
    else:
        backend = InProcessTools(services)
    services.tools = ToolHarness(
        backend,
        audit_file=cfg.tool_audit_file,
        budget=TurnBudget(cfg.max_tool_calls, cfg.max_doc_searches),
    )
    return services, docs


def sqlite_checkpointer(path: str | Path) -> Any:
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


class AssistantApp:
    """The assistant with tracing; one instance serves many sessions."""

    def __init__(
        self, cfg: AssistantConfig, services: Services | None = None, checkpointer: Any = None
    ) -> None:
        self.cfg = cfg
        if services is None:
            services, self.docs = build_services(cfg)
        else:
            self.docs = services.docs if isinstance(services.docs, TracedDocs) else None  # type: ignore[assignment]
        self.services = services
        self.assistant = Assistant(services, checkpointer or sqlite_checkpointer(cfg.checkpoint_db))
        self._lock = threading.Lock()  # the graph and the retriever are used one turn at a time

    def _trace(
        self, session_id: str, events: Iterator[dict[str, Any]], kind: str
    ) -> Iterator[dict[str, Any]]:
        path = Path(self.cfg.trace_dir) / f"{session_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        turn = len(self.assistant.history(session_id)) // 2 + 1
        with self._lock, open(path, "a", encoding="utf-8") as f:
            if self.docs is not None:
                self.docs.last = []
            for event in events:
                row = {
                    "at": datetime.now().isoformat(timespec="seconds"),
                    "turn": turn,
                    "kind": kind,
                    **event,
                }
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                yield event
            f.writelines(
                json.dumps(
                    {"turn": turn, "kind": "doc_trace", **doc.model_dump()}, ensure_ascii=False
                )
                + "\n"
                for doc in (self.docs.last if self.docs is not None else [])
            )

    def ask(self, session_id: str, question: str) -> Iterator[dict[str, Any]]:
        yield from self._trace(session_id, self.assistant.ask(session_id, question), "ask")

    def confirm(self, session_id: str, approved: bool) -> Iterator[dict[str, Any]]:
        yield from self._trace(session_id, self.assistant.confirm(session_id, approved), "confirm")

    def pending(self, session_id: str) -> dict[str, Any] | None:
        return self.assistant.pending(session_id)

    def history(self, session_id: str) -> list[dict[str, str]]:
        return self.assistant.history(session_id)
