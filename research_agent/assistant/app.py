"""Builds the assistant with real services from a config: models, the TiDB docs index, the
local cluster (optional), a SQLite checkpointer for sessions, and per-turn trace files.

Traces: every event of a turn, and the steps of each document search, are appended to
``<trace_dir>/<session>.jsonl`` (one JSON object per line, with the turn number), so a
conversation can be replayed after the fact.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from research_agent.assistant.graph import Assistant, ChatClient, Services
from research_agent.assistant.retrieval import Channel, DocAnswer, DocRetriever
from research_agent.config import LLMConfig, RunConfig
from research_agent.llm.client import client_from_config
from research_agent.llm.pricing import PricingTable
from research_agent.tidb.db import TiDBConfig


class AssistantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_config: str = "configs/tidb_docs_qwen37_plus.yaml"  # A-RAG / Naive RAG over the docs
    router: LLMConfig  # condense + intent (small, fast)
    agent: LLMConfig  # diagnosis loop
    tidb: TiDBConfig = TiDBConfig()
    use_cluster: bool = True
    checkpoint_db: str = "data/assistant/sessions.sqlite"
    trace_dir: str = "data/assistant/traces"
    audit_file: str = "data/assistant/writes.jsonl"
    max_steps: int = 12

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


def build_services(cfg: AssistantConfig) -> tuple[Services, TracedDocs]:
    run = RunConfig.from_yaml(cfg.run_config)
    pricing = PricingTable.from_yaml(run.pricing_file)
    clients: dict[str, LLMConfig] = {"router": cfg.router, "agent": cfg.agent}

    def llm(role: str) -> ChatClient:
        return client_from_config(clients[role], pricing)

    docs = TracedDocs(DocRetriever(run))
    services = Services(llm=llm, docs=docs, max_steps=cfg.max_steps)
    if cfg.use_cluster:
        from research_agent.tidb import cluster
        from research_agent.tidb.db import admin_database, readonly_database
        from research_agent.tidb.readonly import ReadOnlyTools
        from research_agent.tidb.write import WriteExecutor, WriteTools

        if cluster.port_open(cfg.tidb.host, cfg.tidb.port):
            reader = ReadOnlyTools(readonly_database(cfg.tidb), cfg.tidb.max_rows)
            services.reader = reader
            services.writes = WriteTools(reader)
            services.executor = WriteExecutor(admin_database(cfg.tidb), reader, cfg.audit_file)
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
