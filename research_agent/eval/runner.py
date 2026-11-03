"""Batch-run A-RAG (or a baseline) with metering, reusing the upstream BatchRunner.

Upstream ``scripts/batch_runner.py`` records answers, cost and the tool trajectory, but not
token counts, latency or why the loop stopped. ``MeteredBatchRunner`` subclasses it and only
swaps the tool set, system prompt, LLM client and agent class, and enriches each record.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from arag import BaseAgent, LLMClient, ToolRegistry
from arag.tools.read_chunk import ReadChunkTool
from research_agent.agent.prompt import build_system_prompt, load_upstream_prompt
from research_agent.agent.tools import (
    build_tool_registry,
    plain_text_tokens,
    shared_semantic_tool,
)
from research_agent.baselines.naive_rag import SYSTEM_PROMPT as NAIVE_SYSTEM_PROMPT
from research_agent.baselines.naive_rag import ChunkIndex, NaiveRAG
from research_agent.config import RunConfig
from research_agent.eval.citations import load_chunk_refs, record_citations
from research_agent.eval.subsets import sha256_file
from research_agent.llm.client import MeteredLLMClient, UsageSummary, client_from_config
from research_agent.llm.pricing import PricingTable
from research_agent.tracing.recorder import TimedToolRegistry, TraceRecorder
from research_agent.tracing.schema import TRAJECTORIES_FILE
from research_agent.upstream import REPO_ROOT, load_script

_BatchRunner: type = load_script("batch_runner").BatchRunner


class RunAborted(RuntimeError):
    """An account-level API error (llm.client.AccountError) or the cost limit stopped the run.

    Upstream BatchRunner writes a failed question as an ``Error: ...`` answer and skips it on
    resume, so running out of balance mid-run would turn every remaining question into a
    recorded wrong answer. Questions hit by such an error, or not started because of the cost
    limit, are not written instead, and the run raises at the end, so the same command
    resumes them once the account is fixed or the limit raised.
    """


class CostBudget:
    """A spending limit shared by the runs of a suite (agent and judge costs).

    No question starts once ``spent`` plus a reserve for the questions still in flight
    (workers x the most expensive question so far) reaches the limit, so the limit is not
    overshot by more than an unusually expensive in-flight question.
    """

    def __init__(self, limit: float, spent: float = 0.0, workers: int = 1) -> None:
        self.limit = limit
        self.spent = spent
        self.workers = workers
        self.max_question = 0.0
        self._lock = threading.Lock()

    def charge(self, cost: float, question: bool = True) -> None:
        with self._lock:
            self.spent += cost
            if question:
                self.max_question = max(self.max_question, cost)

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self.spent + self.workers * self.max_question >= self.limit

    def describe(self) -> str:
        return f"cost limit {self.limit:.2f} reached (spent {self.spent:.4f})"


class MeteredAgent(Protocol):
    """What the runner needs from an agent: A-RAG's BaseAgent or a baseline."""

    @property
    def llm(self) -> LLMClient: ...

    last_result: dict[str, Any] | None

    def run(self, query: str) -> dict[str, Any]: ...


class RecordingAgent(BaseAgent):
    """BaseAgent that keeps the raw result of the last ``run`` (upstream drops some flags)."""

    last_result: dict[str, Any] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        plain_text_tokens(self)  # tool results may contain "<|endoftext|>"

    def run(self, query: str) -> dict[str, Any]:
        self.last_result = None
        result: dict[str, Any] = super().run(query)
        self.last_result = result
        return result


def termination_reason(result: dict[str, Any] | None, usage: UsageSummary) -> str:
    """Why the agent loop ended.

    Upstream BaseAgent breaks out of the loop on an LLM exception and then reports
    ``max_loops_exceeded``; failed LLM calls let us tell the two apart.
    """
    if result is None:
        return "exception"
    if result.get("token_budget_exceeded"):
        return "token_budget"
    if result.get("max_loops_exceeded"):
        return "llm_error" if usage.failed_calls else "max_loops"
    return "answered"


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    # Untracked files (e.g. local notes) do not change the code that ran.
    dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    return {"commit": run("rev-parse", "HEAD"), "dirty": dirty}


def config_fingerprint(cfg: RunConfig) -> str:
    return hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()[:16]


class MeteredBatchRunner(_BatchRunner):  # type: ignore[misc, valid-type]
    def __init__(
        self,
        run_config: RunConfig,
        pricing: PricingTable,
        questions_file: str,
        output_dir: str,
        limit: int | None = None,
        num_workers: int = 5,
        budget: CostBudget | None = None,
    ) -> None:
        self.run_config = run_config
        self.budget = budget
        self.pricing = pricing
        self._naive_resources: tuple[ChunkIndex, ReadChunkTool] | None = None
        self._abort_reason: str | None = None
        self._recorders: dict[MeteredAgent, TraceRecorder] = {}
        self._chunk_refs = load_chunk_refs(run_config.data.chunks_file)
        super().__init__(
            config=run_config.to_arag_config(),
            questions_file=questions_file,
            output_dir=output_dir,
            limit=limit,
            num_workers=num_workers,
            verbose=False,
        )
        # Upstream __init__ loaded default.txt; rebuild it for the configured tool set.
        if run_config.method == "naive_rag":
            self._system_prompt = NAIVE_SYSTEM_PROMPT
        else:
            self._system_prompt = build_system_prompt(load_upstream_prompt(), run_config.agent)

    def _init_shared_tools(self) -> ToolRegistry:
        if self.run_config.method == "naive_rag":
            return ToolRegistry()
        return build_tool_registry(self.run_config)

    def _naive_rag(self) -> NaiveRAG:
        cfg = self.run_config
        # Reuse the semantic_search model: a second copy would not fit next to it on 6 GB.
        semantic = shared_semantic_tool(cfg.data, cfg.embedding)
        if self._naive_resources is None:
            reader = ReadChunkTool(chunks_file=cfg.data.chunks_file)
            plain_text_tokens(reader)
            index = ChunkIndex.load_or_build(
                reader,
                cfg.data.index_dir,
                semantic.embedding_model,
                cfg.embedding.model,
                cfg.embedding.batch_size,
                semantic._embedding_lock,
            )
            self._naive_resources = (index, reader)
        index, reader = self._naive_resources
        return NaiveRAG(
            llm_client=client_from_config(cfg.llm, self.pricing),
            index=index,
            reader=reader,
            embedding_model=semantic.embedding_model,
            embedding_lock=semantic._embedding_lock,
            top_k=cfg.naive_rag.top_k,
        )

    def _create_agent(self) -> MeteredAgent:
        # Upstream creates every agent in the main thread before submitting the questions,
        # so the recorder map is only written here and read by the worker that owns the agent.
        agent: MeteredAgent
        if self.run_config.method == "naive_rag":
            recorder = TraceRecorder(agent_has_tools=False)
            agent = self._naive_rag()
        else:
            recorder = TraceRecorder(agent_has_tools=True)
            agent = RecordingAgent(
                llm_client=client_from_config(self.run_config.llm, self.pricing),
                tools=TimedToolRegistry(self._shared_tools, recorder),
                system_prompt=self._system_prompt,
                max_loops=self.run_config.agent.max_loops,
                max_token_budget=self.run_config.agent.max_token_budget,
                verbose=False,
            )
        assert isinstance(agent.llm, MeteredLLMClient)
        agent.llm.observer = recorder.on_llm_call
        self._recorders[agent] = recorder
        return agent

    def _process_one(self, item: dict[str, Any], agent: MeteredAgent) -> dict[str, Any]:
        assert isinstance(agent.llm, MeteredLLMClient)
        if self._abort_reason is None and self.budget is not None and self.budget.exhausted:
            self._abort_reason = self.budget.describe()
        if self._abort_reason is not None:
            raise RunAborted(f"not started: {self._abort_reason}")
        start = time.perf_counter()
        record: dict[str, Any] = super()._process_one(item, agent)
        account_errors = [r.error for r in agent.llm.records if r.account_error]
        if account_errors:
            self._abort_reason = str(account_errors[0])
            raise RunAborted(f"account-level error, not recorded: {self._abort_reason}")
        usage = agent.llm.usage_summary()
        record.update(
            model=self.run_config.llm.model,
            latency_s=round(time.perf_counter() - start, 3),
            llm_calls=usage.calls,
            llm_failed_calls=usage.failed_calls,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cached_tokens=usage.cached_tokens,
            termination=termination_reason(agent.last_result, usage),
            tool_calls=dict(Counter(step["tool_name"] for step in record["trajectory"])),
            llm_errors=[r.error for r in agent.llm.records if r.error],
            citations=[c.model_dump() for c in record_citations(record, self._chunk_refs)],
        )
        if item.get("evidence_chunks"):  # built evalsets carry their gold evidence
            record["evidence_chunks"] = list(item["evidence_chunks"])
        trajectory = self._recorders.pop(agent).build(
            record, self.run_config.method, self.run_config.llm.model
        )
        if self.budget is not None:
            self.budget.charge(float(record.get("total_cost") or 0.0))
        # Written before upstream appends the prediction: after a crash in between, the
        # question is rerun and the loader keeps the last trajectory of each question.
        with self.write_lock, open(self.trajectories_file, "a", encoding="utf-8") as f:
            f.write(trajectory.model_dump_json() + "\n")
        return record

    @property
    def trajectories_file(self) -> Path:
        return Path(self.output_dir) / TRAJECTORIES_FILE

    def _write_run_meta(self) -> None:
        meta_path = Path(self.output_dir) / "run_meta.json"
        fingerprint = config_fingerprint(self.run_config)
        if meta_path.exists():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if existing.get("config_fingerprint") != fingerprint:
                raise RuntimeError(
                    f"{meta_path} was produced with a different config; "
                    "use a new output directory instead of mixing configs"
                )
            return
        cfg = self.run_config
        meta = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "git": git_state(),
            "config_fingerprint": fingerprint,
            "config": cfg.model_dump(mode="json"),
            "tools": self._shared_tools.list_tools(),
            "system_prompt": self._system_prompt,
            "llm_snapshot": self.pricing.resolve(cfg.llm.model),
            "judge_snapshot": self.pricing.resolve(cfg.judge.model),
            "currency": self.pricing.currency,
            "questions_file": str(self.questions_file),
            "questions_sha256": sha256_file(self.questions_file),
            "n_questions": len(self.questions),
            "workers": self.num_workers,
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def run(self) -> None:
        self._write_run_meta()
        super().run()
        if self._abort_reason is not None:
            raise RunAborted(
                f"run stopped: {self._abort_reason}. The questions it hit were not recorded; "
                "fix the account or raise the cost limit, then rerun the same command to resume."
            )
        # Upstream only prints questions whose processing raised; make a gap an error.
        done = self._load_completed_qids()
        missing = [q.get("qid") or q.get("id") for q in self.questions]
        missing = [qid for qid in missing if qid not in done]
        if missing:
            raise RuntimeError(
                f"{len(missing)} questions were not recorded (see 'Error processing' above), "
                f"e.g. {missing[:3]}; rerun the same command to resume"
            )
