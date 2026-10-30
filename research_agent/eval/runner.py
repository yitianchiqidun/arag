"""Batch-run A-RAG with metering, reusing the upstream BatchRunner.

Upstream ``scripts/batch_runner.py`` records answers, cost and the tool trajectory, but not
token counts, latency or why the loop stopped. ``MeteredBatchRunner`` subclasses it and only
swaps the LLM client / agent class and enriches each prediction record.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from arag import BaseAgent
from research_agent.config import RunConfig
from research_agent.eval.subsets import sha256_file
from research_agent.llm.client import MeteredLLMClient, UsageSummary, client_from_config
from research_agent.llm.pricing import PricingTable
from research_agent.upstream import REPO_ROOT, load_script

_BatchRunner: type = load_script("batch_runner").BatchRunner


class RecordingAgent(BaseAgent):
    """BaseAgent that keeps the raw result of the last ``run`` (upstream drops some flags)."""

    last_result: dict[str, Any] | None = None

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

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


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
    ) -> None:
        self.run_config = run_config
        self.pricing = pricing
        super().__init__(
            config=run_config.to_arag_config(),
            questions_file=questions_file,
            output_dir=output_dir,
            limit=limit,
            num_workers=num_workers,
            verbose=False,
        )

    def _create_agent(self) -> RecordingAgent:
        return RecordingAgent(
            llm_client=client_from_config(self.run_config.llm, self.pricing),
            tools=self._shared_tools,
            system_prompt=self._system_prompt,
            max_loops=self.run_config.agent.max_loops,
            max_token_budget=self.run_config.agent.max_token_budget,
            verbose=False,
        )

    def _process_one(self, item: dict[str, Any], agent: BaseAgent) -> dict[str, Any]:
        assert isinstance(agent, RecordingAgent) and isinstance(agent.llm, MeteredLLMClient)
        start = time.perf_counter()
        record: dict[str, Any] = super()._process_one(item, agent)
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
        )
        return record

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
