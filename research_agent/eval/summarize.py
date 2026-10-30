"""Aggregate a run directory (predictions + judge meta) into one metrics record."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class RunSummary(BaseModel):
    run_dir: str
    model: str
    judge_model: str
    currency: str
    n: int
    answered: int
    errors: int
    # Upstream eval.py divides by answered (non-empty) predictions; *_all divide by n.
    llm_acc: float
    llm_acc_all: float
    cont_acc: float
    cont_acc_all: float
    avg_loops: float
    avg_llm_calls: float
    avg_tool_calls: float
    tool_calls_by_tool: dict[str, float]
    avg_input_tokens: float
    avg_output_tokens: float
    avg_cached_tokens: float
    cache_hit_rate: float
    avg_retrieved_tokens: float
    avg_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    termination: dict[str, int]
    agent_cost: float
    judge_cost: float

    @property
    def total_cost(self) -> float:
        return self.agent_cost + self.judge_cost


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _percentile(values: list[float], q: float) -> float:
    if len(values) < 2:
        return values[0] if values else 0.0
    return statistics.quantiles(values, n=100, method="inclusive")[round(q * 100) - 1]


def _is_error(record: dict[str, Any]) -> bool:
    pred = record.get("pred_answer")
    return bool(record.get("error")) or (isinstance(pred, str) and pred.startswith("Error:"))


def summarize_run(run_dir: str | Path) -> RunSummary:
    run_dir = Path(run_dir)
    records = _load_jsonl(run_dir / "predictions.jsonl")
    if not records or "llm_accuracy" not in records[0]:
        raise ValueError(f"{run_dir} has no judged predictions; run the judge first")
    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    judge_meta = json.loads((run_dir / "judge_meta.json").read_text(encoding="utf-8"))

    n = len(records)
    answered = [r for r in records if r.get("status") == "answered"]
    tool_totals: Counter[str] = Counter()
    for r in records:
        tool_totals.update(r.get("tool_calls", {}))
    latencies = [float(r["latency_s"]) for r in records]
    input_tokens = sum(r["input_tokens"] for r in records)
    cached_tokens = sum(r["cached_tokens"] for r in records)

    def mean(key: str) -> float:
        return sum(float(r.get(key, 0) or 0) for r in records) / n

    def acc(key: str, denom: int) -> float:
        return sum(float(r[key]) for r in records) / denom if denom else 0.0

    return RunSummary(
        run_dir=str(run_dir),
        model=run_meta["llm_snapshot"],
        judge_model=judge_meta["judge_snapshot"],
        currency=run_meta["currency"],
        n=n,
        answered=len(answered),
        errors=sum(_is_error(r) for r in records),
        llm_acc=acc("llm_accuracy", len(answered)),
        llm_acc_all=acc("llm_accuracy", n),
        cont_acc=acc("contain_accuracy", len(answered)),
        cont_acc_all=acc("contain_accuracy", n),
        avg_loops=mean("loops"),
        avg_llm_calls=mean("llm_calls"),
        avg_tool_calls=sum(tool_totals.values()) / n,
        tool_calls_by_tool={k: v / n for k, v in sorted(tool_totals.items())},
        avg_input_tokens=input_tokens / n,
        avg_output_tokens=mean("output_tokens"),
        avg_cached_tokens=cached_tokens / n,
        cache_hit_rate=cached_tokens / input_tokens if input_tokens else 0.0,
        avg_retrieved_tokens=mean("total_retrieved_tokens"),
        avg_latency_s=statistics.fmean(latencies),
        p50_latency_s=statistics.median(latencies),
        p95_latency_s=_percentile(latencies, 0.95),
        termination=dict(Counter(r.get("termination", "unknown") for r in records)),
        agent_cost=sum(float(r.get("total_cost", 0) or 0) for r in records),
        judge_cost=float(judge_meta["usage"]["cost"]),
    )


def to_markdown(s: RunSummary) -> str:
    tools = ", ".join(f"{k} {v:.2f}" for k, v in s.tool_calls_by_tool.items())
    rows = [
        ("Agent model / judge model", f"{s.model} / {s.judge_model}"),
        ("Questions (answered / errors)", f"{s.n} ({s.answered} / {s.errors})"),
        ("LLM-Acc (upstream def / over all questions)", f"{s.llm_acc:.1%} / {s.llm_acc_all:.1%}"),
        (
            "Cont-Acc (upstream def / over all questions)",
            f"{s.cont_acc:.1%} / {s.cont_acc_all:.1%}",
        ),
        ("Avg loops / LLM calls", f"{s.avg_loops:.2f} / {s.avg_llm_calls:.2f}"),
        ("Avg tool calls (by tool)", f"{s.avg_tool_calls:.2f} ({tools})"),
        ("Avg input / output tokens", f"{s.avg_input_tokens:,.0f} / {s.avg_output_tokens:,.0f}"),
        ("Implicit cache hit rate", f"{s.cache_hit_rate:.1%}"),
        ("Avg retrieved tokens (upstream metric)", f"{s.avg_retrieved_tokens:,.0f}"),
        (
            "Latency avg / P50 / P95 (s)",
            f"{s.avg_latency_s:.1f} / {s.p50_latency_s:.1f} / {s.p95_latency_s:.1f}",
        ),
        ("Termination reasons", ", ".join(f"{k} {v}" for k, v in sorted(s.termination.items()))),
        (
            f"Cost ({s.currency}) agent + judge = total",
            f"{s.agent_cost:.3f} + {s.judge_cost:.3f} = {s.total_cost:.3f}",
        ),
    ]
    lines = ["| Metric | Value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines)
