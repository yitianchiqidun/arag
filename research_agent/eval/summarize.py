"""Aggregate a run directory (predictions + judge meta) into one metrics record."""

from __future__ import annotations

import csv
import io
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from research_agent.eval.citations import CitationStats, citation_stats, load_chunk_refs
from research_agent.eval.metrics import RepeatCounts, hop_group, repeat_counts


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
    # Share of full-text chunk reads that re-requested a chunk already shown (None: no reads).
    repeat_read_rate: float | None
    # Share of search calls identical to an earlier one in the same question.
    repeat_search_rate: float | None
    avg_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    termination: dict[str, int]
    max_loops_rate: float
    # LLM-Acc over all questions of each hop count (MuSiQue ids), and the group sizes.
    llm_acc_by_hop: dict[str, float]
    n_by_hop: dict[str, int]
    agent_cost: float
    judge_cost: float
    # None when the run's chunks file is not available locally.
    citations: CitationStats | None = None
    # method routed: questions and LLM-Acc per channel, channels per question type, and the
    # router's share of agent_cost.
    n_by_channel: dict[str, int] = {}
    llm_acc_by_channel: dict[str, float] = {}
    channel_by_group: dict[str, dict[str, int]] = {}
    route_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.agent_cost + self.judge_cost

    @property
    def agent_cost_per_100(self) -> float:
        return self.agent_cost / self.n * 100 if self.n else 0.0


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


def load_predictions(run_dir: str | Path) -> list[dict[str, Any]]:
    return _load_jsonl(Path(run_dir) / "predictions.jsonl")


def summarize_run(run_dir: str | Path) -> RunSummary:
    run_dir = Path(run_dir)
    records = load_predictions(run_dir)
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
    repeats = RepeatCounts()
    for r in records:
        c = repeat_counts(r.get("trajectory") or [])
        repeats.full_text_chunks += c.full_text_chunks
        repeats.repeated_chunks += c.repeated_chunks
        repeats.searches += c.searches
        repeats.repeated_searches += c.repeated_searches
    by_hop: defaultdict[str, list[float]] = defaultdict(list)
    for r in records:
        group = hop_group(str(r.get("qid", "")))
        if group:
            by_hop[group].append(float(r["llm_accuracy"]))
    termination = Counter(r.get("termination", "unknown") for r in records)
    by_channel: defaultdict[str, list[float]] = defaultdict(list)
    channel_by_group: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for r in records:
        if r.get("channel"):
            by_channel[r["channel"]].append(float(r["llm_accuracy"]))
            channel_by_group[hop_group(str(r.get("qid", ""))) or "all"][r["channel"]] += 1
    chunks_file = run_meta.get("config", {}).get("data", {}).get("chunks_file")
    citations = None
    if chunks_file and Path(chunks_file).exists():
        citations = citation_stats(records, load_chunk_refs(chunks_file))

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
        repeat_read_rate=(
            repeats.repeated_chunks / repeats.full_text_chunks if repeats.full_text_chunks else None
        ),
        repeat_search_rate=(
            repeats.repeated_searches / repeats.searches if repeats.searches else None
        ),
        avg_latency_s=statistics.fmean(latencies),
        p50_latency_s=statistics.median(latencies),
        p95_latency_s=_percentile(latencies, 0.95),
        termination=dict(termination),
        max_loops_rate=termination["max_loops"] / n,
        llm_acc_by_hop={k: sum(v) / len(v) for k, v in sorted(by_hop.items())},
        n_by_hop={k: len(v) for k, v in sorted(by_hop.items())},
        agent_cost=sum(float(r.get("total_cost", 0) or 0) for r in records),
        judge_cost=float(judge_meta["usage"]["cost"]),
        citations=citations,
        n_by_channel={k: len(v) for k, v in sorted(by_channel.items())},
        llm_acc_by_channel={k: sum(v) / len(v) for k, v in sorted(by_channel.items())},
        channel_by_group={k: dict(sorted(v.items())) for k, v in sorted(channel_by_group.items())},
        route_cost=sum(float(r.get("route_cost", 0) or 0) for r in records),
    )


def _rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _citations(c: CitationStats | None) -> str:
    if c is None:
        return "—"
    text = f"{c.answers_citing:.1%} / {_rate(c.cited_read)} / {c.cited_unseen}"
    if c.evidence_n:
        text += f"; answers citing a gold evidence chunk {_rate(c.evidence_cited)}"
    return text


def _channel_rows(s: RunSummary) -> list[tuple[str, str]]:
    if not s.n_by_channel:
        return []
    acc = ", ".join(f"{k} {s.n_by_channel[k]} ({v:.1%})" for k, v in s.llm_acc_by_channel.items())
    groups = "; ".join(
        f"{g}: " + ", ".join(f"{k} {n}" for k, n in c.items())
        for g, c in s.channel_by_group.items()
    )
    return [
        ("Channels: questions (LLM-Acc)", acc),
        ("Channels by question type", groups),
        (f"Router cost ({s.currency}, included in agent cost)", f"{s.route_cost:.4f}"),
    ]


def to_markdown(s: RunSummary) -> str:
    tools = ", ".join(f"{k} {v:.2f}" for k, v in s.tool_calls_by_tool.items())
    hops = " / ".join(f"{k} {v:.0%} (n={s.n_by_hop[k]})" for k, v in s.llm_acc_by_hop.items())
    rows = [
        ("Agent model / judge model", f"{s.model} / {s.judge_model}"),
        ("Questions (answered / errors)", f"{s.n} ({s.answered} / {s.errors})"),
        ("LLM-Acc (upstream def / over all questions)", f"{s.llm_acc:.1%} / {s.llm_acc_all:.1%}"),
        (
            "Cont-Acc (upstream def / over all questions)",
            f"{s.cont_acc:.1%} / {s.cont_acc_all:.1%}",
        ),
        ("LLM-Acc by hop count / question type", hops or "—"),
        ("Avg loops / LLM calls", f"{s.avg_loops:.2f} / {s.avg_llm_calls:.2f}"),
        ("Avg tool calls (by tool)", f"{s.avg_tool_calls:.2f} ({tools})"),
        (
            "Repeat read rate / repeat search rate",
            f"{_rate(s.repeat_read_rate)} / {_rate(s.repeat_search_rate)}",
        ),
        ("Avg input / output tokens", f"{s.avg_input_tokens:,.0f} / {s.avg_output_tokens:,.0f}"),
        ("Implicit cache hit rate", f"{s.cache_hit_rate:.1%}"),
        ("Avg retrieved tokens (upstream metric)", f"{s.avg_retrieved_tokens:,.0f}"),
        (
            "Latency avg / P50 / P95 (s)",
            f"{s.avg_latency_s:.1f} / {s.p50_latency_s:.1f} / {s.p95_latency_s:.1f}",
        ),
        ("Termination reasons", ", ".join(f"{k} {v}" for k, v in sorted(s.termination.items()))),
        ("Hit max loops", f"{s.max_loops_rate:.1%}"),
        (
            "Answers citing a chunk / citations read in full / citations never shown",
            _citations(s.citations),
        ),
        *_channel_rows(s),
        (
            f"Cost ({s.currency}) agent + judge = total",
            f"{s.agent_cost:.3f} + {s.judge_cost:.3f} = {s.total_cost:.3f}",
        ),
    ]
    lines = ["| Metric | Value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines)


def to_csv_row(s: RunSummary) -> dict[str, Any]:
    """Flat record: nested dicts become ``<field>.<key>`` columns."""
    row: dict[str, Any] = {}
    for key, value in s.model_dump().items():
        if isinstance(value, dict):
            row.update({f"{key}.{k}": v for k, v in value.items()})
        else:
            row[key] = value
    row["total_cost"] = s.total_cost
    row["agent_cost_per_100"] = s.agent_cost_per_100
    return row


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    columns: list[str] = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def write_summary(run_dir: str | Path) -> RunSummary:
    """Summarize a judged run and write ``summary.{json,md,csv}`` into it."""
    summary = summarize_run(run_dir)
    out = Path(run_dir)
    (out / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n", "utf-8")
    (out / "summary.md").write_text(to_markdown(summary) + "\n", "utf-8")
    (out / "summary.csv").write_text(rows_to_csv([to_csv_row(summary)]), "utf-8")
    return summary
