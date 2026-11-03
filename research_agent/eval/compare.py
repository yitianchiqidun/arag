"""Comparison tables across runs, with a comparability check and paired significance tests.

Rule from the project plan: every row of one table must use the same model and parameters.
``check_comparable`` enforces it from each run's ``run_meta.json``: runs may differ only in
the method and the ablation switches; anything else (model snapshot, temperature, max_loops,
question file, concurrency, ...) is an error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from research_agent.config import RunConfig
from research_agent.eval.metrics import mcnemar_exact_p, paired_diff_ci
from research_agent.eval.summarize import (
    RunSummary,
    load_predictions,
    rows_to_csv,
    summarize_run,
    to_csv_row,
)

VARIANT_KEYS = ("method", "agent.tools", "agent.search_result", "naive_rag.", "routing.")


class ComparedRun(BaseModel):
    name: str
    run_dir: str
    summary: RunSummary
    differences: dict[str, Any]  # values of the keys that vary across the compared runs
    commit: str
    # Paired comparison with the reference run on LLM-Acc (over all questions).
    delta_llm_acc: float
    ref_only_correct: int
    run_only_correct: int
    mcnemar_p: float
    # 95% CI of delta_llm_acc (Newcombe paired interval), so "not significant" can be read
    # as "the difference is within [ci_low, ci_high]" rather than "no difference".
    ci_low: float
    ci_high: float


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): value}


def comparable_settings(run_meta: dict[str, Any]) -> dict[str, Any]:
    """Everything that must match across a table, as flat ``dotted.key -> value``.

    The config is re-validated so runs made before a field existed get its default.
    """
    cfg = RunConfig.model_validate(run_meta["config"]).model_dump(mode="json")
    flat = _flatten(cfg)
    for key in ("llm_snapshot", "judge_snapshot", "questions_sha256", "workers", "currency"):
        flat[key] = run_meta.get(key)
    return flat


def check_comparable(metas: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per run, the variant settings that differ across runs; raises on any other difference."""
    settings = {name: comparable_settings(meta) for name, meta in metas.items()}
    keys = sorted({k for s in settings.values() for k in s})
    differing = [
        k
        for k in keys
        if len({json.dumps(s.get(k), sort_keys=True) for s in settings.values()}) > 1
    ]
    forbidden = [k for k in differing if not k.startswith(VARIANT_KEYS)]
    if forbidden:
        details = {k: {name: s.get(k) for name, s in settings.items()} for k in forbidden}
        raise ValueError(f"runs are not comparable, they differ in: {details}")
    return {name: {k: s.get(k) for k in differing} for name, s in settings.items()}


def _correct_by_qid(run_dir: str | Path) -> dict[str, bool]:
    return {str(r["qid"]): bool(r.get("llm_accuracy")) for r in load_predictions(run_dir)}


def compare_runs(runs: dict[str, str | Path], reference: str) -> list[ComparedRun]:
    if reference not in runs:
        raise ValueError(f"reference {reference!r} is not among the runs {list(runs)}")
    metas = {
        name: json.loads((Path(d) / "run_meta.json").read_text(encoding="utf-8"))
        for name, d in runs.items()
    }
    differences = check_comparable(metas)
    correct = {name: _correct_by_qid(d) for name, d in runs.items()}
    ref = correct[reference]
    out: list[ComparedRun] = []
    for name, run_dir in runs.items():
        if set(correct[name]) != set(ref):
            raise ValueError(f"{name} and {reference} answered different question sets")
        ref_only = sum(ref[q] and not correct[name][q] for q in ref)
        run_only = sum(correct[name][q] and not ref[q] for q in ref)
        both = sum(ref[q] and correct[name][q] for q in ref)
        ci_low, ci_high = paired_diff_ci(
            both, run_only, ref_only, len(ref) - both - run_only - ref_only
        )
        summary = summarize_run(run_dir)
        out.append(
            ComparedRun(
                name=name,
                run_dir=str(run_dir),
                summary=summary,
                differences=differences[name],
                commit=str(metas[name].get("git", {}).get("commit", ""))[:7],
                delta_llm_acc=(run_only - ref_only) / len(ref),
                ref_only_correct=ref_only,
                run_only_correct=run_only,
                mcnemar_p=mcnemar_exact_p(ref_only, run_only),
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
    return out


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def to_markdown(rows: list[ComparedRun], reference: str) -> str:
    first = rows[0].summary
    hops = list(first.n_by_hop)
    acc_head = ["Variant", "LLM-Acc", f"Δ vs {reference}", "95% CI of Δ"]
    acc_head += ["McNemar p (ref-only / run-only)"]
    acc_head += ["Cont-Acc"] + [f"{h} (n={first.n_by_hop[h]})" for h in hops]
    lines = [
        (
            f"Agent model `{first.model}`, judge `{first.judge_model}`, n = {first.n}. "
            "LLM-Acc and Cont-Acc are over all questions; the McNemar test and the confidence "
            f"interval (Newcombe, paired) compare each run with `{reference}` question by question."
        ),
        "",
        "| " + " | ".join(acc_head) + " |",
        "|" + " --- |" * len(acc_head),
    ]
    for r in rows:
        s = r.summary
        is_ref = r.name == reference
        cells = [
            f"**{r.name}**" if is_ref else r.name,
            f"{s.llm_acc_all:.1%}",
            "—" if is_ref else f"{r.delta_llm_acc * 100:+.1f} pp",
            "—" if is_ref else f"[{r.ci_low * 100:+.1f}, {r.ci_high * 100:+.1f}]",
            "—" if is_ref else f"{r.mcnemar_p:.3f} ({r.ref_only_correct} / {r.run_only_correct})",
            f"{s.cont_acc_all:.1%}",
        ] + [_pct(s.llm_acc_by_hop.get(h)) for h in hops]
        lines.append("| " + " | ".join(cells) + " |")

    cost_head = [
        "Variant",
        "Loops",
        "Tool calls (kw / sem / read)",
        "Repeat read",
        "Repeat search",
        "Hit max loops",
        "Input tokens",
        "Cache hit",
        "Retrieved tokens",
        "Latency P50 / P95 (s)",
        f"Agent cost / 100 q ({first.currency})",
    ]
    lines += ["", "| " + " | ".join(cost_head) + " |", "|" + " --- |" * len(cost_head)]
    for r in rows:
        s = r.summary
        by_tool = s.tool_calls_by_tool
        if "dense_retrieval" in by_tool:
            tools = f"dense retrieval × {by_tool['dense_retrieval']:.0f}"
        else:
            per_tool = " / ".join(
                f"{by_tool.get(t, 0):.2f}"
                for t in ("keyword_search", "semantic_search", "read_chunk")
            )
            tools = f"{s.avg_tool_calls:.2f} ({per_tool})"
        cells = [
            r.name,
            f"{s.avg_loops:.2f}",
            tools,
            _pct(s.repeat_read_rate),
            _pct(s.repeat_search_rate),
            _pct(s.max_loops_rate),
            f"{s.avg_input_tokens:,.0f}",
            _pct(s.cache_hit_rate),
            f"{s.avg_retrieved_tokens:,.0f}",
            f"{s.p50_latency_s:.1f} / {s.p95_latency_s:.1f}",
            f"{s.agent_cost_per_100:.2f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    if any(r.summary.citations for r in rows):
        lines += ["", *citation_table(rows)]

    lines += ["", "Settings that differ between rows (everything else is identical):", ""]
    for r in rows:
        diff = ", ".join(f"`{k}={json.dumps(v)}`" for k, v in r.differences.items()) or "—"
        lines.append(f"- **{r.name}** (`{r.run_dir}`, commit `{r.commit}`): {diff}")
    return "\n".join(lines)


def citation_table(rows: list[ComparedRun]) -> list[str]:
    """Answer citations (D25): how often answers cite chunks and what the agent saw of them."""
    head = [
        "Variant",
        "Answers citing a chunk",
        "Citations read in full",
        "Citations seen only as a snippet",
        "Citations never shown",
        "Answers citing a gold evidence chunk",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for r in rows:
        c = r.summary.citations
        if c is None:
            cells = [r.name] + ["—"] * (len(head) - 1)
        else:
            evidence = f"{_pct(c.evidence_cited)} (n={c.evidence_n})" if c.evidence_n else "—"
            cells = [
                r.name,
                _pct(c.answers_citing),
                _pct(c.cited_read),
                _pct(c.cited_snippet),
                str(c.cited_unseen),
                evidence,
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def to_csv(rows: list[ComparedRun]) -> str:
    records = []
    for r in rows:
        record: dict[str, Any] = {"variant": r.name}
        record.update({f"setting.{k}": json.dumps(v) for k, v in r.differences.items()})
        record.update(
            delta_llm_acc=r.delta_llm_acc,
            ref_only_correct=r.ref_only_correct,
            run_only_correct=r.run_only_correct,
            mcnemar_p=r.mcnemar_p,
            delta_ci_low=r.ci_low,
            delta_ci_high=r.ci_high,
            commit=r.commit,
        )
        record.update(to_csv_row(r.summary))
        records.append(record)
    return rows_to_csv(records)


def write_comparison(rows: list[ComparedRun], reference: str, out_prefix: str | Path) -> str:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    markdown = to_markdown(rows, reference)
    prefix.with_suffix(".md").write_text(markdown + "\n", encoding="utf-8")
    prefix.with_suffix(".csv").write_text(to_csv(rows), encoding="utf-8")
    return markdown
