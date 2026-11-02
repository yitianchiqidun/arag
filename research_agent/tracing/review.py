"""Pure helpers behind the trajectory viewer: filters, suggested tags and context growth.

Kept out of the Streamlit script so they can be unit-tested and reused in stage 4.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from research_agent.eval.metrics import repeat_counts
from research_agent.tracing.labels import Label
from research_agent.tracing.schema import Trajectory


class TrajectoryFilter(BaseModel):
    correctness: Literal["all", "correct", "wrong", "unjudged"] = "all"
    min_steps: int | None = None
    max_steps: int | None = None
    min_tool_calls: int | None = None
    max_tool_calls: int | None = None
    terminations: list[str] = []  # empty = any
    label_state: Literal["all", "labelled", "unlabelled"] = "all"
    tag: str | None = None
    text: str = ""  # substring of the question id, question or answers (case-insensitive)


def step_count(t: Trajectory) -> int:
    """LLM steps (a naive RAG retrieval is not a step the model decided to take)."""
    return sum(s.kind != "retrieval" for s in t.steps)


def _within(value: int, low: int | None, high: int | None) -> bool:
    return (low is None or value >= low) and (high is None or value <= high)


def matches(t: Trajectory, label: Label | None, f: TrajectoryFilter) -> bool:
    if f.correctness == "correct" and t.correct is not True:
        return False
    if f.correctness == "wrong" and t.correct is not False:
        return False
    if f.correctness == "unjudged" and t.correct is not None:
        return False
    if not _within(step_count(t), f.min_steps, f.max_steps):
        return False
    if not _within(t.tool_call_count, f.min_tool_calls, f.max_tool_calls):
        return False
    if f.terminations and t.termination_reason not in f.terminations:
        return False
    if f.label_state == "labelled" and label is None:
        return False
    if f.label_state == "unlabelled" and label is not None:
        return False
    if f.tag is not None and (label is None or f.tag not in label.tags):
        return False
    if f.text:
        haystack = " ".join([t.question_id, t.question, t.final_answer, str(t.gold_answer)])
        if f.text.lower() not in haystack.lower():
            return False
    return True


def apply_filter(
    trajectories: list[Trajectory], labels: dict[str, Label], f: TrajectoryFilter
) -> list[Trajectory]:
    return [t for t in trajectories if matches(t, labels.get(t.question_id), f)]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def suggest_tags(t: Trajectory, available_tools: tuple[str, ...]) -> list[str]:
    """Heuristic tag suggestions for a wrong answer; the reviewer decides.

    The retrieval heuristics look for the gold answer string in the tool results, so they
    miss paraphrased evidence and are only offered for A-RAG trajectories.
    """
    suggestions: list[str] = []
    if t.termination_reason == "max_loops":
        suggestions.append("max_loops")
    if t.termination_reason in ("llm_error", "exception") or any(
        s.kind == "llm_error" for s in t.steps
    ):
        suggestions.append("api_error")
    calls = [c for s in t.steps if s.kind != "retrieval" for c in s.tool_calls]
    entries = [{"tool_name": c.tool_name, "arguments": c.tool_args, **c.log} for c in calls]
    counts = repeat_counts(entries)
    if counts.repeated_chunks or counts.repeated_searches:
        suggestions.append("repeated_search_or_read")

    if t.correct is False and t.method == "arag":
        golds = [t.gold_answer] if isinstance(t.gold_answer, str) else t.gold_answer
        golds = [_norm(g) for g in golds if g.strip()]

        def found(results: list[str]) -> bool:
            return any(g in _norm(r) for g in golds for r in results)

        all_results = [c.tool_result for c in calls]
        read_results = [c.tool_result for c in calls if c.tool_name == "read_chunk"]
        if golds and not found(all_results):
            suggestions.append("not_retrieved")
        elif golds and "read_chunk" in available_tools and not found(read_results):
            suggestions.append("retrieved_not_read")
    return suggestions


def context_growth(t: Trajectory) -> list[dict[str, Any]]:
    """Per LLM call: tokens billed (cached / uncached) and characters sent per role."""
    rows: list[dict[str, Any]] = []
    for s in t.steps:
        if s.llm is None or s.llm.error is not None:
            continue
        row: dict[str, Any] = {
            "step": s.index,
            "prompt_tokens": s.llm.prompt_tokens,
            "cached_tokens": s.llm.cached_tokens,
            "uncached_tokens": s.llm.prompt_tokens - s.llm.cached_tokens,
            "messages": s.llm.context.count,
        }
        for role in ("system", "user", "assistant", "tool"):
            row[f"chars_{role}"] = s.llm.context.chars.get(role, 0)
        rows.append(row)
    return rows
