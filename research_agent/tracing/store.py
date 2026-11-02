"""Load a run directory's trajectories for review, joined with the judged predictions.

Runs made before the recorder existed (stage 0/1) have no ``trajectories.jsonl``; their
trajectories are converted from ``predictions.jsonl`` on the fly. Converted trajectories
keep everything upstream recorded (tool calls, arguments, results, answer, cost) but have
no per-call fields, no assistant text and no messages, which are never reconstructed:
an approximate conversation must not end up in SFT data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from research_agent.config import RunConfig
from research_agent.eval.summarize import load_predictions
from research_agent.tracing.recorder import tool_call_from_entry
from research_agent.tracing.schema import TRAJECTORIES_FILE, Step, Trajectory

PREDICTIONS_FILE = "predictions.jsonl"


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    method: str
    model: str
    tools: tuple[str, ...]  # tools offered to the agent ("dense_retrieval" for naive RAG)
    recorded: bool  # has trajectories.jsonl


def find_runs(root: str | Path) -> list[Path]:
    """Run directories (containing predictions.jsonl) under ``root``, sorted by path."""
    return sorted(p.parent for p in Path(root).rglob(PREDICTIONS_FILE))


def run_info(run_dir: str | Path) -> RunInfo:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    # Re-validated so runs from before a config field existed get its default.
    cfg = RunConfig.model_validate(meta["config"])
    tools = ("dense_retrieval",) if cfg.method == "naive_rag" else tuple(cfg.agent.tools)
    return RunInfo(
        run_dir=run_dir,
        method=cfg.method,
        model=cfg.llm.model,
        tools=tuple(meta.get("tools") or tools),
        recorded=(run_dir / TRAJECTORIES_FILE).exists(),
    )


def convert_prediction(record: dict[str, Any], info: RunInfo) -> Trajectory:
    """Trajectory from an upstream-format prediction record (no recorder data)."""
    entries = list(record.get("trajectory") or [])
    steps: list[Step] = []
    if entries and entries[0].get("tool_name") == "dense_retrieval":
        retrieval = tool_call_from_entry(entries.pop(0))
        steps.append(Step(index=1, kind="retrieval", tool_calls=[retrieval]))
    # Upstream numbers entries by loop; parallel tool calls share one loop, i.e. one step.
    for _, group in groupby(entries, key=lambda e: e.get("loop")):
        calls = [tool_call_from_entry(e) for e in group]
        steps.append(Step(index=len(steps) + 1, kind="tool_calls", tool_calls=calls))
    termination = record.get("termination", "unknown")
    answer = record.get("pred_answer") or ""
    steps.append(
        Step(
            index=len(steps) + 1,
            kind="answer" if termination in ("answered", "unknown") else "forced_answer",
            thought=answer or None,
        )
    )
    return Trajectory(
        source="converted",
        question_id=str(record["qid"]),
        question=record.get("question", ""),
        gold_answer=record.get("gold_answer", ""),
        method=info.method,
        model=str(record.get("model") or info.model),
        steps=steps,
        final_answer=answer,
        termination_reason=termination,
        total_cost=record.get("total_cost", 0.0),
        prompt_tokens=record.get("input_tokens"),
        completion_tokens=record.get("output_tokens"),
        cached_tokens=record.get("cached_tokens"),
        llm_calls=record.get("llm_calls"),
        latency_s=record.get("latency_s"),
    )


def load_recorded(run_dir: str | Path) -> dict[str, Trajectory]:
    """Recorded trajectories by question id; the last line wins (a resumed question)."""
    path = Path(run_dir) / TRAJECTORIES_FILE
    if not path.exists():
        return {}
    out: dict[str, Trajectory] = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                t = Trajectory.model_validate_json(line)
            except ValidationError as e:
                raise ValueError(f"{path}:{n} is not a valid trajectory") from e
            out[t.question_id] = t
    return out


def load_run(run_dir: str | Path) -> list[Trajectory]:
    """One trajectory per prediction, in prediction order, with the judged outcome."""
    info = run_info(run_dir)
    recorded = load_recorded(run_dir)
    out: list[Trajectory] = []
    for record in load_predictions(run_dir):
        qid = str(record["qid"])
        t = recorded.get(qid) or convert_prediction(record, info)
        updates: dict[str, Any] = {}
        if "llm_accuracy" in record:
            updates["correct"] = bool(record["llm_accuracy"])
        if "contain_accuracy" in record:
            updates["contain_correct"] = bool(record["contain_accuracy"])
        out.append(t.model_copy(update=updates))
    return out
