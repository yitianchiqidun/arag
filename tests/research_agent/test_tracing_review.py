from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from research_agent.tracing.labels import DEFAULT_TAXONOMY, LabelStore, Taxonomy
from research_agent.tracing.review import (
    TrajectoryFilter,
    apply_filter,
    context_growth,
    step_count,
    suggest_tags,
)
from research_agent.tracing.schema import (
    LLMCall,
    MessageStats,
    Step,
    ToolCall,
    Trajectory,
)
from research_agent.tracing.store import convert_prediction, find_runs, load_run, run_info

CONFIG = {
    "llm": {"model": "m"},
    "judge": {"model": "j"},
    "data": {"chunks_file": "c.json", "index_dir": "idx"},
}


def entry(loop: int, tool: str, args: dict[str, Any], result: str, **log: Any) -> dict[str, Any]:
    return {"loop": loop, "tool_name": tool, "arguments": args, "tool_result": result, **log}


def prediction(qid: str, **kwargs: Any) -> dict[str, Any]:
    base = {
        "qid": qid,
        "question": f"Question {qid}?",
        "gold_answer": "Paris",
        "pred_answer": "Paris",
        "total_cost": 0.01,
        "termination": "answered",
        "trajectory": [],
    }
    return {**base, **kwargs}


def write_run(run_dir: Path, records: list[dict[str, Any]], config: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(json.dumps({"config": config}))
    lines = "".join(json.dumps(r) + "\n" for r in records)
    (run_dir / "predictions.jsonl").write_text(lines)
    return run_dir


def recorded(qid: str, answer: str) -> Trajectory:
    return Trajectory(
        source="recorded",
        question_id=qid,
        question="?",
        gold_answer="Paris",
        method="arag",
        model="m",
        steps=[Step(index=1, kind="answer", thought=answer)],
        final_answer=answer,
        termination_reason="answered",
        total_cost=0.0,
        messages=[{"role": "user", "content": "?"}],
    )


def test_convert_groups_parallel_calls_and_marks_forced_answers(tmp_path: Path) -> None:
    run = write_run(tmp_path / "run", [], CONFIG)
    record = prediction(
        "q1",
        termination="max_loops",
        trajectory=[
            entry(1, "keyword_search", {"keywords": ["a"]}, "Chunk ID: 1", chunks_found=1),
            entry(2, "keyword_search", {"keywords": ["b"]}, "Chunk ID: 2", chunks_found=1),
            entry(2, "read_chunk", {"chunk_ids": ["2"]}, "text", new_chunks_count=1),
        ],
    )

    t = convert_prediction(record, run_info(run))

    assert t.source == "converted" and t.messages is None
    assert [s.kind for s in t.steps] == ["tool_calls", "tool_calls", "forced_answer"]
    assert [len(s.tool_calls) for s in t.steps] == [1, 2, 0]
    assert t.steps[1].tool_calls[1].log == {"new_chunks_count": 1}
    assert t.steps[2].thought == "Paris" and step_count(t) == 3 and t.tool_call_count == 3


def test_convert_naive_rag(tmp_path: Path) -> None:
    run = write_run(tmp_path / "run", [], {**CONFIG, "method": "naive_rag"})
    dense = entry(1, "dense_retrieval", {"query": "?", "top_k": 5}, "[]", retrieved_tokens=9)
    info = run_info(run)

    t = convert_prediction(prediction("q1", trajectory=[dense]), info)

    assert info.tools == ("dense_retrieval",) and t.method == "naive_rag"
    assert [s.kind for s in t.steps] == ["retrieval", "answer"]
    assert step_count(t) == 1 and t.tool_call_count == 0


def test_load_run_prefers_recorded_and_joins_judgement(tmp_path: Path) -> None:
    records = [
        prediction("q1", llm_accuracy=1.0, contain_accuracy=1.0),
        prediction("q2", llm_accuracy=0.0, contain_accuracy=1.0),
    ]
    run = write_run(tmp_path / "run", records, CONFIG)
    # q1 was resumed after a crash: two lines, the last one wins. q2 predates the recorder.
    lines = [recorded("q1", "old").model_dump_json(), recorded("q1", "Paris").model_dump_json()]
    (run / "trajectories.jsonl").write_text("\n".join(lines) + "\n")

    t1, t2 = load_run(run)

    assert (t1.source, t1.final_answer, t1.correct) == ("recorded", "Paris", True)
    assert (t2.source, t2.correct, t2.contain_correct) == ("converted", False, True)
    assert find_runs(tmp_path) == [run] and run_info(run).recorded


def test_load_run_rejects_a_corrupt_trajectory_line(tmp_path: Path) -> None:
    run = write_run(tmp_path / "run", [prediction("q1")], CONFIG)
    (run / "trajectories.jsonl").write_text('{"question_id": "q1"}\n')
    with pytest.raises(ValueError, match="trajectories.jsonl:1"):
        load_run(run)


def test_taxonomy_file_is_valid() -> None:
    taxonomy = Taxonomy.from_yaml(DEFAULT_TAXONOMY)
    assert "max_loops" in taxonomy.ids and taxonomy.label("max_loops") == "超出最大步数"
    assert taxonomy.label("dropped_tag") == "dropped_tag"


def test_label_store_round_trip(tmp_path: Path) -> None:
    store = LabelStore(tmp_path, Taxonomy.from_yaml())

    label = store.set("q1", ["max_loops", "not_retrieved", "max_loops"], "  stuck  ")
    assert label is not None and label.tags == ["not_retrieved", "max_loops"]
    assert label.note == "stuck"
    store.set("q2", [], "note only")
    assert set(LabelStore(tmp_path, Taxonomy.from_yaml()).load().labels) == {"q1", "q2"}

    assert store.set("q1", [], "") is None  # clearing removes the entry
    assert store.get("q1") is None and store.get("q2") is not None
    with pytest.raises(ValueError, match="unknown tags"):
        store.set("q3", ["no_such_tag"])
    assert [p.name for p in tmp_path.iterdir()] == ["labels.json"]  # no temp files left


def tool(name: str, result: str, **log: Any) -> ToolCall:
    args = {"chunk_ids": ["1"]} if name == "read_chunk" else {"keywords": ["x"]}
    return ToolCall(tool_name=name, tool_args=args, tool_result=result, log=log)


def trajectory(steps: list[Step], correct: bool | None = False, **kwargs: Any) -> Trajectory:
    base: dict[str, Any] = {
        "source": "recorded",
        "question_id": "q",
        "question": "Where?",
        "gold_answer": "Paris",
        "method": "arag",
        "model": "m",
        "final_answer": "Lyon",
        "termination_reason": "answered",
        "total_cost": 0.0,
        "correct": correct,
    }
    return Trajectory(steps=steps, **{**base, **kwargs})


TOOLS = ("keyword_search", "read_chunk", "semantic_search")


def test_suggest_tags() -> None:
    snippet_only = trajectory(
        [
            Step(index=1, kind="tool_calls", tool_calls=[tool("keyword_search", "...PARIS...")]),
            Step(index=2, kind="tool_calls", tool_calls=[tool("keyword_search", "again")]),
            Step(index=3, kind="answer", thought="Lyon"),
        ]
    )
    assert suggest_tags(snippet_only, TOOLS) == ["repeated_search_or_read", "retrieved_not_read"]

    read_twice = [
        Step(index=1, kind="tool_calls", tool_calls=[tool("read_chunk", "x", already_read_count=1)])
    ]
    forced = trajectory(read_twice, termination_reason="max_loops")
    assert suggest_tags(forced, TOOLS) == ["max_loops", "repeated_search_or_read", "not_retrieved"]

    # Correct answers only get the process-based suggestions; naive RAG none of the retrieval.
    assert suggest_tags(trajectory(read_twice, correct=True), TOOLS) == ["repeated_search_or_read"]
    errored = trajectory([Step(index=1, kind="llm_error")], termination_reason="llm_error")
    assert suggest_tags(errored, TOOLS) == ["api_error", "not_retrieved"]
    naive = trajectory([Step(index=1, kind="answer")], method="naive_rag")
    assert suggest_tags(naive, ("dense_retrieval",)) == []


def llm(prompt: int, cached: int, chars: dict[str, int]) -> LLMCall:
    return LLMCall(
        latency_ms=1,
        attempts=1,
        prompt_tokens=prompt,
        completion_tokens=1,
        cached_tokens=cached,
        cost=0,
        tools_offered=True,
        context=MessageStats(count=len(chars), chars=chars),
    )


def test_context_growth_and_filters(tmp_path: Path) -> None:
    t = trajectory(
        [
            Step(index=1, kind="tool_calls", llm=llm(100, 0, {"system": 50, "user": 10})),
            Step(index=2, kind="answer", llm=llm(300, 100, {"system": 50, "tool": 400})),
        ],
        question_id="q1",
    )
    rows = context_growth(t)
    assert [(r["step"], r["uncached_tokens"], r["chars_tool"]) for r in rows] == [
        (1, 100, 0),
        (2, 200, 400),
    ]
    assert t.peak_prompt_tokens == 300

    right = t.model_copy(update={"question_id": "q2", "correct": True})
    store = LabelStore(tmp_path, Taxonomy.from_yaml())
    store.set("q1", ["not_retrieved"])
    labels = store.load().labels
    pool = [t, right]

    def ids(**kwargs: Any) -> list[str]:
        return [x.question_id for x in apply_filter(pool, labels, TrajectoryFilter(**kwargs))]

    assert ids() == ["q1", "q2"]
    assert ids(correctness="wrong") == ["q1"] and ids(correctness="unjudged") == []
    assert ids(min_steps=3) == [] and ids(max_tool_calls=0) == ["q1", "q2"]
    assert ids(label_state="unlabelled") == ["q2"] and ids(tag="not_retrieved") == ["q1"]
    assert ids(terminations=["max_loops"]) == [] and ids(text="WHERE") == ["q1", "q2"]
