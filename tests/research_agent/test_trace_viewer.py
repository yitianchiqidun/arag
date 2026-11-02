from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from research_agent.tracing.schema import LLMCall, MessageStats, Step, ToolCall
from research_agent.upstream import REPO_ROOT

from .test_tracing_review import CONFIG, prediction, recorded, write_run

APP = str(REPO_ROOT / "apps/trace_viewer.py")


@pytest.fixture
def results_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "results"
    records = [
        prediction("q1", pred_answer="Lyon", llm_accuracy=0.0, termination="max_loops"),
        prediction("q2", llm_accuracy=1.0),
    ]
    run = write_run(root / "stage2" / "run", records, CONFIG)
    llm = LLMCall(
        latency_ms=800,
        attempts=1,
        prompt_tokens=1200,
        completion_tokens=20,
        cached_tokens=900,
        cost=0.001,
        tools_offered=True,
        context=MessageStats(count=4, chars={"system": 900, "user": 40, "tool": 300}),
    )
    search = ToolCall(
        call_id="c1", tool_name="keyword_search", tool_args={"keywords": ["x"]}, tool_result="r"
    )
    t = recorded("q2", "Paris").model_copy(
        update={
            "steps": [
                Step(index=1, kind="tool_calls", llm=llm, tool_calls=[search]),
                Step(index=2, kind="answer", thought="Paris", llm=llm),
            ]
        }
    )
    (run / "trajectories.jsonl").write_text(t.model_dump_json() + "\n")
    monkeypatch.setenv("TRACE_VIEWER_ROOT", str(root))
    return run


def test_viewer_replays_and_saves_labels(results_root: Path) -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("共 2 题，筛选后 2 题" in c.value for c in at.caption)

    # q1 is converted from predictions; label it and save.
    at.multiselect(key="tags_q1").select("max_loops").run()
    at.text_area(key="note_q1").input("stuck in a loop").run()
    at.button(key="save_q1").click().run()
    assert not at.exception
    labels = json.loads((results_root / "labels.json").read_text())["labels"]
    assert labels["q1"]["tags"] == ["max_loops"] and labels["q1"]["note"] == "stuck in a loop"

    # Filter to correct answers: only the recorded q2 remains, with its growth charts.
    at.radio(key="f_correct").set_value("correct").run()
    assert not at.exception
    assert any("筛选后 1 题" in c.value for c in at.caption)
    assert len(at.get("vega_lite_chart")) == 2
