from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from research_agent.config import RunConfig
from research_agent.eval import judge as judge_mod
from research_agent.eval.judge import judge_predictions
from research_agent.eval.summarize import summarize_run, to_markdown
from research_agent.llm.client import MeteredLLMClient
from research_agent.llm.pricing import PricingTable

from .fakes import PLUS_PRICING, completion


def prediction(qid: str, pred: str, gold: str, **extra: Any) -> dict[str, Any]:
    base = {
        "qid": qid,
        "question": "?",
        "gold_answer": gold,
        "pred_answer": pred,
        "trajectory": [],
        "total_cost": 0.01,
        "loops": 2,
        "total_retrieved_tokens": 100,
        "latency_s": 10.0,
        "llm_calls": 2,
        "input_tokens": 1000,
        "output_tokens": 50,
        "cached_tokens": 500,
        "termination": "answered",
        "tool_calls": {"read_chunk": 1},
    }
    return base | extra


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    preds = [
        prediction("a", "Paris", "Paris"),
        prediction("b", "The answer is Berlin.", "Berlin", latency_s=30.0),
        prediction(
            "c", "Rome", "Madrid", termination="max_loops", tool_calls={"keyword_search": 2}
        ),
        prediction("d", "", "Oslo"),
    ]
    (tmp_path / "predictions.jsonl").write_text("\n".join(json.dumps(p) for p in preds) + "\n")
    (tmp_path / "run_meta.json").write_text(json.dumps({"llm_snapshot": "m-1", "currency": "CNY"}))
    return tmp_path


def test_judge_records_raw_replies(
    env_api: None, run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Upstream eval.py only accepts an exact "correct"; "Correct." counts as wrong.
    replies = {"Paris": "correct", "The answer is Berlin.": "Correct.", "Rome": "incorrect"}

    def factory(cfg: object, pricing: object) -> MeteredLLMClient:
        client = MeteredLLMClient(model="m", pricing=PLUS_PRICING)

        def post(payload: dict[str, Any], attempts: list[int]) -> dict[str, Any]:
            attempts[0] = 1
            text = payload["messages"][-1]["content"]
            pred = text.split("Generated answer: ")[1].split("\n")[0]
            return completion(replies[pred], prompt_tokens=120, completion_tokens=2).json()

        client._post_with_retry = post  # type: ignore[method-assign]
        return client

    monkeypatch.setattr(judge_mod, "client_from_config", factory)
    cfg = RunConfig.model_validate(
        {
            "llm": {"model": "m"},
            "judge": {"model": "m"},
            "data": {"chunks_file": "x", "index_dir": "y"},
        }
    )
    pricing = PricingTable(currency="CNY", models={"m": PLUS_PRICING})

    meta = judge_predictions(run_dir / "predictions.jsonl", cfg, pricing, workers=2)

    assert meta.response_counts == {"correct": 1, "correct.": 1, "incorrect": 1}
    assert meta.usage.calls == 3  # the empty prediction is never sent to the judge
    assert meta.llm_accuracy == pytest.approx(1 / 3)  # upstream: divided by answered (3)

    summary = summarize_run(run_dir)
    assert summary.n == 4 and summary.answered == 3
    assert summary.llm_acc == pytest.approx(1 / 3)
    assert summary.llm_acc_all == pytest.approx(1 / 4)
    assert summary.cont_acc == pytest.approx(2 / 3)  # "Berlin" is contained
    assert summary.termination == {"answered": 3, "max_loops": 1}
    assert summary.tool_calls_by_tool == {"keyword_search": 0.5, "read_chunk": 0.75}
    assert summary.cache_hit_rate == pytest.approx(0.5)
    assert summary.p50_latency_s == pytest.approx(10.0)
    assert summary.agent_cost == pytest.approx(0.04)
    assert summary.judge_cost == pytest.approx(meta.usage.cost)
    assert "LLM-Acc" in to_markdown(summary)


def test_summarize_requires_judged_predictions(run_dir: Path) -> None:
    with pytest.raises(ValueError, match="judge"):
        summarize_run(run_dir)
