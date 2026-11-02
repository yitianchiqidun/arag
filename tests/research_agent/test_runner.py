from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_agent.config import RunConfig
from research_agent.eval import runner as runner_mod
from research_agent.eval.runner import MeteredBatchRunner, RunAborted, termination_reason
from research_agent.llm import client as client_mod
from research_agent.llm.client import MeteredLLMClient, UsageSummary
from research_agent.llm.pricing import PricingTable

from .fakes import PLUS_PRICING, FakeResponse, ScriptedPost, completion, tool_call


def usage(failed: int = 0) -> UsageSummary:
    return UsageSummary(
        calls=1,
        failed_calls=failed,
        prompt_tokens=0,
        completion_tokens=0,
        cached_tokens=0,
        cost=0.0,
        latency_ms=0.0,
    )


@pytest.mark.parametrize(
    ("result", "failed", "expected"),
    [
        (None, 0, "exception"),
        ({"answer": "x"}, 0, "answered"),
        ({"answer": "x", "max_loops_exceeded": True}, 0, "max_loops"),
        ({"answer": "x", "max_loops_exceeded": True}, 1, "llm_error"),
        ({"answer": "x", "token_budget_exceeded": True}, 0, "token_budget"),
    ],
)
def test_termination_reason(result: dict | None, failed: int, expected: str) -> None:
    assert termination_reason(result, usage(failed)) == expected


def make_config(corpus: dict[str, Path], model: str = "m") -> RunConfig:
    return RunConfig.model_validate(
        {
            "llm": {"model": model},
            "judge": {"model": model},
            "agent": {"max_loops": 3, "tools": ["keyword_search", "read_chunk"]},
            "data": {
                "chunks_file": str(corpus["chunks"]),
                "index_dir": str(corpus["root"] / "no_index"),
            },
        }
    )


def scripted_client_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each agent gets a client that searches once, then answers."""

    def factory(cfg: object, pricing: object) -> MeteredLLMClient:
        client = MeteredLLMClient(model="m", pricing=PLUS_PRICING)
        script = [
            completion(None, [tool_call("keyword_search", {"keywords": ["alice"]})], 200, 20, 0),
            completion("Paris", prompt_tokens=400, completion_tokens=5, cached_tokens=200),
        ]
        client._post_with_retry = lambda payload, attempts: (  # type: ignore[method-assign]
            attempts.__setitem__(0, 1) or script.pop(0).json()
        )
        return client

    monkeypatch.setattr(runner_mod, "client_from_config", factory)


def test_runner_enriches_predictions(
    env_api: None, tiny_corpus: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted_client_factory(monkeypatch)
    pricing = PricingTable(currency="CNY", models={"m": PLUS_PRICING})
    out = tmp_path / "run"

    MeteredBatchRunner(
        make_config(tiny_corpus), pricing, str(tiny_corpus["questions"]), str(out), num_workers=2
    ).run()

    records = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert {r["qid"] for r in records} == {"q1", "q2"}
    for r in records:
        assert r["pred_answer"] == "Paris"
        assert r["termination"] == "answered"
        assert r["tool_calls"] == {"keyword_search": 1}
        assert (r["llm_calls"], r["input_tokens"], r["output_tokens"], r["cached_tokens"]) == (
            2,
            600,
            25,
            200,
        )
        assert r["total_cost"] == pytest.approx(
            PLUS_PRICING.cost(200, 20) + PLUS_PRICING.cost(400, 5, 200)
        )
        assert r["latency_s"] >= 0 and r["llm_errors"] == []
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["n_questions"] == 2 and meta["currency"] == "CNY"


def test_runner_refuses_to_mix_configs(
    env_api: None, tiny_corpus: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted_client_factory(monkeypatch)
    pricing = PricingTable(currency="CNY", models={"m": PLUS_PRICING, "m2": PLUS_PRICING})
    out = tmp_path / "run"
    MeteredBatchRunner(
        make_config(tiny_corpus), pricing, str(tiny_corpus["questions"]), str(out)
    ).run()

    with pytest.raises(RuntimeError, match="different config"):
        MeteredBatchRunner(
            make_config(tiny_corpus, model="m2"), pricing, str(tiny_corpus["questions"]), str(out)
        ).run()


def test_account_error_stops_the_run_without_recording_wrong_answers(
    env_api: None, tiny_corpus: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pricing = PricingTable(currency="CNY", models={"m": PLUS_PRICING})
    out = tmp_path / "run"
    arrears = FakeResponse(400, {"error": {"code": "Arrearage", "message": "in arrears"}})

    def broke(cfg: object, pricing: object) -> MeteredLLMClient:
        client = MeteredLLMClient(model="m", pricing=PLUS_PRICING)
        monkeypatch.setattr(client_mod.requests, "post", ScriptedPost([arrears] * 10))
        return client

    monkeypatch.setattr(runner_mod, "client_from_config", broke)
    with pytest.raises(RunAborted, match="Arrearage"):
        MeteredBatchRunner(
            make_config(tiny_corpus), pricing, str(tiny_corpus["questions"]), str(out)
        ).run()
    predictions = out / "predictions.jsonl"
    assert not predictions.exists() or predictions.read_text() == ""

    # After topping up, the same command resumes and answers every question once.
    scripted_client_factory(monkeypatch)
    MeteredBatchRunner(
        make_config(tiny_corpus), pricing, str(tiny_corpus["questions"]), str(out)
    ).run()
    records = [json.loads(line) for line in predictions.read_text().splitlines()]
    assert sorted(r["qid"] for r in records) == ["q1", "q2"]
    assert all(r["pred_answer"] == "Paris" for r in records)
