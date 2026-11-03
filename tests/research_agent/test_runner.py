from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from arag.tools.read_chunk import ReadChunkTool
from research_agent.config import LLMConfig, RoutingConfig, RunConfig
from research_agent.eval import runner as runner_mod
from research_agent.eval.runner import (
    CostBudget,
    MeteredBatchRunner,
    RunAborted,
    termination_reason,
)
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


def test_cost_limit_stops_new_questions_and_resumes(
    env_api: None, tiny_corpus: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted_client_factory(monkeypatch)
    pricing = PricingTable(currency="CNY", models={"m": PLUS_PRICING})
    out = tmp_path / "run"
    questions = str(tiny_corpus["questions"])
    budget = CostBudget(limit=1e-9, workers=1)  # the first question uses it up
    with pytest.raises(RunAborted, match="cost limit"):
        MeteredBatchRunner(
            make_config(tiny_corpus), pricing, questions, str(out), num_workers=1, budget=budget
        ).run()
    records = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert len(records) == 1 and budget.spent == pytest.approx(records[0]["total_cost"])

    # A higher limit resumes the remaining question only.
    budget = CostBudget(limit=1.0, spent=budget.spent, workers=1)
    MeteredBatchRunner(
        make_config(tiny_corpus), pricing, questions, str(out), num_workers=1, budget=budget
    ).run()
    records = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert sorted(r["qid"] for r in records) == ["q1", "q2"]
    assert budget.spent == pytest.approx(sum(r["total_cost"] for r in records))


def test_cost_budget_reserves_in_flight_questions() -> None:
    budget = CostBudget(limit=1.0, spent=0.5, workers=4)
    budget.charge(0.1)  # 0.6 spent + 4 x 0.1 in flight = 1.0
    assert budget.exhausted and "spent 0.6000" in budget.describe()
    judge = CostBudget(limit=1.0, workers=4)
    judge.charge(0.5, question=False)  # judge calls do not count as a question
    assert not judge.exhausted


def test_routed_runner_answers_with_the_channel_of_each_intent(
    env_api: None, tiny_corpus: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research_agent.baselines.naive_rag import NaiveRAG

    from .test_naive_rag import BagOfWordsModel, build

    def factory(cfg: Any, pricing: object) -> MeteredLLMClient:
        client = MeteredLLMClient(model=cfg.model, pricing=PLUS_PRICING)

        def post(payload: dict[str, Any], attempts: list[int]) -> dict[str, Any]:
            attempts[0] = 1
            messages = payload["messages"]
            if cfg.model == "router":  # Alice: research (A-RAG), Bob: a parameter (Naive RAG)
                intent = "research" if "Alice" in messages[-1]["content"] else "param_lookup"
                return completion(json.dumps({"intent": intent}), None, 50, 5).json()
            if "tools" not in payload:
                return completion("Berlin [Chunk 2]", prompt_tokens=300, completion_tokens=3).json()
            if not any(m["role"] == "tool" for m in messages):
                call = tool_call("keyword_search", {"keywords": ["alice"]})
                return completion(None, [call], 200, 20).json()
            return completion("Paris", prompt_tokens=400, completion_tokens=5).json()

        client._post_with_retry = post  # type: ignore[method-assign]
        return client

    monkeypatch.setattr(runner_mod, "client_from_config", factory)
    model = BagOfWordsModel()
    index = build(tiny_corpus, model)
    reader = ReadChunkTool(str(tiny_corpus["chunks"]))
    monkeypatch.setattr(
        MeteredBatchRunner,
        "_naive_rag",
        lambda self, llm=None: NaiveRAG(llm, index, reader, model, threading.Lock(), 1, True),
    )
    cfg = make_config(tiny_corpus).model_copy(
        update={"method": "routed", "routing": RoutingConfig(router=LLMConfig(model="router"))}
    )
    pricing = PricingTable(currency="CNY", models={"m": PLUS_PRICING, "router": PLUS_PRICING})
    out = tmp_path / "run"

    MeteredBatchRunner(cfg, pricing, str(tiny_corpus["questions"]), str(out), num_workers=2).run()

    records = {
        r["qid"]: r for r in map(json.loads, (out / "predictions.jsonl").read_text().splitlines())
    }
    route_cost = PLUS_PRICING.cost(50, 5)
    alice, bob = records["q1"], records["q2"]
    assert (alice["intent"], alice["channel"], alice["pred_answer"]) == (
        "research",
        "arag",
        "Paris",
    )
    assert alice["tool_calls"] == {"keyword_search": 1} and alice["llm_calls"] == 2
    assert alice["total_cost"] == pytest.approx(
        route_cost + PLUS_PRICING.cost(200, 20) + PLUS_PRICING.cost(400, 5)
    )
    assert (bob["intent"], bob["channel"]) == ("param_lookup", "naive_rag")
    assert bob["pred_answer"] == "Berlin [Chunk 2]" and bob["llm_calls"] == 1
    assert bob["tool_calls"] == {"dense_retrieval": 1}
    assert bob["total_cost"] == pytest.approx(route_cost + PLUS_PRICING.cost(300, 3))
    assert bob["route_cost"] == pytest.approx(route_cost)
    assert [c["chunk_id"] for c in bob["citations"]] == ["2"]
    trajectories = {
        t["question_id"]: t
        for t in map(json.loads, (out / "trajectories.jsonl").read_text().splitlines())
    }
    assert [s["kind"] for s in trajectories["q1"]["steps"]] == ["tool_calls", "answer"]
    assert [s["kind"] for s in trajectories["q2"]["steps"]] == ["retrieval", "answer"]
    assert trajectories["q1"]["method"] == "routed"


def test_routed_method_needs_a_router(tiny_corpus: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="routing.router"):
        RunConfig.model_validate(make_config(tiny_corpus).model_dump() | {"method": "routed"})
