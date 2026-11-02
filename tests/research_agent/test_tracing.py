from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from research_agent.eval import runner as runner_mod
from research_agent.eval.runner import MeteredBatchRunner
from research_agent.llm.client import LLMCallRecord, MeteredLLMClient
from research_agent.llm.pricing import PricingTable
from research_agent.tracing.recorder import TraceRecorder
from research_agent.tracing.schema import Trajectory, message_stats

from .fakes import PLUS_PRICING, FakeResponse, completion, tool_call
from .test_runner import make_config

Script = list[FakeResponse | Exception]


def run_scripted(
    monkeypatch: pytest.MonkeyPatch,
    corpus: dict[str, Path],
    out: Path,
    script: Script,
    max_loops: int = 3,
) -> tuple[dict[str, Trajectory], list[list[dict[str, Any]]]]:
    """Run both tiny-corpus questions with the same script; return trajectories and payloads."""
    payloads: list[list[dict[str, Any]]] = []

    def factory(cfg: object, pricing: object) -> MeteredLLMClient:
        client = MeteredLLMClient(model="m", pricing=PLUS_PRICING)
        queue = list(script)
        sent: list[dict[str, Any]] = []
        payloads.append(sent)

        def post(payload: dict[str, Any], attempts: list[int]) -> dict[str, Any]:
            attempts[0] = 1
            sent.append(json.loads(json.dumps(payload)))  # snapshot, like the HTTP body
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item.json()

        client._post_with_retry = post  # type: ignore[method-assign]
        return client

    monkeypatch.setattr(runner_mod, "client_from_config", factory)
    cfg = make_config(corpus)
    cfg.agent.max_loops = max_loops
    pricing = PricingTable(currency="CNY", models={"m": PLUS_PRICING})
    MeteredBatchRunner(cfg, pricing, str(corpus["questions"]), str(out), num_workers=1).run()
    lines = (out / "trajectories.jsonl").read_text().splitlines()
    trajectories = [Trajectory.model_validate_json(line) for line in lines]
    return {t.question_id: t for t in trajectories}, payloads


def test_recorded_trajectory_matches_the_requests(
    env_api: None, tiny_corpus: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script: Script = [
        completion("Let me search.", [tool_call("keyword_search", {"keywords": ["alice"]})], 200),
        completion("Paris", prompt_tokens=400, completion_tokens=5, cached_tokens=200),
    ]
    trajectories, payloads = run_scripted(monkeypatch, tiny_corpus, tmp_path / "run", script)

    t = trajectories["q1"]
    assert (t.source, t.method, t.final_answer, t.termination_reason) == (
        "recorded",
        "arag",
        "Paris",
        "answered",
    )
    assert [s.kind for s in t.steps] == ["tool_calls", "answer"]
    search, answer = t.steps
    assert search.thought == "Let me search." and answer.thought == "Paris"
    (call,) = search.tool_calls
    assert (call.call_id, call.tool_name, call.tool_args) == (
        "call_0",
        "keyword_search",
        {"keywords": ["alice"]},
    )
    assert "Chunk ID: 0" in call.tool_result and call.latency_ms is not None
    assert call.log["chunks_found"] >= 1
    assert search.llm is not None and answer.llm is not None
    assert (search.llm.prompt_tokens, answer.llm.prompt_tokens, answer.llm.cached_tokens) == (
        200,
        400,
        200,
    )
    # Context stats describe exactly what was sent in each call.
    first, second = payloads[0]
    assert search.llm.context == message_stats(first["messages"])
    assert answer.llm.context.count == 4
    assert set(answer.llm.context.chars) == {"system", "user", "assistant", "tool"}
    # The stored conversation is the last request plus the final reply, for SFT export.
    assert t.messages == second["messages"] + [{"role": "assistant", "content": "Paris"}]
    assert t.tools == first["tools"]
    assert t.prompt_tokens == 600 and t.tool_call_count == 1 and t.peak_prompt_tokens == 400


def test_parallel_tool_calls_and_forced_answer(
    env_api: None, tiny_corpus: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parallel = [
        tool_call("keyword_search", {"keywords": ["bob"]}, "call_a"),
        tool_call("read_chunk", {"chunk_ids": ["2"]}, "call_b"),
    ]
    script: Script = [
        completion(None, [tool_call("keyword_search", {"keywords": ["alice"]})]),
        completion(None, parallel),
        completion("Berlin"),
    ]
    trajectories, payloads = run_scripted(
        monkeypatch, tiny_corpus, tmp_path / "run", script, max_loops=2
    )

    t = trajectories["q2"]
    assert t.termination_reason == "max_loops"
    assert [s.kind for s in t.steps] == ["tool_calls", "tool_calls", "forced_answer"]
    assert [c.call_id for c in t.steps[1].tool_calls] == ["call_a", "call_b"]
    assert [c.tool_name for c in t.steps[1].tool_calls] == ["keyword_search", "read_chunk"]
    forced = t.steps[2]
    assert forced.llm is not None and forced.llm.tools_offered is False
    assert t.messages is not None
    last_request = payloads[1][-1]["messages"]
    assert last_request[-1]["role"] == "user" and "limit" in last_request[-1]["content"]
    assert t.messages == last_request + [{"role": "assistant", "content": "Berlin"}]


def test_llm_error_step(
    env_api: None, tiny_corpus: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked = requests.HTTPError('HTTP 400: {"error": {"code": "DataInspectionFailed"}}')
    script: Script = [blocked, completion("unknown")]
    trajectories, _ = run_scripted(monkeypatch, tiny_corpus, tmp_path / "run", script)

    t = trajectories["q1"]
    assert t.termination_reason == "llm_error"
    assert [s.kind for s in t.steps] == ["llm_error", "forced_answer"]
    error_step = t.steps[0]
    assert error_step.llm is not None and "DataInspectionFailed" in str(error_step.llm.error)
    assert t.peak_prompt_tokens == 100  # failed calls carry no usage and are skipped


def record(**kwargs: Any) -> dict[str, Any]:
    base = {"qid": "q", "question": "Q?", "gold_answer": "A", "pred_answer": "A"}
    return {**base, **kwargs}


def llm_record(**kwargs: Any) -> LLMCallRecord:
    return LLMCallRecord(latency_ms=10, attempts=1, prompt_tokens=50, **kwargs)


def test_naive_rag_trace_has_a_retrieval_step() -> None:
    recorder = TraceRecorder(agent_has_tools=False)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "ctx + Q?"}]
    reply = {"role": "assistant", "content": "A"}
    recorder.on_llm_call(messages, None, llm_record(), reply)
    entry = {
        "loop": 1,
        "tool_name": "dense_retrieval",
        "arguments": {"query": "Q?", "top_k": 5},
        "tool_result": "[]",
        "retrieved_tokens": 30,
        "chunks_found": 5,
        "latency_ms": 12.5,
    }

    t = recorder.build(record(trajectory=[entry]), "naive_rag", "m")

    assert [s.kind for s in t.steps] == ["retrieval", "answer"]
    retrieval = t.steps[0].tool_calls[0]
    assert (retrieval.latency_ms, retrieval.log) == (12.5, {"chunks_found": 5})
    assert t.tool_call_count == 0 and t.messages == messages + [reply]


def test_misaligned_trace_fails_loudly() -> None:
    recorder = TraceRecorder(agent_has_tools=True)
    reply = {"role": "assistant", "content": None}
    reply["tool_calls"] = [tool_call("keyword_search", {"keywords": ["x"]})]
    recorder.on_llm_call([{"role": "user", "content": "Q?"}], [{"t": 1}], llm_record(), reply)
    recorder.on_tool("keyword_search", 3.0)
    wrong_tool = {"loop": 1, "tool_name": "read_chunk", "arguments": {}, "tool_result": "r"}
    with pytest.raises(ValueError, match="order mismatch"):
        recorder.build(record(trajectory=[wrong_tool]), "arag", "m")
    # If the agent raised (upstream records "error"), tool calls are kept without results.
    t = recorder.build(record(trajectory=[], error="boom"), "arag", "m")
    assert t.steps[0].tool_calls[0].log == {"unrecorded": True}


def test_rewritten_history_is_detected() -> None:
    recorder = TraceRecorder(agent_has_tools=True)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Q?"}]
    recorder.on_llm_call(messages, [{"t": 1}], llm_record(), {"role": "assistant", "content": "a"})
    messages[0] = {"role": "user", "content": "compressed"}
    with pytest.raises(ValueError, match="modified"):
        recorder.conversation()
