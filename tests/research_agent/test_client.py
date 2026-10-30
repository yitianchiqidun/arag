from __future__ import annotations

import pytest
import requests

from research_agent.llm import client as client_mod
from research_agent.llm.client import MeteredLLMClient

from .fakes import PLUS_PRICING, FakeResponse, ScriptedPost, completion


def make_client(**kwargs: object) -> tuple[MeteredLLMClient, list[float]]:
    sleeps: list[float] = []
    client = MeteredLLMClient(
        model="qwen-plus",
        pricing=PLUS_PRICING,
        extra_body={"enable_thinking": False},
        sleep=sleeps.append,
        **kwargs,  # type: ignore[arg-type]
    )
    return client, sleeps


def test_payload_and_record(env_api: None, monkeypatch: pytest.MonkeyPatch) -> None:
    post = ScriptedPost(
        [completion("hi", prompt_tokens=1000, completion_tokens=50, cached_tokens=400)]
    )
    monkeypatch.setattr(client_mod.requests, "post", post)
    client, _ = make_client(max_tokens=123)

    result = client.chat([{"role": "user", "content": "q"}], tools=[{"type": "function"}])

    payload = post.payloads[0]
    assert payload["enable_thinking"] is False
    assert "reasoning_effort" not in payload
    assert payload["max_tokens"] == 123 and payload["tool_choice"] == "auto"
    expected = PLUS_PRICING.cost(1000, 50, 400)
    assert result["cost"] == pytest.approx(expected)
    assert result["message"]["content"] == "hi"
    (rec,) = client.records
    assert (rec.prompt_tokens, rec.completion_tokens, rec.cached_tokens) == (1000, 50, 400)
    assert rec.attempts == 1 and rec.error is None and rec.finish_reason == "stop"
    assert client.usage_summary().cost == pytest.approx(expected)


def test_generate_compat_used_by_upstream_eval(
    env_api: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(client_mod.requests, "post", ScriptedPost([completion("correct")]))
    client, _ = make_client()
    text, cost = client.generate(messages=[{"role": "user", "content": "x"}], temperature=0.0)
    assert text == "correct" and cost > 0


def test_retries_transient_status_then_succeeds(
    env_api: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = ScriptedPost([FakeResponse(429), FakeResponse(503), completion("ok")])
    monkeypatch.setattr(client_mod.requests, "post", post)
    client, sleeps = make_client(max_retries=4, backoff_base_s=1.0)

    assert client.chat([{"role": "user", "content": "q"}])["message"]["content"] == "ok"
    assert client.records[0].attempts == 3
    assert len(sleeps) == 2 and 0.5 <= sleeps[0] <= 1.5 and 1.0 <= sleeps[1] <= 3.0


def test_non_retryable_error_fails_fast(env_api: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        client_mod.requests, "post", ScriptedPost([FakeResponse(400, {"error": "bad"})])
    )
    client, sleeps = make_client()
    with pytest.raises(requests.HTTPError, match="400"):
        client.chat([{"role": "user", "content": "q"}])
    (rec,) = client.records
    assert rec.attempts == 1 and rec.error is not None and sleeps == []


def test_gives_up_after_max_retries(env_api: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_mod.requests, "post", ScriptedPost([FakeResponse(429)] * 3))
    client, sleeps = make_client(max_retries=2)
    with pytest.raises(requests.HTTPError, match="429"):
        client.chat([{"role": "user", "content": "q"}])
    assert client.records[0].attempts == 3 and len(sleeps) == 2
    assert client.usage_summary().failed_calls == 1


def test_retries_timeouts(env_api: None, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def flaky(*args: object, **kwargs: object) -> FakeResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.Timeout("slow")
        return completion("ok")

    monkeypatch.setattr(client_mod.requests, "post", flaky)
    client, _ = make_client()
    assert client.chat([{"role": "user", "content": "q"}])["message"]["content"] == "ok"
    assert client.records[0].attempts == 2
