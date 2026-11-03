from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from research_agent.assistant.graph import Assistant
from research_agent.service.app import ServiceConfig, Turns, create_app
from research_agent.service.client import ApiClient, parse_sse

from .test_assistant_graph import assistant, diagnosis_agent

CFG = ServiceConfig(assistant_config="unused.yaml", request_timeout_s=30, max_active=2)


def client_for(intent: str, **kwargs: Any) -> tuple[TestClient, dict[str, Any]]:
    bot, parts = assistant(intent, **kwargs)
    bot = Assistant(bot.services, InMemorySaver())
    return TestClient(create_app(bot, CFG)), parts


def test_chat_returns_the_answer_and_its_events() -> None:
    client, _ = client_for("param_lookup")
    r = client.post(
        "/chat", json={"session_id": "s1", "message": "tidb_mem_quota_query 的默认值是多少？"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "s1" and body["intent"] == "param_lookup"
    assert body["answer"].startswith("默认值是 1073741824")
    assert body["citations"] == ["system-variables.md › tidb_mem_quota_query"]
    assert [e["type"] for e in body["events"]][-1] == "done" and body["pending_write"] is None
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.get("/health").json()["status"] == "ok"


def test_stream_sends_one_sse_event_per_assistant_event() -> None:
    client, _ = client_for("param_lookup")
    with client.stream("POST", "/chat/stream", json={"message": "默认值？"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = list(r.iter_lines())
    events = list(parse_sse(iter(lines)))
    assert [e["type"] for e in events] == [
        "session",
        "route",
        "retrieval_start",
        "tool_result",
        "answer",
        "done",
    ]
    assert "event: answer" in lines


def test_diagnosis_write_waits_for_confirmation() -> None:
    client, parts = client_for("diagnosis", agent=diagnosis_agent())
    body = client.post("/chat", json={"session_id": "d1", "message": "订单查询变慢了"}).json()
    assert body["pending_write"]["statement"] == "CREATE INDEX idx_status ON orders (status)"
    assert client.get("/sessions/d1").json()["pending_write"] is not None
    assert parts["admin"].executed == []
    done = client.post("/chat/confirm", json={"session_id": "d1", "approved": True}).json()
    assert parts["admin"].executed == [("CREATE INDEX idx_status ON orders (status)", None)]
    assert "根因：orders.status 没有索引" in done["answer"]
    assert (
        client.post("/chat/confirm", json={"session_id": "d1", "approved": True}).status_code == 404
    )


def test_api_client_drives_the_service() -> None:
    client, parts = client_for("diagnosis", agent=diagnosis_agent())
    api = ApiClient("http://unused", client=client)
    events = list(api.ask("d2", "订单查询变慢了"))
    assert any(e["type"] == "confirm_required" for e in events)
    assert api.pending("d2") is not None
    events = list(api.confirm("d2", approved=False))
    assert parts["admin"].executed == [] and api.pending("d2") is None
    assert [m["role"] for m in api.history("d2")] == ["user", "assistant"]


def test_admission_limits() -> None:
    turns = Turns(max_active=2)
    turns.admit("a")
    with pytest.raises(Exception, match="409") as busy:
        turns.admit("a")
    assert busy.value.status_code == 409  # type: ignore[attr-defined]
    turns.admit("b")
    with pytest.raises(Exception) as full:
        turns.admit("c")
    assert full.value.status_code == 503  # type: ignore[attr-defined]


class SlowAssistant:
    """Emits one event, then takes longer than the request may."""

    def __init__(self) -> None:
        self.closed = threading.Event()

    def ask(self, session_id: str, question: str) -> Iterator[dict[str, Any]]:
        try:
            yield {"type": "route", "intent": "diagnosis"}
            time.sleep(1.0)
            yield {"type": "tool_call", "tool": "slow_queries"}
            yield {"type": "answer", "text": "too late"}
        finally:
            self.closed.set()

    def confirm(self, session_id: str, approved: bool) -> Iterator[dict[str, Any]]:
        yield from ()

    def pending(self, session_id: str) -> dict[str, Any] | None:
        return None

    def history(self, session_id: str) -> list[dict[str, str]]:
        return []


def test_a_turn_past_the_timeout_ends_with_an_error_and_frees_the_session() -> None:
    slow = SlowAssistant()
    app = create_app(slow, ServiceConfig(assistant_config="x", request_timeout_s=0.3, max_active=1))
    client = TestClient(app)
    body = client.post("/chat", json={"session_id": "t", "message": "慢"}).json()
    assert [e["type"] for e in body["events"]] == ["route", "error", "done"]
    assert "请求超过" in body["error"] and body["answer"] is None
    assert slow.closed.wait(5)  # the worker closed the generator at its next event
    deadline = time.monotonic() + 5
    while client.get("/health").json()["active_turns"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert client.get("/health").json()["active_turns"] == 0
