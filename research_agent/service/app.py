"""HTTP service of the assistant (stage 9): FastAPI with server-sent events.

    ASSISTANT_CONFIG=configs/assistant.yaml uvicorn research_agent.service.app:app --port 8000

- ``POST /chat`` ``{"session_id"?, "message"}``: runs a turn, returns its answer and events;
- ``POST /chat/stream`` (same body): ``text/event-stream``, one SSE event per assistant event
  (route, retrieval_start, tool_call, tool_result, confirm_required, answer, error, done);
- ``POST /chat/confirm`` ``{"session_id", "approved", "stream"?}``: approve or reject the
  pending write (JSON, or SSE with ``stream``);
- ``GET /sessions/{id}``: pending write and history; ``GET /health``.

Stability: one turn at a time per session (409 otherwise) and at most ``max_active`` turns
admitted (503 beyond; the assistant runs turns one at a time and queues the rest). A turn
longer than ``request_timeout_s`` ends with an ``error`` event; the diagnosis loop also stops
itself at the assistant's ``turn_timeout_s``, and LLM calls and SQL keep their own timeouts.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

_END = object()


class AssistantLike(Protocol):
    def ask(self, session_id: str, question: str) -> Iterator[dict[str, Any]]: ...

    def confirm(self, session_id: str, approved: bool) -> Iterator[dict[str, Any]]: ...

    def pending(self, session_id: str) -> dict[str, Any] | None: ...

    def history(self, session_id: str) -> list[dict[str, str]]: ...


class ServiceConfig(BaseModel):
    assistant_config: str = Field(
        default_factory=lambda: os.environ.get("ASSISTANT_CONFIG", "configs/assistant.yaml")
    )
    request_timeout_s: float = Field(
        default_factory=lambda: float(os.environ.get("REQUEST_TIMEOUT_S", "300"))
    )
    max_active: int = Field(default_factory=lambda: int(os.environ.get("MAX_ACTIVE_TURNS", "4")))


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)


class ConfirmRequest(BaseModel):
    session_id: str
    approved: bool
    stream: bool = False


class ChatResponse(BaseModel):
    session_id: str
    answer: str | None
    intent: str | None
    citations: list[str]
    cost: float
    pending_write: dict[str, Any] | None
    error: str | None
    events: list[dict[str, Any]]


class Turns:
    """Admission control and a worker thread per turn.

    The worker drives the assistant's (synchronous) event generator and hands the events to
    the request through a queue. When the request gives up (timeout or client gone), the
    worker closes the generator at the next event, which releases the assistant.
    """

    def __init__(self, max_active: int) -> None:
        self.max_active = max_active
        self._sessions: set[str] = set()
        self._lock = threading.Lock()

    @property
    def active(self) -> int:
        return len(self._sessions)

    def admit(self, session: str) -> None:
        with self._lock:
            if session in self._sessions:
                raise HTTPException(409, "这个会话还有一轮对话在进行，请等它结束")
            if len(self._sessions) >= self.max_active:
                raise HTTPException(503, "助手正忙，请稍后再试")
            self._sessions.add(session)

    def start(
        self, session: str, events: Callable[[], Iterator[dict[str, Any]]]
    ) -> tuple[queue.Queue[Any], threading.Event]:
        """Run ``events()`` in a worker; the session must have been admitted."""
        out: queue.Queue[Any] = queue.Queue()
        stop = threading.Event()

        def work() -> None:
            gen = events()
            try:
                for event in gen:
                    out.put(event)
                    if stop.is_set():
                        break
            except Exception as e:  # noqa: BLE001 - reported to the client as an error event
                out.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                close = getattr(gen, "close", None)
                if close is not None:
                    close()
                with self._lock:
                    self._sessions.discard(session)
                out.put(_END)

        threading.Thread(target=work, daemon=True, name=f"turn-{session}").start()
        return out, stop


async def _drain(
    out: queue.Queue[Any], stop: threading.Event, timeout_s: float
) -> AsyncIterator[dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            remaining = deadline - time.monotonic()
            try:
                item = await run_in_threadpool(out.get, True, max(remaining, 0.0))
            except queue.Empty:
                stop.set()
                yield {"type": "error", "message": f"请求超过 {timeout_s:.0f} 秒，已停止等待"}
                yield {"type": "done", "latency_s": round(timeout_s, 2)}
                return
            if item is _END:
                return
            yield item
    finally:
        stop.set()  # the client went away or the turn ended: let the worker stop


def _sse(event: dict[str, Any]) -> str:
    data = json.dumps(event, ensure_ascii=False, default=str)
    return f"event: {event.get('type', 'message')}\ndata: {data}\n\n"


def create_app(assistant: AssistantLike | None = None, cfg: ServiceConfig | None = None) -> FastAPI:
    """The service; without ``assistant`` it builds one from ``cfg.assistant_config`` at start."""
    cfg = cfg or ServiceConfig()
    state: dict[str, Any] = {"assistant": assistant, "turns": Turns(cfg.max_active)}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if state["assistant"] is None:
            from research_agent.assistant.app import AssistantApp, AssistantConfig
            from research_agent.upstream import load_env

            load_env()
            state["assistant"] = AssistantApp(AssistantConfig.from_yaml(cfg.assistant_config))
        yield

    app = FastAPI(title="TiDB 运维诊断助手", lifespan=lifespan)

    def bot() -> AssistantLike:
        return state["assistant"]  # type: ignore[no-any-return]

    def turn(
        session: str, events: Callable[[], Iterator[dict[str, Any]]]
    ) -> AsyncIterator[dict[str, Any]]:
        turns: Turns = state["turns"]
        turns.admit(session)
        out, stop = turns.start(session, events)
        return _drain(out, stop, cfg.request_timeout_s)

    async def collect(session: str, stream: AsyncIterator[dict[str, Any]]) -> ChatResponse:
        events = [e async for e in stream]
        answers = [e for e in events if e.get("type") == "answer"]
        errors = [str(e.get("message")) for e in events if e.get("type") == "error"]
        last = answers[-1] if answers else {}
        pending = next(
            (e.get("write") for e in events if e.get("type") == "confirm_required"), None
        )
        return ChatResponse(
            session_id=session,
            answer=last.get("text"),
            intent=next((e.get("intent") for e in events if e.get("type") == "route"), None),
            citations=list(last.get("citations") or []),
            cost=round(sum(float(a.get("cost") or 0) for a in answers), 6),
            pending_write=pending,
            error="; ".join(errors) or None,
            events=events,
        )

    def stream_response(session: str, stream: AsyncIterator[dict[str, Any]]) -> StreamingResponse:
        async def body() -> AsyncIterator[str]:
            yield _sse({"type": "session", "session_id": session})
            async for event in stream:
                yield _sse(event)

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/chat")
    async def chat(req: ChatRequest) -> ChatResponse:
        session = req.session_id or uuid.uuid4().hex[:12]
        return await collect(session, turn(session, lambda: bot().ask(session, req.message)))

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        session = req.session_id or uuid.uuid4().hex[:12]
        return stream_response(session, turn(session, lambda: bot().ask(session, req.message)))

    @app.post("/chat/confirm", response_model=None)
    async def confirm(req: ConfirmRequest) -> ChatResponse | StreamingResponse:
        if bot().pending(req.session_id) is None:
            raise HTTPException(404, "这个会话没有等待确认的修改")
        stream = turn(req.session_id, lambda: bot().confirm(req.session_id, req.approved))
        if req.stream:
            return stream_response(req.session_id, stream)
        return await collect(req.session_id, stream)

    @app.get("/sessions/{session_id}")
    async def session_state(session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "pending_write": bot().pending(session_id),
            "history": bot().history(session_id),
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        assistant = bot()
        services = getattr(assistant, "services", None)
        tidb = getattr(getattr(assistant, "cfg", None), "tidb", None)
        cluster = None
        if tidb is not None:
            from research_agent.tidb.cluster import port_open

            cluster = port_open(tidb.host, tidb.port)
        return {
            "status": "ok",
            "active_turns": state["turns"].active,
            "cluster_reachable": cluster,
            "cluster_tools": services is not None and getattr(services, "reader", None) is not None,
        }

    return app


app = create_app()
