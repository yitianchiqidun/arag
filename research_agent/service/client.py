"""Client of the HTTP service with the AssistantApp surface (ask / confirm / pending / history),
so the demo page can run against the service instead of loading its own assistant."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx


def parse_sse(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """The JSON payloads of a ``text/event-stream`` (one ``data:`` line per event)."""
    for line in lines:
        if line.startswith("data:"):
            yield json.loads(line[len("data:") :].strip())


class ApiClient:
    def __init__(
        self, base_url: str, timeout_s: float = 330.0, client: httpx.Client | None = None
    ) -> None:
        self.client = client or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout_s, connect=10.0)
        )

    def _stream(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        with self.client.stream("POST", path, json=payload) as r:
            if r.status_code != 200:
                r.read()
                is_json = r.headers.get("content-type", "").startswith("application/json")
                detail = r.json().get("detail") if is_json else r.text
                yield {"type": "error", "message": f"HTTP {r.status_code}: {detail}"}
                yield {"type": "done", "latency_s": 0.0}
                return
            for event in parse_sse(r.iter_lines()):
                if event.get("type") != "session":
                    yield event

    def ask(self, session_id: str, question: str) -> Iterator[dict[str, Any]]:
        yield from self._stream("/chat/stream", {"session_id": session_id, "message": question})

    def confirm(self, session_id: str, approved: bool) -> Iterator[dict[str, Any]]:
        payload = {"session_id": session_id, "approved": approved, "stream": True}
        yield from self._stream("/chat/confirm", payload)

    def _session(self, session_id: str) -> dict[str, Any]:
        r = self.client.get(f"/sessions/{session_id}")
        r.raise_for_status()
        return dict(r.json())

    def pending(self, session_id: str) -> dict[str, Any] | None:
        return self._session(session_id).get("pending_write")

    def history(self, session_id: str) -> list[dict[str, str]]:
        return list(self._session(session_id).get("history") or [])
