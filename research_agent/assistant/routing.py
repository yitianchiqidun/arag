"""Intent routing (stage 7): classify a question, then pick the document channel for it.

Shared by the assistant's ``route`` node and the evaluation runner (``method: routed``), so
the evaluated router is the one the assistant uses. Imports nothing from LangGraph.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from research_agent.assistant import prompts

Intent = Literal["param_lookup", "research", "diagnosis", "out_of_scope"]
INTENTS: tuple[Intent, ...] = ("param_lookup", "research", "diagnosis", "out_of_scope")
Channel = Literal["arag", "naive_rag"]
FALLBACK_INTENT: Intent = "research"  # an unparsable reply takes the thorough channel


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]: ...


class Route(BaseModel):
    intent: Intent
    reason: str = ""
    cost: float = 0.0
    latency_s: float = 0.0


def parse_json(text: str) -> dict[str, Any]:
    """The first ``{...}`` object of a reply, or ``{}``."""
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start : end + 1]) if start >= 0 else {}
    except json.JSONDecodeError:
        data = {}
    return data if isinstance(data, dict) else {}


def classify(client: ChatClient, question: str) -> Route:
    start = time.perf_counter()
    reply = client.chat(
        [{"role": "user", "content": prompts.ROUTE.format(question=question)}], temperature=0.0
    )
    data = parse_json(str(reply["message"].get("content") or ""))
    intent = data.get("intent")
    return Route(
        intent=intent if intent in INTENTS else FALLBACK_INTENT,
        reason=str(data.get("reason") or ""),
        cost=float(reply.get("cost") or 0.0),
        latency_s=round(time.perf_counter() - start, 3),
    )


def channel_for(intent: str, fast_intents: Iterable[str]) -> Channel:
    """The fast channel (Naive RAG, asked to cite) for ``fast_intents``, A-RAG otherwise."""
    return "naive_rag" if intent in set(fast_intents) else "arag"
