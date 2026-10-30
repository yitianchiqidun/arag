"""Fakes shared by research_agent tests."""

from __future__ import annotations

import json
from typing import Any

from research_agent.llm.pricing import ModelPricing, PriceTier

PLUS_PRICING = ModelPricing(
    tiers=[
        PriceTier(max_input_tokens=131072, input=0.8, output=2.0),
        PriceTier(max_input_tokens=None, input=4.8, output=48.0),
    ],
    cached_input_ratio=0.2,
)


class FakeResponse:
    def __init__(self, status_code: int, body: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._body = body or {}
        self.text = json.dumps(self._body)

    def json(self) -> dict[str, Any]:
        return self._body


def completion(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 10,
    cached_tokens: int = 0,
) -> FakeResponse:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return FakeResponse(
        200,
        {
            "choices": [
                {"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            },
        },
    )


def tool_call(name: str, arguments: dict[str, Any], call_id: str = "call_0") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class ScriptedPost:
    """Stand-in for ``requests.post`` returning queued responses and recording payloads."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []

    def __call__(
        self, url: str, headers: dict[str, str], json: dict[str, Any], timeout: float
    ) -> FakeResponse:
        self.payloads.append(json)
        return self.responses.pop(0)
