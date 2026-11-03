from __future__ import annotations

from typing import Any

import pytest

from research_agent.assistant.routing import channel_for, classify, parse_json


class Router:
    def __init__(self, reply: str, cost: float = 0.0002) -> None:
        self.reply = reply
        self.cost = cost
        self.prompts: list[str] = []

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.prompts.append(messages[-1]["content"])
        return {"message": {"content": self.reply}, "cost": self.cost}


def test_classify_reads_the_intent_and_reason() -> None:
    router = Router('好的：{"intent": "param_lookup", "reason": "问默认值"}')
    route = classify(router, "tidb_mem_quota_query 的默认值是多少？")
    assert (route.intent, route.reason, route.cost) == ("param_lookup", "问默认值", 0.0002)
    assert "tidb_mem_quota_query 的默认值是多少？" in router.prompts[0]


@pytest.mark.parametrize("reply", ["不是 JSON", '{"intent": "chitchat"}', '["research"]', "{bad"])
def test_classify_falls_back_to_research(reply: str) -> None:
    assert classify(Router(reply), "?").intent == "research"


def test_parse_json_takes_the_outer_object() -> None:
    assert parse_json('x {"a": {"b": 1}} y') == {"a": {"b": 1}}
    assert parse_json("none") == {}


def test_channel_for() -> None:
    assert channel_for("param_lookup", ["param_lookup"]) == "naive_rag"
    assert channel_for("research", ["param_lookup"]) == "arag"
    assert channel_for("param_lookup", []) == "arag"  # stage 6: everything to A-RAG
