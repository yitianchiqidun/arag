from __future__ import annotations

from pathlib import Path
from typing import Any

from research_agent.eval.cache_probe import FORCE_PROMPT, cases, loop_prefix, run_probe
from research_agent.llm.client import MeteredLLMClient

from .fakes import PLUS_PRICING, completion

TOOLS = [{"type": "function", "function": {"name": "keyword_search", "parameters": {}}}]


def trajectory(qid: str, forced: bool = True) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": f"question {qid}"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]
    if forced:
        messages += [
            {"role": "user", "content": FORCE_PROMPT},
            {"role": "assistant", "content": "answer"},
        ]
    return {"question_id": qid, "messages": messages, "tools": TOOLS}


def test_loop_prefix_stops_before_the_force_prompt() -> None:
    prefix = loop_prefix(trajectory("q1"))
    assert prefix is not None and [m["role"] for m in prefix] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert loop_prefix(trajectory("q2", forced=False)) is None


def test_each_conversation_serves_one_variant() -> None:
    picked = cases([trajectory(f"q{i}") for i in range(8)] + [trajectory("x", False)], 2)
    assert [(v, t["question_id"]) for v, t in picked] == [
        ("no_tools", "q0"),
        ("tool_choice_none", "q1"),
        ("tools_auto", "q2"),
        ("no_tools", "q3"),
        ("tool_choice_none", "q4"),
        ("tools_auto", "q5"),
    ]


def test_probe_sends_each_variant_and_records_the_cache(env_api: None, tmp_path: Path) -> None:
    payloads: list[dict[str, Any]] = []

    def new_client() -> MeteredLLMClient:
        client = MeteredLLMClient(model="m", pricing=PLUS_PRICING)

        def post(payload: dict[str, Any], attempts: list[int]) -> dict[str, Any]:
            attempts[0] = 1
            payloads.append(payload)
            warm = payload["messages"][-1]["role"] == "tool"
            cached = 0 if warm or "tools" not in payload else 800
            return completion("ok", None, 1000, 5, cached).json()

        client._post_with_retry = post  # type: ignore[method-assign]
        return client

    results = run_probe([trajectory(f"q{i}") for i in range(3)], new_client, 1, tmp_path)

    forced = payloads[1::2]
    assert [("tools" in p, p.get("tool_choice")) for p in forced] == [
        (False, None),
        (True, "none"),
        (True, "auto"),
    ]
    assert all(p["messages"][-1] == {"role": "user", "content": FORCE_PROMPT} for p in forced)
    assert [r.forced_cached_tokens for r in results] == [0, 800, 800]
    summary = (tmp_path / "summary.md").read_text()
    assert "| no_tools | 1 | 1,000 | 0% |" in summary
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 3
