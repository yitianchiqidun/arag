from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_agent.assistant import router_eval as re_

QUESTIONS = [
    {
        "id": f"tidb_params__{kind}__{kind[0]}{i:03d}",
        "question": f"{kind} {i}",
        "question_type": kind,
    }
    for kind in ("direct", "descriptive", "version")
    for i in range(12)
]


class Router:
    def __init__(self, intents: dict[str, str]) -> None:
        self.intents = intents

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        text = messages[-1]["content"].split("问题：")[-1]  # the prompt itself has examples
        intent = next((v for k, v in self.intents.items() if k in text), "research")
        return {"message": {"content": json.dumps({"intent": intent})}, "cost": 0.0001}


def test_build_mixes_handwritten_and_sampled_questions(tmp_path: Path) -> None:
    hand = tmp_path / "hand.yaml"
    hand.write_text("diagnosis: [页面变慢, 接口超时]\nout_of_scope: [今天天气]\n", "utf-8")
    evalset = tmp_path / "q.json"
    evalset.write_text(json.dumps(QUESTIONS), "utf-8")

    built = re_.build(hand, evalset, per_group=4)

    assert [(q.group, q.expected) for q in built[:3]] == [
        ("symptom", "diagnosis"),
        ("symptom", "diagnosis"),
        ("unrelated", "out_of_scope"),
    ]
    params = [q for q in built if q.group == "param"]
    assert [q.id.split("__")[1] for q in params] == ["direct"] * 2 + ["descriptive"] * 2
    assert len([q for q in built if q.group == "version"]) == 4
    assert [q.id for q in built] == [q.id for q in re_.build(hand, evalset, per_group=4)]  # seeded


def test_evaluate_and_summary_against_an_earlier_run() -> None:
    questions = [
        re_.RouterQuestion(
            id="a", group="symptom", question="页面变慢", expected="diagnosis", source="h"
        ),
        re_.RouterQuestion(
            id="b", group="symptom", question="接口超时", expected="diagnosis", source="h"
        ),
        re_.RouterQuestion(
            id="c", group="unrelated", question="天气", expected="out_of_scope", source="h"
        ),
    ]
    before = re_.evaluate(
        Router({"变慢": "diagnosis", "超时": "out_of_scope", "天气": "out_of_scope"}), questions
    )
    after = re_.evaluate(
        Router({"变慢": "diagnosis", "超时": "diagnosis", "天气": "out_of_scope"}), questions
    )
    table = re_.summary_markdown(after, before)
    assert "| symptom | diagnosis | 2 | 100%（2） | diagnosis 2 | 1 / 2 |" in table
    assert "| unrelated | out_of_scope | 1 | 100%（1） | out_of_scope 1 | 1 / 1 |" in table
    assert "| symptom | diagnosis | 2 | 50%（1） |" in re_.summary_markdown(before)
