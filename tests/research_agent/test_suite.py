from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from research_agent.eval import judge as judge_mod
from research_agent.eval import runner as runner_mod
from research_agent.eval.compare import check_comparable
from research_agent.eval.suite import Suite, deep_merge, run_suite
from research_agent.llm.client import MeteredLLMClient

from .fakes import PLUS_PRICING, completion, tool_call


def test_deep_merge_replaces_lists_and_merges_dicts() -> None:
    base = {"agent": {"max_loops": 15, "tools": ["a", "b"]}, "llm": {"model": "m"}}
    merged = deep_merge(base, {"agent": {"tools": ["a"]}, "method": "x"})
    assert merged == {
        "agent": {"max_loops": 15, "tools": ["a"]},
        "llm": {"model": "m"},
        "method": "x",
    }
    assert base["agent"]["tools"] == ["a", "b"]  # base untouched


def meta(**llm: Any) -> dict[str, Any]:
    return {
        "config": {
            "llm": {"model": "m", **llm},
            "judge": {"model": "j"},
            "data": {"chunks_file": "c", "index_dir": "i"},
        },
        "llm_snapshot": "m",
        "judge_snapshot": "j",
        "questions_sha256": "q",
        "workers": 5,
    }


def test_check_comparable_allows_only_variant_switches() -> None:
    ablated = meta()
    ablated["config"]["agent"] = {"tools": ["keyword_search", "read_chunk"]}
    diffs = check_comparable({"full": meta(), "wo_semantic": ablated})
    assert diffs["wo_semantic"] == {"agent.tools": ["keyword_search", "read_chunk"]}

    with pytest.raises(ValueError, match="llm.temperature"):
        check_comparable({"a": meta(), "b": meta(temperature=0.7)})
    other_questions = meta()
    other_questions["questions_sha256"] = "other"
    with pytest.raises(ValueError, match="questions_sha256"):
        check_comparable({"a": meta(), "b": other_questions})


@pytest.fixture
def suite_file(tiny_corpus: dict[str, Path], tmp_path: Path) -> Path:
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        yaml.safe_dump({"currency": "CNY", "models": {"m": PLUS_PRICING.model_dump()}})
    )
    base = {
        "llm": {"model": "m"},
        "judge": {"model": "m"},
        "agent": {"max_loops": 3, "tools": ["keyword_search", "read_chunk"]},
        "data": {
            "chunks_file": str(tiny_corpus["chunks"]),
            "index_dir": str(tmp_path / "no_index"),
        },
        "pricing_file": str(pricing),
    }
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    suite = {
        "name": "t",
        "base_config": str(tmp_path / "base.yaml"),
        "questions": str(tiny_corpus["questions"]),
        "output_root": str(tmp_path / "out"),
        "reference": "snippets",
        "variants": [
            {"name": "snippets"},
            {
                "name": "full_chunks",
                "overrides": {
                    "agent": {"tools": ["keyword_search"], "search_result": "full_chunk"}
                },
            },
        ],
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(suite))
    return path


def scripted_clients(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Agents search once, then answer "Berlin" if they saw full chunks, else "Paris"."""
    calls: list[dict[str, Any]] = []

    def agent_factory(cfg: object, pricing: object) -> MeteredLLMClient:
        client = MeteredLLMClient(model="m", pricing=PLUS_PRICING)

        def post(payload: dict[str, Any], attempts: list[int]) -> dict[str, Any]:
            attempts[0] = 1
            calls.append(payload)
            tool_msgs = [m["content"] for m in payload["messages"] if m["role"] == "tool"]
            if not tool_msgs:
                call = tool_call("keyword_search", {"keywords": ["paris"]})
                return completion(None, [call]).json()
            answer = "Berlin" if "-" * 80 in tool_msgs[0] else "Paris"
            return completion(answer).json()

        client._post_with_retry = post  # type: ignore[method-assign]
        return client

    def judge_factory(cfg: object, pricing: object) -> MeteredLLMClient:
        client = MeteredLLMClient(model="m", pricing=PLUS_PRICING)

        def post(payload: dict[str, Any], attempts: list[int]) -> dict[str, Any]:
            attempts[0] = 1
            text = payload["messages"][-1]["content"]
            pred = text.split("Generated answer: ")[1].split("\n")[0]
            gold = text.split("Gold answer: ")[1].split("\n")[0]
            return completion("correct" if pred == gold else "incorrect").json()

        client._post_with_retry = post  # type: ignore[method-assign]
        return client

    monkeypatch.setattr(runner_mod, "client_from_config", agent_factory)
    monkeypatch.setattr(judge_mod, "client_from_config", judge_factory)
    return calls


def test_suite_runs_every_variant_and_builds_the_table(
    env_api: None, suite_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = scripted_clients(monkeypatch)
    suite = Suite.from_yaml(suite_file)

    markdown = run_suite(suite, workers=2)

    out = Path(suite.output_root)
    for name in ("snippets", "full_chunks"):
        assert (out / name / "summary.csv").exists() and (out / name / "config.yaml").exists()
    full_meta = json.loads((out / "full_chunks" / "run_meta.json").read_text())
    assert full_meta["tools"] == ["keyword_search"]
    assert "**read_chunk**" not in full_meta["system_prompt"]
    # q1 (Paris) only right with snippets, q2 (Berlin) only right with full chunks.
    assert "| full_chunks | 50.0% | +0.0 pp | [" in markdown and "| 1.000 (1 / 1) |" in markdown
    assert '`agent.search_result="full_chunk"`' in markdown
    csv_header = (out / "comparison.csv").read_text().splitlines()[0]
    assert csv_header.startswith("variant,setting.agent.search_result,setting.agent.tools")

    n_calls = len(calls)
    run_suite(suite, workers=2)  # everything judged already: no new LLM calls
    assert len(calls) == n_calls


def test_suite_rejects_unknown_reference(suite_file: Path) -> None:
    data = yaml.safe_load(suite_file.read_text())
    data["reference"] = "nope"
    with pytest.raises(ValueError, match="reference"):
        Suite.model_validate(data)
