from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_agent.config import LLMConfig
from research_agent.evalset import tidb_build
from research_agent.evalset.dataset import Evidence
from research_agent.evalset.tidb_build import (
    BuildConfig,
    Composition,
    Draft,
    forbidden_terms,
    rewrite_failures,
    rewrite_one,
    rewrite_prompt,
)
from research_agent.evalset.tidb_questions import DraftQuestion
from research_agent.llm.pricing import PricingTable

from .evalset_fixtures import fake_factory

FACT: dict[str, object] = {
    "component": "tidb_config",
    "name": "log.slow-threshold",
    "field": "默认值",
    "value": "300",
    "quote": "+ 默认值：300",
    "description": "输出慢日志的耗时阈值。",
    "since": [],
    "doc": "tidb-configuration-file.md",
    "path": ["TiDB 配置文件", "log", "`slow-threshold`"],
}


def draft() -> Draft:
    q = DraftQuestion(
        kind="direct",
        question="TiDB 配置项 `log.slow-threshold` 的默认值是多少？",
        answer="300",
        doc=str(FACT["doc"]),
        quote=str(FACT["quote"]),
        source="tidb-configuration-file.md#log.slow-threshold",
        rule="param:默认值",
        fact=FACT,
    )
    return Draft(
        candidate_id="s001",
        group="descriptive",
        draft=q,
        evidence=[Evidence(doc=q.doc, chunk_id="7", quote=q.quote)],
        description="- 输出慢日志的耗时阈值。\n- 单位：毫秒",
        review_notes="说明最相似的其他参数：…",
    )


def test_rewrite_rules() -> None:
    assert forbidden_terms("log.slow-threshold") == ["log.slow-threshold", "slow-threshold"]
    d = draft()
    assert rewrite_failures("TiDB 中输出慢日志的耗时阈值配置项，默认值是多少？", d) == []
    assert rewrite_failures("slow-threshold 默认是多少？", d) == [
        "问题里出现了参数名 `slow-threshold`"
    ]
    assert rewrite_failures("超过 300 毫秒记慢日志的配置项默认值？", d) == [
        "问题里出现了默认值 300"
    ]
    assert rewrite_failures("超过 3000 毫秒的配置项默认值？", d) == []  # not the value 300
    assert rewrite_failures("一个配置项", d) == ["不是一个问句"]
    prompt = rewrite_prompt(d)
    assert "`log.slow-threshold`" in prompt and "TiDB 配置文件中的一个配置项" in prompt
    assert "- 单位：毫秒" in prompt


def config(tmp_path: Path) -> BuildConfig:
    return BuildConfig(
        evalset=str(tmp_path / "evalset.yaml"),
        corpus="c.yaml",
        composition=Composition(direct={}, descriptive={}, version={}),
        rewriter=LLMConfig(model="rewriter"),
    )


def test_rewrite_one_revises_once(
    env_api: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies = iter(
        [
            json.dumps({"question": "slow-threshold 的默认值是多少？"}),
            json.dumps({"question": "TiDB 记录慢查询的耗时阈值配置项，默认值是多少？"}),
        ]
    )
    monkeypatch.setattr(tidb_build, "client_from_config", fake_factory(lambda p, t: next(replies)))
    pricing = PricingTable(currency="CNY", models={})
    cand = rewrite_one(config(tmp_path), pricing, draft())
    assert cand.question == "TiDB 记录慢查询的耗时阈值配置项，默认值是多少？"
    assert cand.reject_reason is None and cand.cost > 0
    assert [r["problems"] for r in cand.origin["rewrites"]] == [
        ["问题里出现了参数名 `slow-threshold`"],
        [],
    ]
    assert cand.review_notes.startswith("说明最相似") and cand.answer == "300"

    monkeypatch.setattr(
        tidb_build, "client_from_config", fake_factory(lambda p, t: '{"skip": "说明太笼统"}')
    )
    skipped = rewrite_one(config(tmp_path), pricing, draft())
    assert skipped.reject_reason == "rewriter: 说明太笼统"


def test_supplement_plan(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    cfg.composition.direct = {"tidb_config:默认值": 4}
    cfg.composition.descriptive_ready = {"tidb_config": 2}
    base = draft()
    drafts = [
        base.model_copy(update={"candidate_id": f"d{i}", "group": "direct"}) for i in range(4)
    ] + [base.model_copy(update={"candidate_id": f"s{i}"}) for i in range(2)]
    status = {"d0": "ready", "d1": "closed_book", "d2": "closed_book", "d3": "closed_book"}
    status |= {"s0": "ready", "s1": "ready"}
    # direct: 1 of 4 ready (rate 0.25), 3 missing -> 3 / 0.25 * 1.2 = 14.4 -> 15
    assert tidb_build.supplement_plan(cfg, drafts, status) == {("direct", "tidb_config:默认值"): 15}


def test_leak_check_handles_cjk_neighbours_and_similarity_rule() -> None:
    d = draft().model_copy(update={"draft": draft().draft.model_copy(update={"answer": "0"})})
    # "为0的": CJK characters around the digit used to hide the leak (\w includes CJK).
    assert rewrite_failures("默认值为0的那个参数是什么？", d) == ["问题里出现了默认值 0"]
    assert rewrite_failures("v10 中默认值是多少？", d) == []  # part of another number
    notes = (
        "说明最相似的其他参数：\n"
        "- TiKV 配置项 `rocksdb.max-manifest-file-size`（相似度 0.96，默认值 128MiB）：…\n"
        '- TiKV 配置项 `raftdb.wal-dir`（相似度 0.80，默认值 ""）：…\n'
        '- TiKV 配置项 `raftdb.other`（相似度 0.97，默认值 "20MiB"）：…'
    )
    assert tidb_build.similar_conflicts(notes, '"20MiB"', 0.95) == [
        "rocksdb.max-manifest-file-size"
    ]
    assert tidb_build.similar_conflicts(notes, '"20MiB"', 0.99) == []
