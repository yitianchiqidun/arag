from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_agent.evalset import closed_book, dataset, review
from research_agent.evalset.dataset import Candidate, Decision, ReviewConfig
from research_agent.evalset.review import (
    MARKER_END,
    MARKER_START,
    PreReview,
    agreement,
    manual_scope,
    parse_prereview,
    replace_block,
)
from research_agent.llm.pricing import PricingTable

from .evalset_fixtures import CANDIDATES, CRITERIA, fake_factory, load, write_evalset
from .fakes import PLUS_PRICING

RULES = ReviewConfig.model_validate(
    {
        "criteria": CRITERIA,
        "llm": {"model": "reviewer"},
        "spot_check_seed": 7,
        "rules": [
            {"groups": ["direct"], "spot_check_n": 2},
            {"groups": ["descriptive"], "prereview": True, "spot_check_n": 3},
        ],
    }
)


def cand(cid: str, group: str = "descriptive") -> Candidate:
    return CANDIDATES[0].model_copy(update={"candidate_id": cid, "group": group})


def pre(cid: str, decision: str | None) -> PreReview:
    return PreReview(candidate_id=cid, question="q", answer="a", model="r", decision=decision)  # type: ignore[arg-type]


def test_manual_scope() -> None:
    ready = [cand("b"), cand("c"), cand("d"), cand("e")] + [cand(f"k{i}") for i in range(6)]
    ready += [cand(f"r{i}", "direct") for i in range(5)]
    pres = {"b": pre("b", "modify"), "c": pre("c", "drop"), "d": pre("d", None)}
    pres |= {f"k{i}": pre(f"k{i}", "keep") for i in range(6)}  # "e" is not pre-reviewed yet
    scope = manual_scope(ready, pres, RULES)
    assert {k: v for k, v in scope.items() if v != "spot_check"} == {
        "b": "prereview_modify",
        "c": "prereview_drop",
        "d": "prereview_error",
    }
    spot = sorted(k for k, v in scope.items() if v == "spot_check")
    assert len(spot) == 5 and sum(k.startswith("k") for k in spot) == 3
    assert manual_scope(list(reversed(ready)), pres, RULES) == scope  # seeded, order-free


def test_agreement() -> None:
    result = agreement([("keep", "keep"), ("modify", "keep"), ("drop", "modify"), ("drop", "drop")])
    assert (result.n, result.exact, result.usable) == (4, 0.5, 0.75)
    assert result.pairs["modify/keep"] == 1
    assert agreement([]).exact is None


def test_parse_prereview() -> None:
    reasons = RULES.reasons
    ok = parse_prereview(
        '好的：{"decision": "drop", "reasons": ["ambiguous", "ambiguous", "x"], "explanation": "两个"}',
        CANDIDATES[0],
        "m",
        reasons,
    )
    assert (ok.decision, ok.reasons, ok.error) == ("drop", ["ambiguous", "other"], None)
    assert (ok.question, ok.answer) == (CANDIDATES[0].question, CANDIDATES[0].answer)
    bad = parse_prereview('{"decision": "maybe"}', CANDIDATES[0], "m", reasons)
    assert bad.decision is None and bad.error is not None


def test_replace_block() -> None:
    text = f"head\n{MARKER_START}\nold\n{MARKER_END}\ntail\n"
    assert replace_block(text, "new") == f"head\n{MARKER_START}\nnew\n{MARKER_END}\ntail\n"
    with pytest.raises(ValueError, match="marker"):
        replace_block("no markers", "new")


PRICES = PricingTable(currency="CNY", models={"reviewer": PLUS_PRICING})


def test_prereview_model_and_estimate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load(write_evalset(tmp_path))
    agent_reviews = cfg.model_copy(
        update={"review": cfg.review.model_copy(update={"llm": cfg.load_run_config().llm})}  # type: ignore[union-attr]
    )
    with pytest.raises(ValueError, match="must differ from the agent model"):
        review.review_config(agent_reviews)

    dataset.write_candidates(cfg, CANDIDATES)
    no_attempts = [
        dataset.ClosedBookResult(
            candidate_id=c.candidate_id,
            question=c.question,
            answer=c.answer,
            model="agent",
            attempts=[dataset.Attempt(temperature=0.0, reply="?", correct=False)],
        )
        for c in CANDIDATES
    ]
    lines = "".join(r.model_dump_json() + "\n" for r in no_attempts)
    (cfg.out / dataset.CLOSED_BOOK_FILE).write_text(lines)
    prompts = review.prereview_prompts(cfg)
    assert sorted(prompts) == ["s1", "s2"]  # only the group with a pre-review rule
    assert "1. 只有一个正确答案。" in prompts["s1"] and "关于" not in prompts["s1"]
    assert "params.md，块 0" in prompts["s1"] and "默认值为 1 GB" in prompts["s1"]
    assert "- 控制语句执行超时的变量默认值是多少？" in prompts["s1"]  # same document
    assert "ambiguous（有歧义）" in prompts["s1"]
    monkeypatch.setattr(PricingTable, "from_yaml", lambda path: PRICES)
    estimate = review.estimate_prereview(cfg, lambda text: len(text))
    assert estimate.calls == 2 and estimate.output_tokens == 400 and estimate.cost > 0


PREREVIEW_REPLY = {"decision": "modify", "reasons": ["answer_too_long"], "suggested_answer": "1G"}


def test_round_trip(env_api: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load(write_evalset(tmp_path, spot_direct=1, spot_descriptive=0))
    dataset.write_candidates(cfg, CANDIDATES)

    def model(prompt: str, temperature: float) -> str:
        if prompt.startswith("你在审核"):
            return json.dumps(
                PREREVIEW_REPLY if "标准答案：1 GB" in prompt else {"decision": "keep"}
            )
        return "0" if "mini_timeout" in prompt else "不知道"  # memorized: d2 and v1

    for module in (closed_book, review):
        monkeypatch.setattr(module, "client_from_config", fake_factory(model))
    monkeypatch.setattr(PricingTable, "from_yaml", lambda path: PRICES)

    with pytest.raises(ValueError, match="5 candidates have no closed-book check"):
        review.finalize(cfg)
    closed_book.run_closed_book(cfg)
    with pytest.raises(ValueError, match="2 candidates have no pre-review"):
        review.finalize(cfg)
    assert {p.decision for p in review.run_prereview(cfg, workers=2)} == {"modify", "keep"}
    assert review.run_prereview(cfg) == []  # nothing left to do

    # Ready: d1 (direct), s1 (descriptive, pre-review "modify"), s2 (descriptive, "keep").
    # Scope: s1 (pre-review modify) and d1 (the one spot check of the direct group).
    scope, _ = review.human_scope(cfg)
    assert scope == {"s1": "prereview_modify", "d1": "spot_check"}
    (cfg.out / "README.md").write_text(f"x\n{MARKER_START}\nold\n{MARKER_END}\n")
    with pytest.raises(ValueError, match="2 candidates in the human review scope"):
        review.finalize(cfg)
    # s1: first judged "modify" (agrees), dropped after seeing the pre-review.
    modify = Decision(decision="keep", answer="1G", updated_at="t")
    drop = Decision(decision="drop", reasons=["ambiguous"], updated_at="t")
    dataset.record_first_judgment(cfg, "s1", modify)
    dataset.save_decision(cfg, "s1", drop)
    keep = Decision(decision="keep", updated_at="t")
    dataset.record_first_judgment(cfg, "d1", keep)
    dataset.save_decision(cfg, "d1", keep)

    summary = review.finalize(cfg)
    questions = json.loads((cfg.out / "questions.json").read_text())
    assert [q["id"] for q in questions] == ["mini__descriptive__s2", "mini__direct__d1"]
    assert [q["review"] for q in questions] == ["prereview", "human"]
    memorized = json.loads((cfg.out / "memorized.json").read_text())
    assert [q["id"] for q in memorized] == ["mini__direct__d2", "mini__version__v1"]
    assert {q["closed_book"] for q in memorized} == {"closed_book"}

    direct, descriptive = summary.rules
    assert (direct.scope, direct.spot_check_first, direct.kept_by_source) == (
        {"spot_check": 1},
        {"keep": 1},
        {"human": 1},
    )
    assert direct.agreement_all_human is None
    assert descriptive.agreement_all_human is not None
    assert descriptive.agreement_all_human.exact == 1.0  # first judgment "modify" = pre-review
    assert descriptive.changed_after_first == {"modify -> drop": 1}
    assert (descriptive.kept, descriptive.dropped, descriptive.drop_reasons) == (
        1,
        1,
        {"ambiguous": 1},
    )
    assert (summary.kept, summary.memorized, summary.without_first_judgment) == (2, 2, 0)
    readme = (cfg.out / "README.md").read_text()
    assert "old" not in readme and "| 描述问法 | 丢弃 | 1 题：有歧义 1 |" in readme
    assert "| 描述问法 | 第一次判断之后改判 | 1 题（修改 → 丢弃 1） |" in readme
    assert "| 直接问法、版本变更 | 抽检结果（按第一次判断） | 保留 1 |" in readme
    assert "| 合计 | 闭卷可答（`memorized.json`） | 2 题（直接问法 1，版本变更 1） |" in readme
