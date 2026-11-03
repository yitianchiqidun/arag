from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from research_agent.config import RunConfig
from research_agent.evalset import closed_book, dataset
from research_agent.evalset.closed_book import ClosedBookChecker
from research_agent.evalset.dataset import (
    Attempt,
    ClosedBookConfig,
    ClosedBookResult,
    Decision,
    ReviewConfig,
)
from research_agent.llm.client import AccountError
from research_agent.llm.pricing import PricingTable

from .evalset_fixtures import CANDIDATES, CRITERIA, Brain, fake_factory, load, write_evalset
from .fakes import FakeResponse


def test_review_config_validation() -> None:
    with pytest.raises(ValidationError, match="more than one review rule"):
        ReviewConfig(criteria=CRITERIA, rules=[{"groups": ["a"]}, {"groups": ["a", "b"]}])  # type: ignore[list-item]
    with pytest.raises(ValidationError, match="needs review.llm"):
        ReviewConfig(criteria=CRITERIA, rules=[{"groups": ["a"], "prereview": True}])  # type: ignore[list-item]
    with pytest.raises(ValidationError, match='needs an "other" reason'):
        ReviewConfig(criteria=CRITERIA, reasons={"x": "X"}, rules=[{"groups": ["a"]}])  # type: ignore[list-item]
    review = ReviewConfig(criteria=CRITERIA, rules=[{"groups": ["a"]}])  # type: ignore[list-item]
    assert review.rule_for("a").spot_check_n == 0
    with pytest.raises(ValueError, match="no review rule covers group 'b'"):
        review.rule_for("b")


def test_unknown_groups_in_rules(tmp_path: Path) -> None:
    data = load(write_evalset(tmp_path)).model_dump()
    data["groups"] = {"direct": "直接问法"}
    with pytest.raises(ValidationError, match="unknown groups"):
        dataset.EvalsetConfig.model_validate(data)


def test_write_and_load_candidates(tmp_path: Path) -> None:
    cfg = load(write_evalset(tmp_path))
    dataset.write_candidates(cfg, CANDIDATES)
    assert dataset.load_candidates(cfg) == CANDIDATES
    meta = json.loads((cfg.out / dataset.META_FILE).read_text())
    assert meta["groups"] == {"direct": 2, "descriptive": 2, "version": 1}
    assert len(meta["chunks_sha256"]) == 64
    dataset.check_build(cfg)

    with pytest.raises(ValueError, match="duplicate candidate ids"):
        dataset.write_candidates(cfg, [CANDIDATES[0], CANDIDATES[0]])
    odd = CANDIDATES[0].model_copy(update={"group": "other"})
    with pytest.raises(ValueError, match="unknown groups"):
        dataset.write_candidates(cfg, [odd])
    (tmp_path / "build/chunks.json").write_text("[]")
    with pytest.raises(RuntimeError, match="changed since the candidates were built"):
        dataset.check_build(cfg)


def result(
    cid: str, *correct: bool, question: str = "q", error: str | None = None
) -> ClosedBookResult:
    attempts = [
        Attempt(temperature=0.7 if i else 0.0, reply="r", correct=c) for i, c in enumerate(correct)
    ]
    return ClosedBookResult(
        candidate_id=cid, question=question, answer="a", model="m", attempts=attempts, error=error
    )


def test_status_and_stale_results() -> None:
    cand = CANDIDATES[0].model_copy(update={"question": "q", "answer": "a"})
    assert dataset.status(cand, None) == "pending"
    assert dataset.status(cand, result("d1", False, error="timeout")) == "pending"
    assert dataset.status(cand, result("d1", True)) == "closed_book"
    assert dataset.status(cand, result("d1", False, False, True)) == "partial_memory"
    assert dataset.status(cand, result("d1", False, False, False, False)) == "ready"
    rejected = cand.model_copy(update={"reject_reason": "rule: no default"})
    assert dataset.status(rejected, result("d1", False)) == "rejected"
    # A result for an earlier version of the question no longer applies.
    rows = {"d1": result("d1", False), "zz": result("zz", False)}
    assert dataset.current(rows, [cand]) == {"d1": rows["d1"]}
    assert dataset.current(rows, [cand.model_copy(update={"question": "new"})]) == {}


def checker(monkeypatch: pytest.MonkeyPatch, brain: Brain) -> ClosedBookChecker:
    monkeypatch.setattr(closed_book, "client_from_config", fake_factory(brain))
    run = RunConfig.model_validate(
        {
            "llm": {"model": "agent"},
            "judge": {"model": "j"},
            "data": {"chunks_file": "x", "index_dir": "y"},
        }
    )
    pricing = PricingTable(currency="CNY", models={})
    return ClosedBookChecker(run, pricing, "MiniDB", ClosedBookConfig(samples=3))


def test_closed_book_checker(env_api: None, monkeypatch: pytest.MonkeyPatch) -> None:
    cand = CANDIDATES[0]  # answer "1 GB"
    prompts: list[str] = []

    def by_temperature(replies: dict[float, str]) -> Brain:
        def reply(prompt: str, temperature: float) -> str:
            prompts.append(prompt)
            return replies[temperature]

        return reply

    ready = checker(monkeypatch, by_temperature({0.0: "2 GB", 0.7: "4 GB"})).check(cand)
    assert [a.temperature for a in ready.attempts] == [0.0, 0.7, 0.7, 0.7]
    assert not any(a.correct for a in ready.attempts) and ready.cost > 0
    assert "关于MiniDB的问题" in prompts[0] and cand.question in prompts[0]
    assert dataset.status(cand, ready) == "ready"

    memorized = checker(monkeypatch, by_temperature({0.0: "1 GB"})).check(cand)
    assert len(memorized.attempts) == 1 and dataset.status(cand, memorized) == "closed_book"
    partial = checker(monkeypatch, by_temperature({0.0: "2 GB", 0.7: "1 GB"})).check(cand)
    assert len(partial.attempts) == 2 and dataset.status(cand, partial) == "partial_memory"

    def flaky(prompt: str, temperature: float) -> str:
        raise KeyError("choices")

    failed = checker(monkeypatch, flaky).check(cand)
    assert failed.error is not None and dataset.status(cand, failed) == "pending"

    def broke(prompt: str, temperature: float) -> str:
        raise AccountError("HTTP 400: Arrearage", response=FakeResponse(400))  # type: ignore[arg-type]

    with pytest.raises(AccountError):
        checker(monkeypatch, broke).check(cand)


def test_run_closed_book_resumes(
    env_api: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load(write_evalset(tmp_path))
    rejected = CANDIDATES[-1].model_copy(update={"reject_reason": "rule"})
    dataset.write_candidates(cfg, [*CANDIDATES[:-1], rejected])
    monkeypatch.setattr(closed_book, "client_from_config", fake_factory(lambda p, t: "不知道"))
    monkeypatch.setattr(
        PricingTable, "from_yaml", lambda path: PricingTable(currency="CNY", models={})
    )
    assert len(closed_book.run_closed_book(cfg, workers=2)) == 4  # the rejected one is skipped
    assert closed_book.run_closed_book(cfg) == []
    st = dataset.stats(cfg)
    assert st.by_group_status == {
        "descriptive": {"ready": 2},
        "direct": {"ready": 2},
        "version": {"rejected": 1},
    }
    assert st.reject_reasons == {"rule": 1} and st.closed_book_cost > 0
    # Rebuilding a candidate with a new question makes its old result stale.
    changed = CANDIDATES[0].model_copy(update={"question": "改过的问题？"})
    dataset.write_candidates(cfg, [changed, *CANDIDATES[1:]])
    assert dataset.statuses(cfg, dataset.load_candidates(cfg))["d1"] == "pending"
    assert sorted(r.candidate_id for r in closed_book.run_closed_book(cfg)) == ["d1", "v1"]


def test_decisions_and_first_judgments(tmp_path: Path) -> None:
    cfg = load(write_evalset(tmp_path))
    assert Decision(decision="keep", updated_at="t").label == "keep"
    assert Decision(decision="keep", answer="x", updated_at="t").label == "modify"
    assert Decision(decision="drop", question="x", updated_at="t").label == "drop"
    keep = Decision(decision="keep", updated_at="t")
    drop = Decision(decision="drop", reasons=["ambiguous"], updated_at="t")
    assert dataset.record_first_judgment(cfg, "d1", keep)
    dataset.save_decision(cfg, "d1", keep)
    dataset.save_decision(cfg, "d1", drop)
    assert not dataset.record_first_judgment(cfg, "d1", drop)  # write-once
    assert dataset.load_first_judgments(cfg)["d1"].decision == "keep"
    assert dataset.load_review(cfg)["d1"].decision == "drop"
    dataset.save_decision(cfg, "d1", None)
    assert dataset.load_review(cfg) == {} and "d1" in dataset.load_first_judgments(cfg)
    assert oct((cfg.out / dataset.REVIEW_FILE).stat().st_mode & 0o777) == "0o644"


def test_question_record(tmp_path: Path) -> None:
    cfg = load(write_evalset(tmp_path))
    edited = Decision(decision="keep", answer="1GB", updated_at="t")
    record = dataset.question_record(cfg, CANDIDATES[0], edited, "ready", "human")
    assert record["id"] == "mini__direct__d1" and record["question_type"] == "direct"
    assert record["question"] == CANDIDATES[0].question and record["answer"] == "1GB"
    assert record["evidence_chunks"] == ["0"] and record["evidence_docs"] == ["params.md"]
    assert record["evidence_quotes"][0].startswith("`mini_mem_quota`")
    assert (record["closed_book"], record["review"]) == ("ready", "human")
