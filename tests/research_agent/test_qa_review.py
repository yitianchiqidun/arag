from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from research_agent.evalset import dataset
from research_agent.evalset.dataset import Attempt, ClosedBookResult, EvalsetConfig
from research_agent.evalset.review import PREREVIEW_FILE, PreReview
from research_agent.upstream import REPO_ROOT

from .evalset_fixtures import CANDIDATES, load, write_evalset

APP = str(REPO_ROOT / "apps/qa_review.py")


@pytest.fixture
def evalset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EvalsetConfig:
    path = write_evalset(tmp_path, spot_direct=0, spot_descriptive=0)
    cfg = load(path)
    s1 = next(c for c in CANDIDATES if c.candidate_id == "s1")
    dataset.write_candidates(cfg, [s1])
    wrong = Attempt(temperature=0.0, reply="2 GB", correct=False)
    checked = ClosedBookResult(
        candidate_id="s1", question=s1.question, answer=s1.answer, model="agent", attempts=[wrong]
    )
    (cfg.out / dataset.CLOSED_BOOK_FILE).write_text(checked.model_dump_json() + "\n")
    pre = PreReview(
        candidate_id="s1",
        question=s1.question,
        answer=s1.answer,
        model="reviewer",
        decision="modify",
        reasons=["answer_too_long"],
        explanation="写成 1G",
        suggested_answer="1G",
    )
    (cfg.out / PREREVIEW_FILE).write_text(pre.model_dump_json() + "\n")
    monkeypatch.setenv("QA_REVIEW_CONFIG", str(path))
    return cfg


def test_review_page_keeps_the_first_judgment_apart_from_the_final_one(
    evalset: EvalsetConfig,
) -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("限制单条查询内存的变量" in m.value for m in at.markdown)
    assert any("闭卷（温度 0.0）：`2 GB`（错）" in m.value for m in at.markdown)
    assert any("复核原因**：预审判为修改" in m.value for m in at.markdown)
    assert any("保存你的判断后显示预审结论" in c.value for c in at.caption)
    assert not any("预审（reviewer）" in i.value for i in at.info)

    # First judgment: drop, before seeing the pre-review.
    at.radio(key="decision_s1").set_value("drop").run()
    at.multiselect(key="reasons_s1").select("ambiguous").run()
    at.button(key="save_s1").click().run()
    assert not at.exception
    first = dataset.load_first_judgments(evalset)["s1"]
    assert (first.decision, first.reasons) == ("drop", ["ambiguous"])
    assert dataset.load_review(evalset)["s1"].decision == "drop"

    # The pre-review is now shown; adopting its suggestion changes only the final decision.
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["f_state"] = "全部"
    at.run()
    assert any("预审（reviewer）**：修改" in i.value for i in at.info)
    assert any("你的第一次判断：丢弃" in c.value for c in at.caption)
    at.button(key="adopt_s1").click().run()
    assert dataset.load_review(evalset)["s1"].answer == "1G"
    assert dataset.load_first_judgments(evalset)["s1"].decision == "drop"

    # Clearing and saving again cannot replace the first judgment either.
    at.button(key="clear_s1").click().run()
    assert "s1" not in dataset.load_review(evalset)
    at.radio(key="decision_s1").set_value("keep").run()
    at.button(key="save_s1").click().run()
    assert dataset.load_review(evalset)["s1"].decision == "keep"
    assert dataset.load_first_judgments(evalset)["s1"].decision == "drop"
    assert any("预审（reviewer）**：修改" in i.value for i in at.info)  # stays visible
