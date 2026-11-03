"""Human review of an evaluation set built from a document corpus.

Run with ``uv run streamlit run apps/qa_review.py``. ``QA_REVIEW_CONFIG`` selects the evalset
config (default ``configs/evalset/tidb_params.yaml``). Decisions go to ``review.json`` next to
the candidates as soon as they are saved; ``python -m research_agent.evalset finalize`` turns
them into ``questions.json``.

The default view is the human review scope (research_agent/evalset/review.py). The pre-review
of a candidate is shown only after the reviewer saved a judgment; the first saved judgment
goes write-once to ``review_first.json`` (the agreement rate uses it), later saves only
change the final decision in ``review.json``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import streamlit as st

from research_agent.evalset.dataset import (
    Candidate,
    Decision,
    EvalsetConfig,
    Status,
    load_candidates,
    load_chunks,
    load_closed_book,
    load_first_judgments,
    load_review,
    record_first_judgment,
    save_decision,
)
from research_agent.evalset.review import PreReview, human_scope, load_prereviews

CONFIG = os.environ.get("QA_REVIEW_CONFIG", "configs/evalset/tidb_params.yaml")
STATUS_LABELS: dict[Status, str] = {
    "ready": "待审核（闭卷答不对）",
    "partial_memory": "部分记忆（采样时答对过）",
    "closed_book": "闭卷可答",
    "pending": "未做闭卷检查",
    "rejected": "出题时的检查未通过",
}
SCOPE_LABELS = {
    "prereview_modify": "预审判为修改",
    "prereview_drop": "预审判为丢弃",
    "prereview_error": "预审失败",
    "spot_check": "随机抽检",
}
LABELS = {"keep": "保留", "modify": "修改", "drop": "丢弃"}

st.set_page_config(page_title="评测集审核", layout="wide")


@st.cache_data(show_spinner=False)
def _chunks(path: str) -> dict[str, dict[str, object]]:
    return load_chunks(EvalsetConfig.from_yaml(path))


def sidebar(
    cfg: EvalsetConfig,
    candidates: list[Candidate],
    status: dict[str, Status],
    review: dict[str, Decision],
    scope: dict[str, str],
) -> list[Candidate]:
    if scope:
        decided = sum(cid in review for cid in scope)
        st.sidebar.metric("人工复核进度", f"{decided} / {len(scope)}")
    modes = ["待人工复核", "全部候选"] if scope else ["全部候选"]
    mode = st.sidebar.radio("范围", modes, key="f_mode") or modes[0]
    if mode == "待人工复核":
        shown = [c for c in candidates if c.candidate_id in scope]
    else:
        wanted = st.sidebar.selectbox(
            "状态", list(STATUS_LABELS), format_func=STATUS_LABELS.__getitem__, key="f_status"
        )
        shown = [c for c in candidates if status[c.candidate_id] == wanted]
    state = st.sidebar.radio("审核", ["未审核", "保留", "丢弃", "全部"], key="f_state") or "全部"
    groups = st.sidebar.multiselect(
        "题型", list(cfg.groups), default=list(cfg.groups), format_func=cfg.groups.__getitem__
    )
    shown = [c for c in shown if c.group in groups]
    want = {"未审核": None, "保留": "keep", "丢弃": "drop"}
    if state != "全部":
        shown = [
            c
            for c in shown
            if (review[c.candidate_id].decision if c.candidate_id in review else None)
            == want[state]
        ]
    return shown


def pick(cfg: EvalsetConfig, shown: list[Candidate]) -> Candidate:
    ids = [c.candidate_id for c in shown]
    current = st.session_state.get("cid")
    index = ids.index(current) if current in ids else 0
    prev, select, nxt = st.columns([1, 8, 1])
    if prev.button("◀", disabled=index == 0):
        index -= 1
    if nxt.button("▶", disabled=index >= len(ids) - 1):
        index += 1
    by_id = {c.candidate_id: c for c in shown}

    def label(i: str) -> str:
        return f"{i} · {cfg.groups[by_id[i].group]} · {by_id[i].question[:90]}"

    cid = select.selectbox(
        "候选", ids, index=index, format_func=label, label_visibility="collapsed"
    )
    cid = cid or ids[0]
    st.session_state["cid"] = cid
    return by_id[cid]


def checks(cfg: EvalsetConfig, c: Candidate) -> None:
    rows: list[str] = []
    result = load_closed_book(cfg).get(c.candidate_id)
    if result is not None and (result.question, result.answer) == (c.question, c.answer):
        for a in result.attempts:
            verdict = "对" if a.correct else "错"
            rows.append(f"- 闭卷（温度 {a.temperature}）：`{a.reply}`（{verdict}）")
        if result.error:
            rows.append(f"- 闭卷检查失败：{result.error}")
    if c.reject_reason:
        rows.append(f"- 未通过：{c.reject_reason}")
    st.markdown("**自动检查**\n" + ("\n".join(rows) or "（无）"))
    if c.origin:
        with st.expander("出题来源"):
            st.code(json.dumps(c.origin, ensure_ascii=False, indent=1), language="json")


def evidence(c: Candidate, chunks: dict[str, dict[str, object]]) -> None:
    for i, e in enumerate(c.evidence, 1):
        where = f"块 {e.chunk_id} · " if e.chunk_id is not None else ""
        st.markdown(f"**证据 {i}**：{where}`{e.doc}`")
        st.info(f"引用：{e.quote}")
        if e.chunk_id is not None:
            with st.expander("块全文"):
                st.markdown(str(chunks.get(e.chunk_id, {}).get("text", "（块不存在）")))


def _reset_widgets(key: str) -> None:
    for name in ("decision", "question", "answer", "reasons", "note"):
        st.session_state.pop(f"{name}_{key}", None)


def form(
    cfg: EvalsetConfig, c: Candidate, current: Decision | None, reasons: dict[str, str]
) -> None:
    key = c.candidate_id
    options = ["keep", "drop"]
    choice = st.radio(
        "决定",
        options,
        index=options.index(current.decision) if current else 0,
        format_func={"keep": "保留", "drop": "丢弃"}.__getitem__,
        horizontal=True,
        key=f"decision_{key}",
    )
    question = st.text_area(
        "问题（可修改）",
        current.question if current and current.question else c.question,
        key=f"question_{key}",
    )
    answer = st.text_input(
        "答案（可修改）",
        current.answer if current and current.answer else c.answer,
        key=f"answer_{key}",
    )
    chosen = st.multiselect(
        "原因（丢弃或修改时选）",
        list(reasons),
        default=current.reasons if current else [],
        format_func=lambda r: reasons.get(r, r),
        key=f"reasons_{key}",
    )
    note = st.text_input("备注", current.note if current else "", key=f"note_{key}")
    save, clear = st.columns(2)
    if save.button("保存", type="primary", key=f"save_{key}"):
        decision = Decision(
            decision=choice,  # type: ignore[arg-type]
            question=None if question.strip() == c.question else question.strip(),
            answer=None if answer.strip() == c.answer else answer.strip(),
            reasons=chosen,
            note=note.strip(),
            updated_at=datetime.now().isoformat(timespec="seconds"),
        )
        # The first save of a candidate is its first judgment, made before the pre-review was
        # shown; it is stored once and never changed (the agreement rate uses only it).
        record_first_judgment(cfg, key, decision)
        save_decision(cfg, key, decision)
        st.rerun()
    if clear.button("清除决定", key=f"clear_{key}", disabled=current is None):
        save_decision(cfg, key, None)
        _reset_widgets(key)
        st.rerun()


def prereview_view(
    cfg: EvalsetConfig,
    c: Candidate,
    pre: PreReview | None,
    current: Decision | None,
    first: Decision | None,
    in_scope: bool,
    reasons: dict[str, str],
) -> None:
    if in_scope and first is not None:
        st.caption(f"你的第一次判断：{LABELS[first.label]}（用于统计，之后不会改变）")
    if pre is None:
        return
    if in_scope and first is None:
        st.caption("保存你的判断后显示预审结论（一致率只用你第一次保存的判断）。")
        return
    if pre.decision is None:
        st.warning(f"预审失败：{pre.error}")
        return
    rows = [f"**预审（{pre.model}）**：{LABELS[pre.decision]}"]
    if pre.reasons:
        rows.append("原因：" + "、".join(reasons.get(r, r) for r in pre.reasons))
    rows.append(f"说明：{pre.explanation}")
    if pre.suggested_question:
        rows.append(f"建议问题：{pre.suggested_question}")
    if pre.suggested_answer:
        rows.append(f"建议答案：{pre.suggested_answer}")
    st.info("\n\n".join(rows))
    adoptable = (
        in_scope
        and current is not None
        and pre.decision == "modify"
        and bool(pre.suggested_question or pre.suggested_answer)
    )
    if adoptable and st.button("采用预审的修改建议", key=f"adopt_{c.candidate_id}"):
        assert current is not None
        adopted = current.model_copy(
            update={
                "decision": "keep",
                "question": pre.suggested_question
                if pre.suggested_question != c.question
                else None,
                "answer": pre.suggested_answer if pre.suggested_answer != c.answer else None,
                "reasons": current.reasons or pre.reasons,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        save_decision(cfg, c.candidate_id, adopted)
        _reset_widgets(c.candidate_id)
        st.rerun()


def main() -> None:
    cfg = EvalsetConfig.from_yaml(CONFIG)
    candidates = load_candidates(cfg)
    review = load_review(cfg)
    prereviews = load_prereviews(cfg)
    first = load_first_judgments(cfg)
    scope, status = human_scope(cfg)
    reasons = cfg.review.reasons if cfg.review else {}
    st.title(f"评测集审核 · {cfg.name}")
    if cfg.review is not None:
        with st.expander("审核标准", expanded=not review):
            st.markdown(cfg.review.criteria)
    if not candidates:
        st.info(f"{cfg.out_dir} 里还没有候选题。")
        return
    shown = sidebar(cfg, candidates, status, review, scope)
    st.caption(f"共 {len(candidates)} 个候选，当前筛选 {len(shown)} 个")
    if not shown:
        st.info("没有符合筛选条件的候选。")
        return
    c = pick(cfg, shown)
    st.markdown(f"### {c.question}")
    aliases = f"（也可：{'、'.join(c.answer_aliases)}）" if c.answer_aliases else ""
    st.markdown(f"**答案**：{c.answer}{aliases}")
    st.caption(f"题型：{cfg.groups[c.group]} · 状态：{STATUS_LABELS[status[c.candidate_id]]}")
    left, right = st.columns([3, 2])
    with left:
        evidence(c, _chunks(CONFIG))
    with right:
        checks(cfg, c)
        current = review.get(c.candidate_id)
        in_scope = c.candidate_id in scope
        if in_scope:
            st.markdown(f"**复核原因**：{SCOPE_LABELS[scope[c.candidate_id]]}")
        if status[c.candidate_id] != "ready":
            st.caption("只有“待审核”的候选进入最终集合，其余状态仅供查看。")
        elif cfg.review is not None and not in_scope:
            rule = cfg.review.rule_for(c.group)
            if rule.prereview and c.candidate_id not in prereviews:
                st.caption("还没有预审结论；预审后再确定是否需要人工复核。")
            else:
                source = "预审结论（保留）" if rule.prereview else "出题结果（保留）"
                st.caption(f"不在人工复核范围内：定稿时采用{source}。")
        else:
            form(cfg, c, current, reasons)
        prereview_view(
            cfg,
            c,
            prereviews.get(c.candidate_id),
            current,
            first.get(c.candidate_id),
            in_scope,
            reasons,
        )


main()
