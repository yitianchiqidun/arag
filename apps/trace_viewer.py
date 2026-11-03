"""Trajectory viewer: replay any question of any run step by step and label bad cases.

    uv run streamlit run apps/trace_viewer.py            # runs under ./results
    TRACE_VIEWER_ROOT=path/to/results uv run streamlit run apps/trace_viewer.py

All logic lives in research_agent.tracing (store / review / labels); this file is layout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd  # type: ignore[import-untyped]
import streamlit as st

from research_agent.eval.citations import (
    ChunkRef,
    load_chunk_refs,
    resolve_citations,
    shown_chunk_ids,
)
from research_agent.tracing.labels import LabelStore, Taxonomy
from research_agent.tracing.review import (
    TrajectoryFilter,
    apply_filter,
    context_growth,
    step_count,
    suggest_tags,
)
from research_agent.tracing.schema import Step, Trajectory
from research_agent.tracing.store import RunInfo, find_runs, load_run, run_info

ROOT = Path(os.environ.get("TRACE_VIEWER_ROOT", "results"))
KIND_LABELS = {
    "tool_calls": "调用工具",
    "answer": "作答",
    "forced_answer": "强制作答",
    "llm_error": "LLM 报错",
    "retrieval": "向量检索",
}
# Reference categorical palette, slots 1-4 in fixed order (dataviz skill, palette.md).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

st.set_page_config(page_title="轨迹回放", layout="wide")


@st.cache_data(show_spinner="加载轨迹…")
def _load(run_dir: str, stamp: tuple[float, ...]) -> tuple[RunInfo, list[Trajectory]]:
    # ``stamp`` (file mtimes) is only part of the cache key: a rerun or judge invalidates it.
    return run_info(run_dir), load_run(run_dir)


def _stamp(run_dir: Path) -> tuple[float, ...]:
    files = ("predictions.jsonl", "trajectories.jsonl", "run_meta.json")
    return tuple((run_dir / f).stat().st_mtime if (run_dir / f).exists() else 0.0 for f in files)


def _verdict(t: Trajectory) -> str:
    return {True: "✅", False: "❌", None: "·"}[t.correct]


def _gold(t: Trajectory) -> str:
    return t.gold_answer if isinstance(t.gold_answer, str) else " / ".join(t.gold_answer)


def sidebar(trajectories: list[Trajectory], taxonomy: Taxonomy) -> TrajectoryFilter:
    st.sidebar.subheader("筛选")
    correctness = st.sidebar.radio(
        "判分",
        ["all", "wrong", "correct", "unjudged"],
        format_func={"all": "全部", "wrong": "错误", "correct": "正确", "unjudged": "未判分"}.get,
        horizontal=True,
        key="f_correct",
    )
    steps = [step_count(t) for t in trajectories] or [0]
    calls = [t.tool_call_count for t in trajectories] or [0]
    step_range = st.sidebar.slider(
        "步数", min(steps), max(max(steps), min(steps) + 1), (min(steps), max(steps))
    )
    call_range = st.sidebar.slider(
        "工具调用次数", min(calls), max(max(calls), min(calls) + 1), (min(calls), max(calls))
    )
    terminations = sorted({t.termination_reason for t in trajectories})
    chosen = st.sidebar.multiselect("结束原因", terminations, key="f_term")
    label_state = st.sidebar.radio(
        "标注状态",
        ["all", "unlabelled", "labelled"],
        format_func={"all": "全部", "unlabelled": "未标注", "labelled": "已标注"}.get,
        horizontal=True,
        key="f_label",
    )
    tag = st.sidebar.selectbox(
        "含标签",
        [None, *taxonomy.ids],
        format_func=lambda x: "不限" if x is None else taxonomy.label(x),
    )
    text = st.sidebar.text_input("搜索（题号 / 题目 / 答案）", key="f_text")
    return TrajectoryFilter.model_validate(
        {
            "correctness": correctness,
            "min_steps": step_range[0],
            "max_steps": step_range[1],
            "min_tool_calls": call_range[0],
            "max_tool_calls": call_range[1],
            "terminations": chosen,
            "label_state": label_state,
            "tag": tag,
            "text": text,
        }
    )


def overview(shown: list[Trajectory], store: LabelStore, taxonomy: Taxonomy) -> None:
    labels = store.load().labels
    rows = [
        {
            "判分": _verdict(t),
            "题号": t.question_id,
            "步数": step_count(t),
            "工具调用": t.tool_call_count,
            "结束原因": t.termination_reason,
            "花费": round(t.total_cost, 4),
            "标签": "、".join(taxonomy.label(x) for x in labels[t.question_id].tags)
            if t.question_id in labels
            else "",
            "题目": t.question,
        }
        for t in shown
    ]
    with st.expander(f"题目列表（{len(shown)}）", expanded=False):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def pick(shown: list[Trajectory]) -> Trajectory:
    ids = [t.question_id for t in shown]
    current = st.session_state.get("qid")
    index = ids.index(current) if current in ids else 0
    prev_col, select_col, next_col = st.columns([1, 10, 1])
    if prev_col.button("◀", key="prev", disabled=index == 0):
        index -= 1
    if next_col.button("▶", key="next", disabled=index >= len(ids) - 1):
        index += 1
    by_id = {t.question_id: t for t in shown}
    qid = select_col.selectbox(
        "题目",
        ids,
        index=index,
        format_func=lambda q: f"{_verdict(by_id[q])} {q} · {by_id[q].question[:80]}",
        label_visibility="collapsed",
    )
    qid = qid or ids[0]
    st.session_state["qid"] = qid
    return by_id[qid]


@st.cache_data(show_spinner=False)
def chunk_refs(chunks_file: str) -> dict[str, ChunkRef]:
    path = Path(chunks_file)
    return load_chunk_refs(path) if path.exists() else {}


CITATION_STATUS = {
    "read": "读过全文",
    "snippet": "只看过搜索片段",
    "unseen": "从未出现过",
    "unknown": "不存在的块",
}


def citations_view(t: Trajectory, refs: dict[str, ChunkRef]) -> None:
    full, snippet = shown_chunk_ids(
        (c.tool_name, c.tool_result) for s in t.steps for c in s.tool_calls
    )
    citations = resolve_citations(t.final_answer, refs, full, snippet)
    if not citations:
        st.caption("答案没有引用任何块。")
        return
    lines = [
        f"- Chunk {c.chunk_id}"
        + (f"：{c.label()}" if c.doc else "")
        + f"（{CITATION_STATUS[c.status]}）"
        for c in citations
    ]
    st.markdown("**答案引用**\n" + "\n".join(lines))


def header(t: Trajectory, refs: dict[str, ChunkRef]) -> None:
    st.markdown(f"**问题**：{t.question}")
    left, right = st.columns(2)
    left.markdown(f"**标准答案**：{_gold(t)}")
    right.markdown(f"**模型答案**：{t.final_answer or '（空）'}")
    citations_view(t, refs)
    verdict = {True: "正确", False: "错误", None: "未判分"}[t.correct]
    contain = {True: "包含", False: "不包含", None: "—"}[t.contain_correct]
    cols = st.columns(6)
    cols[0].metric("LLM 判分", verdict)
    cols[1].metric("答案包含", contain)
    cols[2].metric("结束原因", t.termination_reason)
    cols[3].metric("步数 / 工具调用", f"{step_count(t)} / {t.tool_call_count}")
    cols[4].metric("花费", f"{t.total_cost:.4f}")
    cols[5].metric("耗时（秒）", "—" if t.latency_s is None else f"{t.latency_s:.1f}")
    if t.prompt_tokens is not None:
        st.caption(
            f"输入 {t.prompt_tokens:,} token（缓存 {t.cached_tokens or 0:,}），"
            f"输出 {t.completion_tokens or 0:,} token，LLM 调用 {t.llm_calls} 次，"
            f"峰值单次输入 {t.peak_prompt_tokens or '—'}"
        )
    if t.source == "converted":
        st.info(
            "这条轨迹来自记录器上线之前的运行，由 predictions.jsonl 转换而来：有工具调用、"
            "参数、结果和答案，没有每次 LLM 调用的 token、模型的思考文本和完整 messages。"
        )


def labelling(t: Trajectory, store: LabelStore, taxonomy: Taxonomy, info: RunInfo) -> None:
    current = store.get(t.question_id)
    suggested = suggest_tags(t, info.tools)
    st.markdown("**标注**")
    if suggested:
        st.caption(
            "建议标签（启发式，需人工确认）：" + "、".join(taxonomy.label(x) for x in suggested)
        )
    tags = st.multiselect(
        "标签",
        taxonomy.ids,
        default=current.tags if current else [],
        format_func=taxonomy.label,
        key=f"tags_{t.question_id}",
        help="\n".join(f"{x.label}：{x.description}" for x in taxonomy.tags),
    )
    note = st.text_area(
        "备注", value=current.note if current else "", key=f"note_{t.question_id}", height=80
    )
    if st.button("保存标注", key=f"save_{t.question_id}", type="primary"):
        saved = store.set(t.question_id, tags, note)
        st.success("已保存" if saved else "已清除该题的标注")
    elif current:
        st.caption(f"上次保存：{current.updated_at}")


def growth_charts(t: Trajectory) -> None:
    rows = context_growth(t)
    if not rows:
        return
    df = pd.DataFrame(rows)
    st.markdown("**上下文增长**（每次 LLM 调用的计费输入，按是否命中缓存拆分）")
    tokens = df.melt(
        id_vars=["step"], value_vars=["cached_tokens", "uncached_tokens"], var_name="kind"
    )
    tokens["kind"] = tokens["kind"].map({"cached_tokens": "命中缓存", "uncached_tokens": "未命中"})
    token_chart = (
        alt.Chart(tokens)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("step:O", title="步"),
            y=alt.Y("sum(value):Q", title="输入 token"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=["命中缓存", "未命中"], range=SERIES[:2]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            order=alt.Order("kind_order:Q"),  # cached at the bottom, uncached on top
            tooltip=[
                alt.Tooltip("step:O", title="步"),
                alt.Tooltip("kind:N", title="类型"),
                alt.Tooltip("value:Q", title="token", format=","),
            ],
        )
        .transform_calculate(kind_order="datum.kind === '命中缓存' ? 0 : 1")
        .properties(height=220)
    )
    roles = ["system", "user", "assistant", "tool"]
    chars = df.melt(id_vars=["step"], value_vars=[f"chars_{r}" for r in roles], var_name="role")
    chars["role"] = chars["role"].str.removeprefix("chars_")
    chars_chart = (
        alt.Chart(chars)
        .mark_bar(stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("step:O", title="步"),
            y=alt.Y("sum(value):Q", title="发送的字符数"),
            color=alt.Color(
                "role:N",
                scale=alt.Scale(domain=roles, range=SERIES),
                legend=alt.Legend(title=None, orient="top"),
            ),
            order=alt.Order("role_order:Q"),
            tooltip=[
                alt.Tooltip("step:O", title="步"),
                alt.Tooltip("role:N", title="角色"),
                alt.Tooltip("value:Q", title="字符", format=","),
            ],
        )
        .transform_calculate(role_order=f"indexof({json.dumps(roles)}, datum.role)")
        .properties(height=220)
    )
    left, right = st.columns(2)
    left.altair_chart(token_chart, width="stretch")
    right.altair_chart(chars_chart, width="stretch")
    with st.expander("数据表"):
        st.dataframe(df, hide_index=True, width="stretch")


def _step_title(s: Step) -> str:
    parts = [f"第 {s.index} 步 · {KIND_LABELS[s.kind]}"]
    if s.tool_calls:
        parts.append("、".join(c.tool_name for c in s.tool_calls))
    if s.llm is not None:
        parts.append(f"输入 {s.llm.prompt_tokens:,}（缓存 {s.llm.cached_tokens:,}）")
    if s.latency_ms is not None:
        parts.append(f"{s.latency_ms / 1000:.1f}s")
    return " · ".join(parts)


def steps_view(t: Trajectory) -> None:
    st.markdown("**逐步轨迹**")
    expand_all = st.toggle("全部展开", key="expand_all")
    for s in t.steps:
        with st.expander(_step_title(s), expanded=expand_all or s.kind != "tool_calls"):
            if s.llm is not None:
                c = s.llm.context
                chars = "，".join(f"{r} {n:,}" for r, n in c.chars.items())
                st.caption(
                    f"LLM：输入 {s.llm.prompt_tokens:,} / 缓存 {s.llm.cached_tokens:,} / "
                    f"输出 {s.llm.completion_tokens:,} token，{s.llm.latency_ms / 1000:.1f}s，"
                    f"花费 {s.llm.cost:.5f}；发送 {c.count} 条消息（字符：{chars}）"
                )
                if s.llm.error:
                    st.error(s.llm.error)
            if s.thought:
                st.markdown(s.thought)
            for call in s.tool_calls:
                timing = "" if call.latency_ms is None else f" · {call.latency_ms:.0f} ms"
                st.markdown(f"`{call.tool_name}`{timing}")
                st.json(call.tool_args, expanded=True)
                if call.log:
                    st.caption(", ".join(f"{k}={v}" for k, v in call.log.items()))
                st.code(call.tool_result or "（空）", language=None, wrap_lines=True)


def raw_view(t: Trajectory) -> None:
    if t.messages is not None:
        with st.expander(f"完整 messages（{len(t.messages)} 条，用于 SFT 数据）"):
            st.json(t.messages, expanded=False)
    if t.tools:
        with st.expander("工具定义"):
            st.json(t.tools, expanded=False)


def main() -> None:
    st.title("轨迹回放")
    runs = find_runs(ROOT)
    if not runs:
        st.warning(f"{ROOT} 下没有找到包含 predictions.jsonl 的运行目录")
        return
    run_dir = st.sidebar.selectbox(
        "运行", runs, format_func=lambda p: str(p.relative_to(ROOT)), key="run"
    )
    if run_dir is None:
        return
    info, trajectories = _load(str(run_dir), _stamp(run_dir))
    st.sidebar.caption(
        f"{info.method} · {info.model} · {len(trajectories)} 题 · "
        + ("有记录器轨迹" if info.recorded else "由预测结果转换")
    )
    taxonomy = Taxonomy.from_yaml()
    store = LabelStore(run_dir, taxonomy)
    shown = apply_filter(trajectories, store.load().labels, sidebar(trajectories, taxonomy))

    judged = [t for t in shown if t.correct is not None]
    acc = f"，其中正确 {sum(bool(t.correct) for t in judged)}/{len(judged)}" if judged else ""
    st.caption(f"共 {len(trajectories)} 题，筛选后 {len(shown)} 题{acc}")
    if not shown:
        st.info("没有符合筛选条件的题目")
        return
    overview(shown, store, taxonomy)
    t = pick(shown)
    header(t, chunk_refs(info.chunks_file))
    labelling(t, store, taxonomy, info)
    growth_charts(t)
    steps_view(t)
    raw_view(t)


main()
