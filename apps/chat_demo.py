"""Chat demo of the TiDB operations assistant (stage 6).

Run with ``uv run streamlit run apps/chat_demo.py``. The assistant comes from
``configs/assistant.yaml`` (``CHAT_DEMO_CONFIG`` overrides it); with ``CHAT_DEMO_API_URL`` the
page talks to the HTTP service instead (stage 9, docker compose); ``CHAT_DEMO_FACTORY``
("module:function") replaces the factory, which tests use to inject fake services.

The page streams the events of a turn: the route, each tool call and result, the answer
with its citations, and for a write the statement with confirm / reject buttons.
"""

from __future__ import annotations

import importlib
import os
import uuid
from collections.abc import Iterator
from typing import Any

import streamlit as st

CONFIG = os.environ.get("CHAT_DEMO_CONFIG", "configs/assistant.yaml")
INTENT_LABELS = {
    "param_lookup": "参数查询",
    "research": "跨文档研究",
    "diagnosis": "故障诊断",
    "out_of_scope": "越界问题",
}

st.set_page_config(page_title="TiDB 运维助手", layout="wide")


@st.cache_resource(show_spinner="加载助手（嵌入模型和索引）…")
def get_app() -> Any:
    factory = os.environ.get("CHAT_DEMO_FACTORY")
    api = os.environ.get("CHAT_DEMO_API_URL")
    if api:
        from research_agent.service.client import ApiClient

        return ApiClient(api)
    if factory:
        module, name = factory.split(":")
        return getattr(importlib.import_module(module), name)()
    from research_agent.assistant.app import AssistantApp, AssistantConfig
    from research_agent.upstream import load_env

    load_env()
    return AssistantApp(AssistantConfig.from_yaml(CONFIG))


def describe(event: dict[str, Any]) -> str | None:
    """One progress line per event (None for events shown elsewhere)."""
    kind = event["type"]
    if kind == "route":
        return f"路由：**{INTENT_LABELS.get(event['intent'], event['intent'])}**（{event.get('reason', '')}）"
    if kind == "retrieval_start":
        return f"检索文档（{'A-RAG 多轮检索' if event['channel'] == 'arag' else '快速检索'}）：{event['question']}"
    if kind == "tool_call":
        return f"调用 `{event['tool']}` {event.get('args') or ''}"
    if kind == "tool_result":
        if event.get("tool") == "retrieval":
            return f"检索完成（{event.get('latency_s', 0)} 秒），引用 {len(event.get('citations') or [])} 处"
        text = str(event.get("result", ""))
        return f"`{event['tool']}` 返回：\n\n```\n{text[:800]}\n```"
    if kind == "error":
        return f":red[错误：{event['message']}]"
    return None


def render_answer(event: dict[str, Any]) -> None:
    st.markdown(event["text"])
    citations = event.get("citations") or []
    if citations:
        st.caption("引用：" + "；".join(dict.fromkeys(citations)))
    st.caption(f"花费 {event.get('cost', 0):.4f} 元")


def play(events: Iterator[dict[str, Any]]) -> None:
    """Show a turn's events as they arrive; remember the answer for the transcript."""
    with st.chat_message("assistant"):
        status = st.status("处理中…", expanded=True)
        answer: dict[str, Any] | None = None
        for event in events:
            line = describe(event)
            if line:
                status.markdown(line)
            if event["type"] == "answer":
                answer = event
            elif event["type"] == "confirm_required":
                status.update(label="等待确认", state="running", expanded=False)
            elif event["type"] == "done":
                status.update(
                    label=f"完成（{event['latency_s']} 秒）", state="complete", expanded=False
                )
        if answer is not None:
            render_answer(answer)
            st.session_state.transcript.append({"role": "assistant", "answer": answer})


def confirm_panel(app: Any, session: str) -> None:
    pending = app.pending(session)
    if not pending:
        return
    with st.container(border=True):
        st.warning("助手提出了一项修改，需要你确认后才会执行：")
        st.code(pending["statement"], language="sql")
        st.markdown(pending["impact"])
        if pending.get("rollback"):
            st.caption(f"回滚语句：`{pending['rollback']}`")
        approve, reject = st.columns(2)
        if approve.button("确认执行", type="primary", key="approve"):
            st.session_state.decision = True
            st.rerun()
        if reject.button("拒绝", key="reject"):
            st.session_state.decision = False
            st.rerun()


def main() -> None:
    app = get_app()
    st.session_state.setdefault("session", uuid.uuid4().hex[:12])
    st.session_state.setdefault("transcript", [])
    session = st.session_state.session
    st.title("TiDB 运维诊断助手")
    with st.sidebar:
        st.caption(f"会话 `{session}`")
        if st.button("新会话"):
            st.session_state.session = uuid.uuid4().hex[:12]
            st.session_state.transcript = []
            st.rerun()
        st.markdown("可以问：参数的默认值和含义、v8.1 与 v8.5 的差异，或者让我诊断本地集群的问题。")
    for item in st.session_state.transcript:
        with st.chat_message(item["role"]):
            if item["role"] == "user":
                st.markdown(item["text"])
            else:
                render_answer(item["answer"])
    decision = st.session_state.pop("decision", None)
    if decision is not None:
        st.session_state.transcript.append(
            {"role": "user", "text": "确认执行" if decision else "拒绝修改"}
        )
        with st.chat_message("user"):
            st.markdown("确认执行" if decision else "拒绝修改")
        play(app.confirm(session, decision))
    question = st.chat_input("输入问题")
    if question:
        st.session_state.transcript.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(question)
        play(app.ask(session, question))
    confirm_panel(app, session)


main()
