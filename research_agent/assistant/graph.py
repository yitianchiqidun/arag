"""The assistant's main graph (LangGraph): routing, document retrieval, cluster diagnosis with
confirmed writes, grounding check, multi-turn sessions and streamed events.

    START → condense → route ─┬─ param_lookup / research → retrieve → check → END
                              ├─ diagnosis → diagnose ⇄ confirm (interrupt) ; diagnose → check
                              └─ out_of_scope → refuse → END

- ``condense`` rewrites a follow-up ("那这个参数在 8.1 里呢？") into a standalone question.
- ``route`` classifies the intent with a small model.
- ``retrieve`` asks a document channel (A-RAG, or the fast channel in stage 7).
- ``diagnose`` is a tool-calling loop over the read-only cluster tools, ``search_docs`` (A-RAG
  as one tool) and three ``propose_*`` write tools. A proposal stops the loop; ``confirm``
  pauses the graph with ``interrupt()`` until the user approves or rejects it, runs it only if
  approved, and returns the outcome to the loop.
- ``check`` refuses to answer without evidence (no citation and no cluster data).

Sessions are threads of a checkpointer (SQLite in the service), so a follow-up sees the
history and a pending confirmation survives between requests. Nodes emit events through the
LangGraph stream writer: route, retrieval_start, tool_call, tool_result, confirm_required,
answer, error (``done`` is added by :class:`Assistant`).
"""

from __future__ import annotations

import json
import operator
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from research_agent.assistant import prompts
from research_agent.assistant.retrieval import Channel, DocAnswer
from research_agent.tidb.db import DatabaseError
from research_agent.tidb.readonly import ReadOnlyTools, UnsafeSQL
from research_agent.tidb.write import PendingWrite, WriteExecutor, WriteTools

Intent = Literal["param_lookup", "research", "diagnosis", "out_of_scope"]
INTENTS: tuple[Intent, ...] = ("param_lookup", "research", "diagnosis", "out_of_scope")
EVIDENCE_STATUSES = ("read", "snippet")


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]: ...


class DocSource(Protocol):
    def answer(self, question: str, channel: Channel = "arag") -> DocAnswer: ...


@dataclass
class Services:
    """What the nodes call; tests pass fakes."""

    llm: Callable[[str], ChatClient]  # role ("router" or "agent") -> a client
    docs: DocSource
    reader: ReadOnlyTools | None = None  # None: no cluster
    writes: WriteTools | None = None
    executor: WriteExecutor | None = None
    choose_channel: Callable[[str, str], Channel] = lambda question, intent: "arag"
    max_steps: int = 12


class AssistantState(TypedDict, total=False):
    history: Annotated[list[dict[str, str]], operator.add]  # user / assistant turns
    question: str
    standalone: str
    intent: str
    channel: str
    doc: dict[str, Any] | None  # DocAnswer of the retrieve node
    diag_messages: list[dict[str, Any]]  # the diagnosis loop's conversation
    pending_write: dict[str, Any] | None
    writes: list[dict[str, Any]]  # WriteResult of this turn
    evidence: int  # cluster tool results and cited chunks behind the answer
    citations: list[dict[str, Any]]
    answer: str
    cost: float


def _emit(event: str, **data: Any) -> None:
    get_stream_writer()({"type": event, **data})


def _text(reply: dict[str, Any]) -> str:
    return str(reply["message"].get("content") or "").strip()


def _cost(reply: dict[str, Any]) -> float:
    return float(reply.get("cost") or 0.0)


def _json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start : end + 1]) if start >= 0 else {}
    except json.JSONDecodeError:
        data = {}
    return data if isinstance(data, dict) else {}


# --- diagnosis tools ------------------------------------------------------------------------


def _fn(
    name: str, description: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_STR = {"type": "string"}
_INT = {"type": "integer"}
TOOLS = [
    _fn(
        "search_docs",
        "查 TiDB 官方文档（多轮检索，较慢），返回带引用的回答。",
        {"question": _STR},
        ["question"],
    ),
    _fn(
        "run_sql",
        "执行一条只读 SQL（SELECT / SHOW / EXPLAIN / DESC），最多返回 200 行。",
        {"sql": _STR},
        ["sql"],
    ),
    _fn(
        "show_variables",
        "查看系统变量的全局值，pattern 是 SQL LIKE 模式。",
        {"pattern": _STR},
        ["pattern"],
    ),
    _fn(
        "slow_queries",
        "最近的慢查询（耗时、扫描行数、索引、SQL）。",
        {"limit": _INT, "min_seconds": {"type": "number"}},
        [],
    ),
    _fn(
        "explain",
        "查看 SELECT 的执行计划；analyze=true 会实际执行并给出实际行数。",
        {"sql": _STR, "analyze": {"type": "boolean"}},
        ["sql"],
    ),
    _fn("lock_waits", "当前的悲观锁等待：等待方、持有方事务及其 SQL。", {}, []),
    _fn("transactions", "正在运行的事务，按开始时间排序。", {}, []),
    _fn("deadlocks", "最近的死锁记录。", {"limit": _INT}, []),
    _fn("statement_summary", "总耗时最多的 SQL 语句（statements_summary）。", {"limit": _INT}, []),
    _fn(
        "table_stats",
        "表的统计信息健康度和修改行数，判断统计信息是否过期。",
        {"table": _STR},
        ["table"],
    ),
    _fn("table_schema", "表结构和索引（SHOW CREATE TABLE）。", {"table": _STR}, ["table"]),
    _fn("bindings", "全局 SQL 绑定（可能强制执行计划）。", {}, []),
    _fn("cluster_info", "集群各组件的版本和运行时间。", {}, []),
    _fn(
        "propose_set_global",
        "提出修改系统变量的全局值（需要用户确认后才执行）。",
        {"name": _STR, "value": _STR, "reason": _STR},
        ["name", "value", "reason"],
    ),
    _fn(
        "propose_analyze",
        "提出对表执行 ANALYZE TABLE 重新收集统计信息（需要用户确认）。",
        {"table": _STR, "reason": _STR},
        ["table", "reason"],
    ),
    _fn(
        "propose_create_index",
        "提出在表上创建索引（需要用户确认）。",
        {"table": _STR, "columns": {"type": "array", "items": _STR}, "reason": _STR},
        ["table", "columns", "reason"],
    ),
]


def _read_tool(reader: ReadOnlyTools, name: str, args: dict[str, Any]) -> str:
    calls: dict[str, Callable[[], Any]] = {
        "run_sql": lambda: reader.run_sql(str(args["sql"])),
        "show_variables": lambda: reader.system_variables(str(args["pattern"])),
        "slow_queries": lambda: reader.slow_queries(
            int(args.get("limit", 10)), float(args.get("min_seconds", 0))
        ),
        "explain": lambda: reader.explain(str(args["sql"]), bool(args.get("analyze", False))),
        "lock_waits": reader.lock_waits,
        "transactions": reader.transactions,
        "deadlocks": lambda: reader.deadlocks(int(args.get("limit", 10))),
        "statement_summary": lambda: reader.statement_summary(int(args.get("limit", 10))),
        "table_stats": lambda: reader.table_stats(str(args["table"])),
        "table_schema": lambda: reader.table_schema(str(args["table"])),
        "bindings": reader.bindings,
        "cluster_info": reader.cluster_info,
    }
    return str(calls[name]().as_text())


def _propose(writes: WriteTools, name: str, args: dict[str, Any]) -> PendingWrite:
    if name == "propose_set_global":
        return writes.propose_set_global(str(args["name"]), str(args["value"]))
    if name == "propose_analyze":
        return writes.propose_analyze(str(args["table"]))
    return writes.propose_create_index(str(args["table"]), [str(c) for c in args["columns"]])


# --- graph ----------------------------------------------------------------------------------


def build_graph(services: Services, checkpointer: Any) -> Any:
    def condense(state: AssistantState) -> dict[str, Any]:
        question = state["question"]
        history = state.get("history") or []
        if not history:
            return {"standalone": question}
        turns = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])
        reply = services.llm("router").chat(
            [
                {
                    "role": "user",
                    "content": prompts.CONDENSE.format(history=turns, question=question),
                }
            ],
            temperature=0.0,
        )
        return {
            "standalone": _text(reply) or question,
            "cost": state.get("cost", 0.0) + _cost(reply),
        }

    def route(state: AssistantState) -> dict[str, Any]:
        question = state["standalone"]
        reply = services.llm("router").chat(
            [{"role": "user", "content": prompts.ROUTE.format(question=question)}], temperature=0.0
        )
        data = _json(_text(reply))
        intent = data.get("intent") if data.get("intent") in INTENTS else "research"
        _emit("route", intent=intent, reason=str(data.get("reason") or ""), question=question)
        return {"intent": intent, "cost": state.get("cost", 0.0) + _cost(reply)}

    def retrieve(state: AssistantState) -> dict[str, Any]:
        question = state["standalone"]
        channel = services.choose_channel(question, state["intent"])
        _emit("retrieval_start", channel=channel, question=question)
        doc = services.docs.answer(question, channel)
        _emit(
            "tool_result",
            tool="retrieval",
            channel=channel,
            citations=[c.label for c in doc.citations],
            latency_s=doc.latency_s,
        )
        cited = [c for c in doc.citations if c.status in EVIDENCE_STATUSES]
        return {
            "channel": channel,
            "doc": doc.model_dump(),
            "answer": doc.answer,
            "citations": [c.model_dump() for c in doc.citations],
            "evidence": len(cited),
            "cost": state.get("cost", 0.0) + doc.cost,
        }

    def diagnose(state: AssistantState) -> dict[str, Any]:
        messages = list(state.get("diag_messages") or []) or [
            {"role": "system", "content": prompts.DIAGNOSE},
            {"role": "user", "content": state["standalone"]},
        ]
        evidence = state.get("evidence", 0)
        citations = list(state.get("citations") or [])
        cost = state.get("cost", 0.0)
        llm = services.llm("agent")
        for _ in range(services.max_steps):
            reply = llm.chat(messages, tools=TOOLS, temperature=0.0)
            cost += _cost(reply)
            message = reply["message"]
            calls = message.get("tool_calls") or []
            messages.append(
                {"role": "assistant", "content": message.get("content") or "", "tool_calls": calls}
                if calls
                else {"role": "assistant", "content": message.get("content") or ""}
            )
            if not calls:
                return {
                    "diag_messages": messages,
                    "answer": _text(reply),
                    "evidence": evidence,
                    "citations": citations,
                    "cost": cost,
                    "pending_write": None,
                }
            pending: PendingWrite | None = None
            for call in calls:
                name = call["function"]["name"]
                args = _json(call["function"].get("arguments") or "{}")
                _emit("tool_call", tool=name, args=args)
                if name.startswith("propose_") and pending is None:
                    try:
                        if services.writes is None:
                            raise UnsafeSQL("集群不可用，无法提出修改")
                        pending = _propose(services.writes, name, args)
                        pending.impact = f"{pending.impact}\n原因：{args.get('reason', '')}"
                        pending_call = call["id"]
                        continue  # its tool result is added after the user decides
                    except (UnsafeSQL, DatabaseError) as e:
                        result = f"无法提出修改：{e}"
                elif name.startswith("propose_"):
                    result = "未处理：一次只能提出一项修改，请等这项修改确认后再提出下一项。"
                elif name == "search_docs":
                    doc = services.docs.answer(
                        str(args.get("question") or state["standalone"]), "arag"
                    )
                    cost += doc.cost
                    labels = [f"Chunk {c.chunk_id}（{c.label}）" for c in doc.citations]
                    citations += [c.model_dump() for c in doc.citations]
                    evidence += sum(c.status in EVIDENCE_STATUSES for c in doc.citations)
                    result = doc.answer + ("\n\n引用：" + "；".join(labels) if labels else "")
                elif services.reader is None:
                    result = "集群不可用：本地 TiDB 集群没有启动或无法连接。"
                else:
                    try:
                        result = _read_tool(services.reader, name, args)
                        evidence += 1
                    except (UnsafeSQL, DatabaseError, KeyError, ValueError) as e:
                        result = f"执行失败：{e}"
                _emit("tool_result", tool=name, result=result[:2000])
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            if pending is not None:
                return {
                    "diag_messages": messages,
                    "pending_write": {**pending.model_dump(), "tool_call_id": pending_call},
                    "evidence": evidence,
                    "citations": citations,
                    "cost": cost,
                }
        return {
            "diag_messages": messages,
            "answer": "诊断步骤达到上限，未能得出结论。",
            "evidence": evidence,
            "citations": citations,
            "cost": cost,
            "pending_write": None,
        }

    def confirm(state: AssistantState) -> dict[str, Any]:
        pending = dict(state["pending_write"] or {})
        call_id = pending.pop("tool_call_id")
        write = PendingWrite.model_validate(pending)
        # Pauses here; resumed with Command(resume={"approved": bool}). Nothing runs before it.
        decision = interrupt({"type": "confirm_required", "write": write.model_dump()})
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        if services.executor is None:
            outcome = "集群不可用，修改没有执行。"
            results: list[dict[str, Any]] = []
        else:
            result = services.executor.execute(write, confirmed=approved)
            results = [result.model_dump()]
            if not approved:
                outcome = "用户拒绝了这项修改，没有执行。"
            elif result.executed:
                outcome = f"已执行：{write.statement}。执行后状态：{json.dumps(result.after, ensure_ascii=False, default=str)[:1500]}"
            else:
                outcome = f"执行失败：{result.error}"
        _emit("tool_result", tool="confirm", approved=approved, result=outcome)
        messages = [
            *state["diag_messages"],
            {"role": "tool", "tool_call_id": call_id, "content": outcome},
        ]
        return {
            "diag_messages": messages,
            "pending_write": None,
            "writes": [*(state.get("writes") or []), *results],
        }

    def check(state: AssistantState) -> dict[str, Any]:
        answer = state.get("answer") or ""
        if state.get("evidence", 0) == 0:
            answer = prompts.ANSWER_PREFIX_NO_EVIDENCE + (
                "\n\n（检索得到的回答没有引用任何文档段落，因此不予采用。）" if answer else ""
            )
        _emit(
            "answer",
            text=answer,
            intent=state.get("intent"),
            channel=state.get("channel"),
            citations=[c.get("label") for c in state.get("citations") or []],
            cost=round(state.get("cost", 0.0), 6),
        )
        return {
            "answer": answer,
            "history": [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": answer},
            ],
        }

    def refuse(state: AssistantState) -> dict[str, Any]:
        _emit(
            "answer",
            text=prompts.OUT_OF_SCOPE,
            intent="out_of_scope",
            citations=[],
            cost=round(state.get("cost", 0.0), 6),
        )
        return {
            "answer": prompts.OUT_OF_SCOPE,
            "history": [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": prompts.OUT_OF_SCOPE},
            ],
        }

    def after_route(state: AssistantState) -> str:
        return {"diagnosis": "diagnose", "out_of_scope": "refuse"}.get(state["intent"], "retrieve")

    def after_diagnose(state: AssistantState) -> str:
        return "confirm" if state.get("pending_write") else "check"

    graph = StateGraph(AssistantState)
    for name, node in [
        ("condense", condense),
        ("route", route),
        ("retrieve", retrieve),
        ("diagnose", diagnose),
        ("confirm", confirm),
        ("check", check),
        ("refuse", refuse),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "condense")
    graph.add_edge("condense", "route")
    graph.add_conditional_edges("route", after_route, ["retrieve", "diagnose", "refuse"])
    graph.add_edge("retrieve", "check")
    graph.add_conditional_edges("diagnose", after_diagnose, ["confirm", "check"])
    graph.add_edge("confirm", "diagnose")
    graph.add_edge("check", END)
    graph.add_edge("refuse", END)
    return graph.compile(checkpointer=checkpointer)


class Assistant:
    """Runs turns of a session and turns the graph's stream into events."""

    def __init__(self, services: Services, checkpointer: Any) -> None:
        self.services = services
        self.graph = build_graph(services, checkpointer)

    def _config(self, session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id}}

    def _run(self, payload: Any, session_id: str) -> Iterator[dict[str, Any]]:
        start = time.perf_counter()
        try:
            for mode, chunk in self.graph.stream(
                payload, self._config(session_id), stream_mode=["custom", "updates"]
            ):
                if mode == "custom":
                    yield chunk
                elif "__interrupt__" in chunk:
                    for item in chunk["__interrupt__"]:
                        yield dict(item.value)
        except Exception as e:  # noqa: BLE001 - any node failure ends the stream with an error event
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        yield {"type": "done", "latency_s": round(time.perf_counter() - start, 2)}

    def ask(self, session_id: str, question: str) -> Iterator[dict[str, Any]]:
        if self.pending(session_id) is not None:
            yield {"type": "error", "message": "还有一项修改等待确认，请先确认或拒绝。"}
            yield {"type": "done", "latency_s": 0.0}
            return
        turn: dict[str, Any] = {
            "question": question,
            "standalone": "",
            "intent": "",
            "channel": "",
            "doc": None,
            "diag_messages": [],
            "pending_write": None,
            "writes": [],
            "evidence": 0,
            "citations": [],
            "answer": "",
            "cost": 0.0,
        }
        yield from self._run(turn, session_id)

    def confirm(self, session_id: str, approved: bool) -> Iterator[dict[str, Any]]:
        if self.pending(session_id) is None:
            yield {"type": "error", "message": "没有等待确认的修改。"}
            yield {"type": "done", "latency_s": 0.0}
            return
        yield from self._run(Command(resume={"approved": approved}), session_id)

    def pending(self, session_id: str) -> dict[str, Any] | None:
        state = self.graph.get_state(self._config(session_id))
        if not state.next:
            return None
        return dict(state.values.get("pending_write") or {}) or None

    def history(self, session_id: str) -> list[dict[str, str]]:
        values = self.graph.get_state(self._config(session_id)).values
        return list(values.get("history") or [])
