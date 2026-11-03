"""The assistant's main graph (LangGraph): routing, document retrieval, cluster diagnosis with
confirmed writes, grounding check, multi-turn sessions and streamed events.

    START → condense → route ─┬─ param_lookup / research → retrieve → check → END
                              ├─ diagnosis → diagnose ⇄ confirm (interrupt) ; diagnose → check
                              └─ out_of_scope → refuse → END

- ``condense`` rewrites a follow-up ("那这个参数在 8.1 里呢？") into a standalone question.
- ``route`` classifies the intent with a small model.
- ``retrieve`` asks a document channel (A-RAG, or the fast channel in stage 7).
- ``diagnose`` is a tool-calling loop over the diagnosis tools of the registry
  (``research_agent/toolkit``: cluster reads, logs, configuration, ``search_docs`` = A-RAG as
  one tool, the report, three ``propose_*`` tools). Every call goes through the
  :class:`ToolHarness` (permission, validation, budget, timeout, truncation, audit), whose
  backend runs in this process or behind an MCP server. A proposal stops the loop; ``confirm``
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
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from research_agent.assistant import prompts
from research_agent.assistant.retrieval import DocAnswer
from research_agent.assistant.routing import Channel, ChatClient, classify, parse_json
from research_agent.tidb.readonly import ReadOnlyTools
from research_agent.tidb.write import PendingWrite, WriteExecutor, WriteTools
from research_agent.toolkit.harness import ToolHarness
from research_agent.toolkit.local import EVIDENCE_STATUSES, InProcessTools


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
    logs: Any = None  # LogSearch
    configs: Any = None  # ConfigReader
    reports: Any = None  # ReportWriter
    tools: ToolHarness | None = None  # default: the in-process backend over these services
    choose_channel: Callable[[str, str], Channel] = lambda question, intent: "arag"
    max_steps: int = 12
    turn_timeout_s: float | None = None  # the diagnosis loop stops once a turn takes longer
    deadline: float | None = None  # set by Assistant at the start of each turn


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
    cluster_down: bool  # a cluster tool found no cluster
    citations: list[dict[str, Any]]
    answer: str
    cost: float
    steps: int  # LLM calls of the diagnosis loop in this turn


def _emit(event: str, **data: Any) -> None:
    get_stream_writer()({"type": event, **data})


def _text(reply: dict[str, Any]) -> str:
    return str(reply["message"].get("content") or "").strip()


def _cost(reply: dict[str, Any]) -> float:
    return float(reply.get("cost") or 0.0)


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
        found = classify(services.llm("router"), question)
        _emit("route", intent=found.intent, reason=found.reason, question=question)
        return {"intent": found.intent, "cost": state.get("cost", 0.0) + found.cost}

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

    harness = services.tools or ToolHarness(InProcessTools(services))

    def diagnose(state: AssistantState) -> dict[str, Any]:
        fresh = not state.get("diag_messages")
        messages = list(state.get("diag_messages") or []) or [
            {"role": "system", "content": prompts.DIAGNOSE},
            {"role": "user", "content": state["standalone"]},
        ]
        if fresh:
            harness.start_turn()
        evidence = state.get("evidence", 0)
        cluster_down = state.get("cluster_down", False)
        citations = list(state.get("citations") or [])
        cost = state.get("cost", 0.0)
        steps = state.get("steps", 0)
        llm = services.llm("agent")
        tools = harness.schemas("diagnosis")

        def done(answer: str, **extra: Any) -> dict[str, Any]:
            return {
                "diag_messages": messages,
                "answer": answer,
                "evidence": evidence,
                "cluster_down": cluster_down,
                "citations": citations,
                "cost": cost,
                "steps": steps,
                "pending_write": None,
                **extra,
            }

        while steps < services.max_steps:
            if services.deadline is not None and time.monotonic() > services.deadline:
                return done(prompts.DIAG_TIMEOUT)
            reply = llm.chat(messages, tools=tools, temperature=0.0)
            steps += 1
            cost += _cost(reply)
            message = reply["message"]
            calls = message.get("tool_calls") or []
            messages.append(
                {"role": "assistant", "content": message.get("content") or "", "tool_calls": calls}
                if calls
                else {"role": "assistant", "content": message.get("content") or ""}
            )
            if not calls:
                return done(_text(reply))
            pending: PendingWrite | None = None
            for call in calls:
                name = call["function"]["name"]
                args = parse_json(call["function"].get("arguments") or "{}")
                _emit("tool_call", tool=name, args=args)
                if name.startswith("propose_") and pending is not None:
                    result = "未处理：一次只能提出一项修改，请等这项修改确认后再提出下一项。"
                else:
                    out = harness.call(name, args, "diagnosis")
                    cost += out.cost
                    evidence += out.evidence
                    citations += out.citations
                    cluster_down = cluster_down or out.cluster_down
                    if out.pending is not None:
                        pending = PendingWrite.model_validate(out.pending)
                        pending_call = call["id"]
                        continue  # its tool result is added after the user decides
                    result = out.text
                _emit("tool_result", tool=name, result=result[:2000])
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            if pending is not None:
                return {
                    "diag_messages": messages,
                    "pending_write": {**pending.model_dump(), "tool_call_id": pending_call},
                    "evidence": evidence,
                    "citations": citations,
                    "cost": cost,
                    "steps": steps,
                }
            left = services.max_steps - steps
            if 0 < left <= 2 and messages[-1]["role"] == "tool":
                messages[-1]["content"] += prompts.DIAG_STEPS_LEFT.format(left=left)
        # Budget used up: one more call without tools for "the conclusion so far + what is
        # missing", instead of discarding what was gathered (writes can only be suggested).
        messages.append({"role": "user", "content": prompts.DIAG_STEP_LIMIT})
        try:
            reply = llm.chat(messages, tools=None, temperature=0.0)
            cost += _cost(reply)
            answer = _text(reply) or prompts.DIAG_NO_CONCLUSION
            messages.append({"role": "assistant", "content": answer})
        except Exception:  # noqa: BLE001 - the canned message is still an answer
            answer = prompts.DIAG_NO_CONCLUSION
        return done(answer)

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
            {"role": "tool", "tool_call_id": call_id, "content": outcome + prompts.AFTER_CONFIRM},
        ]
        return {
            "diag_messages": messages,
            "pending_write": None,
            "writes": [*(state.get("writes") or []), *results],
        }

    def check(state: AssistantState) -> dict[str, Any]:
        answer = state.get("answer") or ""
        if answer == prompts.DIAG_TIMEOUT:
            pass  # a stopped diagnosis says so; it is not an answer without evidence
        elif state.get("evidence", 0) == 0 and state.get("cluster_down"):
            answer = prompts.NO_CLUSTER
        elif state.get("evidence", 0) == 0:
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
            steps=state.get("steps", 0),
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
        timeout = self.services.turn_timeout_s
        self.services.deadline = time.monotonic() + timeout if timeout else None
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
            "cluster_down": False,
            "steps": 0,
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
