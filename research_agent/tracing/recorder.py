"""Record a trajectory while one question runs, without changing the upstream agent loop.

Two hooks collect what upstream ``BaseAgent.run`` does not return:

- ``MeteredLLMClient.observer`` sees every LLM call with the exact message list that was
  sent, its usage and the assistant message that came back;
- ``TimedToolRegistry`` wraps the shared tool registry for one agent and times each call.

``TraceRecorder.build`` then merges them with the upstream result (tool arguments, results
and logs, in call order) into a :class:`Trajectory`. Every join is checked, so a change in
the upstream loop fails loudly instead of producing a silently misaligned trace.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from arag.tools.registry import ToolRegistry
from research_agent.llm.client import LLMCallRecord
from research_agent.tracing.schema import (
    LLMCall,
    MessageStats,
    Step,
    StepKind,
    ToolCall,
    Trajectory,
    message_stats,
)

if TYPE_CHECKING:
    from arag.core.context import AgentContext

# Keys of an upstream trajectory entry that are not part of the tool log.
_ENTRY_KEYS = {"loop", "tool_name", "arguments", "tool_result", "retrieved_tokens", "latency_ms"}


@dataclass
class CallCapture:
    record: LLMCallRecord
    tools_offered: bool
    context: MessageStats
    sent: list[dict[str, Any]]  # the live list passed to chat(); the agent keeps appending
    sent_ids: list[int]  # identity of the messages actually sent, to verify the prefix later
    response: dict[str, Any] | None


@dataclass
class TraceRecorder:
    """Collects one question's LLM calls and tool timings; one instance per agent."""

    agent_has_tools: bool
    calls: list[CallCapture] = field(default_factory=list)
    tool_timings: list[tuple[str, float]] = field(default_factory=list)
    tool_schemas: list[dict[str, Any]] | None = None

    def on_llm_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        record: LLMCallRecord,
        response: dict[str, Any] | None,
    ) -> None:
        if tools and self.tool_schemas is None:
            self.tool_schemas = tools
        self.calls.append(
            CallCapture(
                record=record,
                tools_offered=bool(tools),
                context=message_stats(messages),
                sent=messages,
                sent_ids=[id(m) for m in messages],
                response=response,
            )
        )

    def on_tool(self, name: str, latency_ms: float) -> None:
        self.tool_timings.append((name, latency_ms))

    def conversation(self) -> list[dict[str, Any]]:
        """The full message list: everything sent in the last call plus its reply.

        Upstream appends to one list in place and never edits earlier messages; this is
        verified, because a recorder for a loop that rewrites history (e.g. stage 5e) would
        need to store every request instead of only the final list.
        """
        if not self.calls:
            return []
        last = self.calls[-1]
        final = list(last.sent)
        for capture in self.calls:
            if [id(m) for m in final[: len(capture.sent_ids)]] != capture.sent_ids:
                raise ValueError("the message history was modified between LLM calls")
        if last.response is not None and (not final or final[-1] is not last.response):
            final.append(last.response)
        return final

    def build(self, record: dict[str, Any], method: str, model: str) -> Trajectory:
        """Merge the captured calls with an enriched prediction record (runner output).

        If the agent raised (upstream then records ``error`` and an empty trajectory), the
        requested tool calls are kept without results instead of failing the join.
        """
        entries = list(record.get("trajectory") or [])
        strict = "error" not in record
        steps: list[Step] = []
        if entries and entries[0].get("tool_name") == "dense_retrieval":  # naive RAG
            steps.append(Step(index=1, kind="retrieval", tool_calls=[_tool_call(entries[0])]))
            entries = entries[1:]

        timings = list(self.tool_timings)
        loop = 0
        for capture in self.calls:
            rec = capture.record
            response = capture.response or {}
            requested = response.get("tool_calls") or []
            if capture.tools_offered:
                loop += 1
            kind: StepKind
            if rec.error is not None:
                kind = "llm_error"
            elif requested:
                kind = "tool_calls"
            elif self.agent_has_tools and not capture.tools_offered:
                kind = "forced_answer"
            else:
                kind = "answer"

            tool_calls: list[ToolCall] = []
            for call in requested:
                if not strict and not entries:
                    tool_calls.append(_unrecorded_tool_call(call))
                    continue
                if not entries or not timings:
                    raise ValueError("fewer executed tool calls than the LLM requested")
                entry, (timed_name, latency_ms) = entries.pop(0), timings.pop(0)
                name = (call.get("function") or {}).get("name")
                if not (name == entry.get("tool_name") == timed_name):
                    raise ValueError(f"tool call order mismatch: {name} / {entry.get('tool_name')}")
                if entry.get("loop") != loop:
                    raise ValueError(f"tool call of loop {entry.get('loop')} found in loop {loop}")
                tool_calls.append(_tool_call(entry, call.get("id"), latency_ms))

            steps.append(
                Step(
                    index=len(steps) + 1,
                    kind=kind,
                    thought=response.get("content") or None,
                    llm=LLMCall(
                        latency_ms=rec.latency_ms,
                        attempts=rec.attempts,
                        prompt_tokens=rec.prompt_tokens,
                        completion_tokens=rec.completion_tokens,
                        cached_tokens=rec.cached_tokens,
                        cost=rec.cost,
                        finish_reason=rec.finish_reason,
                        error=rec.error,
                        tools_offered=capture.tools_offered,
                        context=capture.context,
                    ),
                    tool_calls=tool_calls,
                )
            )
        if strict and (entries or timings):
            raise ValueError(f"{len(entries)} tool results were not matched to an LLM call")

        return Trajectory(
            source="recorded",
            question_id=str(record["qid"]),
            question=record["question"],
            gold_answer=record["gold_answer"],
            method=method,
            model=model,
            steps=steps,
            final_answer=record.get("pred_answer") or "",
            termination_reason=record.get("termination", "unknown"),
            total_cost=record.get("total_cost", 0.0),
            prompt_tokens=record.get("input_tokens"),
            completion_tokens=record.get("output_tokens"),
            cached_tokens=record.get("cached_tokens"),
            llm_calls=record.get("llm_calls"),
            latency_s=record.get("latency_s"),
            messages=self.conversation(),
            tools=self.tool_schemas,
        )


def _tool_call(
    entry: dict[str, Any], call_id: str | None = None, latency_ms: float | None = None
) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        tool_name=entry["tool_name"],
        tool_args=entry.get("arguments") or {},
        tool_result=entry.get("tool_result") or "",
        latency_ms=latency_ms if latency_ms is not None else entry.get("latency_ms"),
        retrieved_tokens=entry.get("retrieved_tokens"),
        log={k: v for k, v in entry.items() if k not in _ENTRY_KEYS},
    )


def _unrecorded_tool_call(call: dict[str, Any]) -> ToolCall:
    function = call.get("function") or {}
    try:
        args = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}
    return ToolCall(
        call_id=call.get("id"),
        tool_name=function.get("name") or "",
        tool_args=args if isinstance(args, dict) else {},
        tool_result="",
        log={"unrecorded": True},
    )


class TimedToolRegistry(ToolRegistry):
    """Per-agent view of a shared registry that reports how long each tool call took."""

    def __init__(self, inner: ToolRegistry, recorder: TraceRecorder) -> None:
        super().__init__()
        self._tools = inner._tools  # shared, read-only: the tools themselves are not copied
        self.recorder = recorder

    def execute(
        self, name: str, context: AgentContext, **kwargs: Any
    ) -> tuple[str, dict[str, Any]]:
        start = time.perf_counter()
        try:
            return super().execute(name, context, **kwargs)
        finally:
            self.recorder.on_tool(name, (time.perf_counter() - start) * 1000)
