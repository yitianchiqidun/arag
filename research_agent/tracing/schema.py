"""Trajectory data model: one record per question, stored as ``trajectories.jsonl``.

A step is one LLM call together with the tool calls it requested (a response may request
several tools at once), so per-call token counts, cache hits and the size of the context
that was sent can be read off each step. Field names follow the OpenAI usage block:
``prompt_tokens`` is the plan's ``tokens_in``, ``completion_tokens`` its ``tokens_out``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

StepKind = Literal["tool_calls", "answer", "forced_answer", "llm_error", "retrieval"]


class MessageStats(BaseModel):
    """Size of the message list sent to the LLM in one call."""

    count: int
    # Characters per role (system / user / assistant / tool). Assistant tool calls count
    # their function name and JSON arguments, since those are sent back on every call too.
    chars: dict[str, int]

    @property
    def total_chars(self) -> int:
        return sum(self.chars.values())


class LLMCall(BaseModel):
    latency_ms: float
    attempts: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost: float
    finish_reason: str | None = None
    error: str | None = None
    tools_offered: bool  # False for a forced final answer (and for naive RAG)
    context: MessageStats


class ToolCall(BaseModel):
    call_id: str | None = None  # id from the assistant message; None for converted runs
    tool_name: str
    tool_args: dict[str, Any]
    tool_result: str
    latency_ms: float | None = None
    retrieved_tokens: int | None = None
    # Remaining fields of the upstream tool log (chunks_found, new_chunks_count, error, ...).
    log: dict[str, Any] = Field(default_factory=dict)


class Step(BaseModel):
    index: int  # 1-based position in the trajectory
    kind: StepKind
    thought: str | None = None  # assistant text sent alongside the tool calls / the answer
    llm: LLMCall | None = None  # None for a retrieval step or a converted run
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @property
    def latency_ms(self) -> float | None:
        parts = [self.llm.latency_ms] if self.llm else []
        parts += [t.latency_ms for t in self.tool_calls if t.latency_ms is not None]
        return sum(parts) if parts else None


class Trajectory(BaseModel):
    schema_version: int = SCHEMA_VERSION
    # "recorded": written while the question ran. "converted": rebuilt from a run that
    # predates the recorder (stage 0/1), so per-call fields and the messages are missing.
    source: Literal["recorded", "converted"]
    question_id: str
    question: str
    gold_answer: str | list[str]
    method: str
    model: str
    steps: list[Step]
    final_answer: str
    termination_reason: str
    total_cost: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    llm_calls: int | None = None
    latency_s: float | None = None
    # Judged outcome; not known while recording, filled in from predictions.jsonl on load.
    correct: bool | None = None
    contain_correct: bool | None = None
    # The exact conversation (OpenAI chat format) and tool schemas, kept for SFT data.
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None

    @property
    def tool_call_count(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps if s.kind != "retrieval")

    @property
    def peak_prompt_tokens(self) -> int | None:
        values = [s.llm.prompt_tokens for s in self.steps if s.llm and s.llm.error is None]
        return max(values) if values else None


def message_stats(messages: list[dict[str, Any]]) -> MessageStats:
    chars: dict[str, int] = {}
    for message in messages:
        role = str(message.get("role", "unknown"))
        size = _content_chars(message.get("content"))
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            size += len(function.get("name") or "") + len(function.get("arguments") or "")
        chars[role] = chars.get(role, 0) + size
    return MessageStats(count=len(messages), chars=chars)


def _content_chars(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):  # multi-part content: count the text parts
        return sum(len(p.get("text") or "") for p in content if isinstance(p, dict))
    return len(str(content))
