"""Tool-call middleware of the agent (stage 11, D38): every tool call of the diagnosis loop goes
through :meth:`ToolHarness.call`, whichever backend runs it (in-process or MCP).

Per call, in order: permission (unknown or ``forbidden`` tools are refused; a tool must be
exposed to the intent), argument validation against the tool's schema, the per-turn budget
(total calls, document searches), duplicate detection (the same call again returns the
earlier result instead of running), the timeout, result truncation, and one audit log line.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from research_agent.toolkit.local import ToolBackend, ToolOutput
from research_agent.toolkit.specs import SPEC_BY_NAME, ToolSpec, exposed

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
}


@dataclass
class TurnBudget:
    max_calls: int = 16  # tool calls per turn
    max_doc_searches: int = 2  # search_docs calls per turn


def validate(spec: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
    """Arguments checked against the spec: required present, known keys only, types and
    ranges; defaults filled in. Raises ValueError with a message for the model."""
    if not isinstance(args, dict):
        raise ValueError("参数必须是 JSON 对象")  # noqa: TRY004 - one error type for the model
    missing = [k for k in spec.required if args.get(k) in (None, "")]
    if missing:
        raise ValueError(f"缺少参数：{', '.join(missing)}")
    out = dict(spec.defaults)
    for key, value in args.items():
        schema = spec.properties.get(key)
        if schema is None:
            continue  # unknown keys are dropped
        kind = schema.get("type", "string")
        if kind == "integer" and isinstance(value, str) and value.strip().isdigit():
            value = int(value)
        if not isinstance(value, _TYPES[kind]) or (
            kind in ("integer", "number") and isinstance(value, bool)
        ):
            raise ValueError(f"参数 {key} 应为 {kind}")
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"参数 {key} 只能是 {', '.join(schema['enum'])}")
        if kind in ("integer", "number"):
            low, high = schema.get("minimum"), schema.get("maximum")
            if (low is not None and value < low) or (high is not None and value > high):
                raise ValueError(f"参数 {key} 超出范围 [{low}, {high}]")
        if isinstance(value, list) and not all(isinstance(v, str) for v in value):
            raise ValueError(f"参数 {key} 应为字符串列表")
        out[key] = value
    return out


class ToolHarness:
    def __init__(
        self,
        backend: ToolBackend,
        audit_file: str | Path | None = None,
        budget: TurnBudget | None = None,
        specs: dict[str, ToolSpec] | None = None,
    ) -> None:
        self.backend = backend
        self.audit_file = Path(audit_file) if audit_file else None
        self.budget = budget or TurnBudget()
        self.specs = specs or SPEC_BY_NAME
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")
        self._lock = threading.Lock()
        self.start_turn()

    def schemas(self, intent: str) -> list[dict[str, Any]]:
        """Tool definitions exposed to ``intent`` (none for out_of_scope)."""
        return [s.schema() for s in exposed(intent) if s.name in self.specs]

    def start_turn(self, session: str = "") -> None:
        self.session = session
        self.calls = 0
        self.doc_searches = 0
        self._seen: dict[str, tuple[int, ToolOutput]] = {}

    def remaining(self) -> int:
        return self.budget.max_calls - self.calls

    def call(self, name: str, args: dict[str, Any], intent: str = "diagnosis") -> ToolOutput:
        start = time.perf_counter()
        spec = self.specs.get(name)
        status, out = "ok", None
        if spec is None:
            status, out = "refused", ToolOutput(f"拒绝：没有名为 {name} 的工具。", error=True)
        elif spec.permission == "forbidden":
            status, out = (
                "refused",
                ToolOutput(f"拒绝：{name} 不允许调用。{spec.description}", error=True),
            )
        elif intent not in spec.intents:
            status, out = (
                "refused",
                ToolOutput(f"拒绝：{name} 不对“{intent}”类问题开放。", error=True),
            )
        if out is None and spec is not None:
            try:
                args = validate(spec, args)
            except ValueError as e:
                status, out = "invalid", ToolOutput(f"参数错误：{e}", error=True)
        key = f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)}"
        if out is None and key in self._seen:
            n, earlier = self._seen[key]
            status = "duplicate"
            if earlier.pending is not None:  # never hand a proposal out twice
                out = ToolOutput(
                    f"这项修改本轮第 {n} 次调用时已经提出过，不再重复提出。", error=True
                )
            else:
                out = ToolOutput(
                    f"（与本轮第 {n} 次调用相同，没有重复执行，下面是上次的结果；"
                    "请基于已有信息继续或给出结论。）\n" + earlier.text
                )
        if out is None and self.calls >= self.budget.max_calls:
            status, out = (
                "over_budget",
                ToolOutput(
                    f"本轮工具调用预算（{self.budget.max_calls} 次）已用完，请根据已有信息给出结论。",
                    error=True,
                ),
            )
        if (
            out is None
            and name == "search_docs"
            and self.doc_searches >= self.budget.max_doc_searches
        ):
            status, out = (
                "over_budget",
                ToolOutput(
                    f"本轮文档检索次数（{self.budget.max_doc_searches} 次）已用完，请根据已有信息继续。",
                    error=True,
                ),
            )
        if out is None and spec is not None:
            self.calls += 1
            self.doc_searches += name == "search_docs"
            future = self._pool.submit(self.backend.call, name, args)
            try:
                out = future.result(timeout=spec.timeout_s)
                status = "error" if out.error else "ok"
            except FutureTimeout:
                status, out = (
                    "timeout",
                    ToolOutput(f"执行超时（{spec.timeout_s:.0f} 秒）。", error=True),
                )
            except Exception as e:  # noqa: BLE001 - a broken backend is reported to the model
                status, out = "error", ToolOutput(f"执行失败：{type(e).__name__}: {e}", error=True)
            if status == "ok":
                self._seen[key] = (self.calls, out)
        assert out is not None
        limit = spec.max_chars if spec is not None else 2000
        truncated = len(out.text) > limit
        if truncated:
            out.text = out.text[:limit] + f"…（结果共 {len(out.text)} 字，已截断）"
        self._audit(name, args, spec, status, start, len(out.text), truncated)
        return out

    def _audit(
        self,
        name: str,
        args: dict[str, Any],
        spec: ToolSpec | None,
        status: str,
        start: float,
        chars: int,
        truncated: bool,
    ) -> None:
        if self.audit_file is None:
            return
        row = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "session": self.session,
            "tool": name,
            "permission": spec.permission if spec else None,
            "args": args,
            "status": status,
            "ms": round((time.perf_counter() - start) * 1000),
            "chars": chars,
            "truncated": truncated,
            "calls_used": self.calls,
        }
        with self._lock:
            self.audit_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
