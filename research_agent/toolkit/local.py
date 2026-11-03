"""In-process tool backend (stage 10): runs every tool of the registry against the services of
this process. The MCP server wraps the same backend, so both tool sources behave alike.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from research_agent.tidb.db import DatabaseError
from research_agent.tidb.readonly import UnsafeSQL

EVIDENCE_STATUSES = ("read", "snippet")


@dataclass
class ToolOutput:
    text: str
    evidence: int = 0  # cluster results and readable citations behind it
    citations: list[dict[str, Any]] = field(default_factory=list)  # DocCitation dumps
    pending: dict[str, Any] | None = None  # PendingWrite dump of a propose_* call
    cost: float = 0.0  # money spent inside the tool (search_docs)
    cluster_down: bool = False
    error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolBackend(Protocol):
    def call(self, name: str, args: dict[str, Any]) -> ToolOutput: ...


class InProcessTools:
    """Dispatches by tool name. ``source`` has the attributes ``reader``, ``writes``, ``docs``,
    ``logs``, ``configs`` and ``reports`` (e.g. the graph's Services); they are read at call
    time, and a missing one turns into a clear message instead of an exception."""

    def __init__(self, source: Any) -> None:
        self.source = source

    def _get(self, name: str) -> Any:
        return getattr(self.source, name, None)

    def call(self, name: str, args: dict[str, Any]) -> ToolOutput:
        try:
            if name == "search_docs":
                return self._docs(str(args["question"]))
            if name.startswith("propose_"):
                return self._propose(name, args)
            if name == "search_logs":
                return self._logs(args)
            if name == "show_config":
                return self._config(args)
            if name == "export_report":
                reports = self._get("reports")
                if reports is None:
                    return ToolOutput("报告目录不可用。", error=True)
                return ToolOutput(reports.export(**args))
            return self._read(name, args)
        except (UnsafeSQL, DatabaseError, KeyError, ValueError, OSError) as e:
            return ToolOutput(f"执行失败：{e}", error=True)

    def _docs(self, question: str) -> ToolOutput:
        doc = self._get("docs").answer(question, "arag")
        labels = [f"Chunk {c.chunk_id}（{c.label}）" for c in doc.citations]
        return ToolOutput(
            doc.answer + ("\n\n引用：" + "；".join(labels) if labels else ""),
            evidence=sum(c.status in EVIDENCE_STATUSES for c in doc.citations),
            citations=[c.model_dump() for c in doc.citations],
            cost=doc.cost,
        )

    def _propose(self, name: str, args: dict[str, Any]) -> ToolOutput:
        writes = self._get("writes")
        if writes is None:
            return ToolOutput("无法提出修改：集群不可用。", cluster_down=True, error=True)
        if name == "propose_set_global":
            pending = writes.propose_set_global(str(args["name"]), str(args["value"]))
        elif name == "propose_analyze":
            pending = writes.propose_analyze(str(args["table"]))
        else:
            pending = writes.propose_create_index(
                str(args["table"]), [str(c) for c in args["columns"]]
            )
        pending.impact = f"{pending.impact}\n原因：{args.get('reason', '')}"
        return ToolOutput(f"{pending.statement}\n{pending.impact}", pending=pending.model_dump())

    def _logs(self, args: dict[str, Any]) -> ToolOutput:
        logs = self._get("logs")
        if logs is None:
            return ToolOutput("日志目录不可用（服务不在集群所在的机器上运行）。", error=True)
        text = logs.search(
            str(args["component"]), str(args.get("keyword", "")), int(args.get("max_lines", 50))
        )
        return ToolOutput(text, evidence=1)

    def _config(self, args: dict[str, Any]) -> ToolOutput:
        configs = self._get("configs")
        if configs is None:
            return ToolOutput("集群不可用：读不到组件配置。", cluster_down=True, error=True)
        result = configs.show_config(str(args["component"]), str(args.get("name", "")))
        return ToolOutput(result.as_text(), evidence=1)

    def _read(self, name: str, args: dict[str, Any]) -> ToolOutput:
        reader = self._get("reader")
        if reader is None:
            return ToolOutput("集群不可用：本地 TiDB 集群没有启动或无法连接。", cluster_down=True)
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
        if name not in calls:
            return ToolOutput(f"没有名为 {name} 的工具。", error=True)
        return ToolOutput(str(calls[name]().as_text()), evidence=1)
