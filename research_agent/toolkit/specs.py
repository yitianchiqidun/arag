"""The tool registry (stages 10–11): every tool the agent can see, with its permission level,
category, the intents it is exposed to, its timeout and how much of its result is kept.

Permission levels (D38):

- ``read``: runs directly (cluster reads, logs, configuration, documents, the local report);
- ``confirm``: only produces a proposal (statement + impact); it runs after the user confirms,
  inside the agent (``confirm`` node), never through a tool call;
- ``forbidden``: registered so that a call is recognised and refused, never exposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Permission = Literal["read", "confirm", "forbidden"]
Category = Literal["docs", "cluster", "logs", "config", "report", "write"]
DIAGNOSIS = ("diagnosis",)
DOC_INTENTS = ("param_lookup", "research", "diagnosis")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    properties: dict[str, Any]
    required: tuple[str, ...] = ()
    permission: Permission = "read"
    category: Category = "cluster"
    intents: tuple[str, ...] = DIAGNOSIS
    timeout_s: float = 30.0
    max_chars: int = 4000
    defaults: dict[str, Any] = field(default_factory=dict)

    def schema(self) -> dict[str, Any]:
        """OpenAI-style function definition sent to the model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": list(self.required),
                },
            },
        }

    def input_schema(self) -> dict[str, Any]:
        """JSON schema of the arguments (MCP ``inputSchema``)."""
        return self.schema()["function"]["parameters"]  # type: ignore[no-any-return]


_STR = {"type": "string"}
_INT = {"type": "integer", "minimum": 1, "maximum": 200}
_LIST = {"type": "array", "items": _STR}
_COMPONENT = {"type": "string", "enum": ["tidb", "tikv", "pd"]}

SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "search_docs",
        "查 TiDB 官方文档（多轮检索，较慢），返回带引用的回答。",
        {"question": _STR},
        ("question",),
        category="docs",
        intents=DOC_INTENTS,
        timeout_s=240.0,
        max_chars=6000,
    ),
    ToolSpec(
        "run_sql",
        "执行一条只读 SQL（SELECT / SHOW / EXPLAIN / DESC），最多返回 200 行。",
        {"sql": _STR},
        ("sql",),
    ),
    ToolSpec(
        "show_variables",
        "查看系统变量的全局值，pattern 是 SQL LIKE 模式。",
        {"pattern": _STR},
        ("pattern",),
    ),
    ToolSpec(
        "slow_queries",
        "最近的慢查询（耗时、扫描行数、索引、SQL）。",
        {"limit": _INT, "min_seconds": {"type": "number", "minimum": 0}},
        defaults={"limit": 10, "min_seconds": 0},
    ),
    ToolSpec(
        "explain",
        "查看 SELECT 的执行计划；analyze=true 会实际执行并给出实际行数。",
        {"sql": _STR, "analyze": {"type": "boolean"}},
        ("sql",),
        defaults={"analyze": False},
    ),
    ToolSpec("lock_waits", "当前的悲观锁等待：等待方、持有方事务及其 SQL。", {}),
    ToolSpec("transactions", "正在运行的事务，按开始时间排序。", {}),
    ToolSpec("deadlocks", "最近的死锁记录。", {"limit": _INT}, defaults={"limit": 10}),
    ToolSpec(
        "statement_summary",
        "总耗时最多的 SQL 语句（statements_summary）。",
        {"limit": _INT},
        defaults={"limit": 10},
    ),
    ToolSpec(
        "table_stats",
        "表的统计信息健康度和修改行数，判断统计信息是否过期。",
        {"table": _STR},
        ("table",),
    ),
    ToolSpec("table_schema", "表结构和索引（SHOW CREATE TABLE）。", {"table": _STR}, ("table",)),
    ToolSpec("bindings", "全局 SQL 绑定（可能强制执行计划）。", {}),
    ToolSpec("cluster_info", "集群各组件的版本和运行时间。", {}),
    ToolSpec(
        "search_logs",
        "检索集群组件（tidb / tikv / pd）的日志，返回包含关键词的最新若干行；用于报错、警告、"
        "登录失败、账号锁定等安全操作（日志中的 CRUCIAL OPERATION）。",
        {"component": _COMPONENT, "keyword": _STR, "max_lines": _INT},
        ("component",),
        category="logs",
        defaults={"keyword": "", "max_lines": 50},
    ),
    ToolSpec(
        "show_config",
        "查看组件（tidb / tikv / pd）的配置项（SHOW CONFIG），name 是配置项名中的一段。",
        {"component": _COMPONENT, "name": _STR},
        ("component",),
        category="config",
        defaults={"name": ""},
    ),
    ToolSpec(
        "export_report",
        "把本次诊断整理成 Markdown 报告保存到本地（症状、证据、根因、处理、引用）。",
        {
            "title": _STR,
            "symptom": _STR,
            "evidence": _LIST,
            "root_cause": _STR,
            "actions": _LIST,
            "citations": _LIST,
        },
        ("title", "symptom", "evidence", "root_cause", "actions"),
        category="report",
        defaults={"citations": []},
        max_chars=500,
    ),
    ToolSpec(
        "propose_set_global",
        "提出修改系统变量的全局值（需要用户确认后才执行）。",
        {"name": _STR, "value": _STR, "reason": _STR},
        ("name", "value", "reason"),
        permission="confirm",
        category="write",
    ),
    ToolSpec(
        "propose_analyze",
        "提出对表执行 ANALYZE TABLE 重新收集统计信息（需要用户确认）。",
        {"table": _STR, "reason": _STR},
        ("table", "reason"),
        permission="confirm",
        category="write",
    ),
    ToolSpec(
        "propose_create_index",
        "提出在表上创建索引（需要用户确认）。",
        {"table": _STR, "columns": _LIST, "reason": _STR},
        ("table", "columns", "reason"),
        permission="confirm",
        category="write",
    ),
    ToolSpec(
        "execute_write",
        "直接执行写操作。不开放给模型：写操作只能经用户确认后由 Agent 执行。",
        {"statement": _STR},
        ("statement",),
        permission="forbidden",
        category="write",
        intents=(),
    ),
)
SPEC_BY_NAME = {s.name: s for s in SPECS}
# Retrieval tools of the A-RAG agent behind ``search_docs`` and the document channel; listed
# for the tool-definition token count (stage 11), not dispatched here.
ARAG_RETRIEVAL_TOOLS = ("keyword_search", "semantic_search", "read_chunk")


def exposed(intent: str) -> list[ToolSpec]:
    return [s for s in SPECS if s.permission != "forbidden" and intent in s.intents]
