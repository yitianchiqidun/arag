"""Write operations as proposals that run only after a user confirms them.

The agent may propose three kinds of writes: ``SET GLOBAL`` of a system variable,
``ANALYZE TABLE`` and ``CREATE INDEX``. A proposal (:class:`PendingWrite`) holds the exact
statement, a description of its impact, the state before, and for a system variable the
statement that restores the old value. Nothing is executed when proposing; the read-only
connection is enough to build it.

:class:`WriteExecutor` runs a proposal on the admin connection only when called with
``confirmed=True`` (stage 6 connects this to LangGraph's ``interrupt()``), records the state
after it, and appends every confirmed or rejected proposal to an audit log. A system
variable change can be rolled back in one step.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from research_agent.tidb.db import Database, DatabaseError
from research_agent.tidb.readonly import ReadOnlyTools, UnsafeSQL, check_identifier

WriteKind = Literal["set_global", "analyze_table", "create_index"]
_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_KEYWORD = re.compile(r"^(ON|OFF|TRUE|FALSE|DEFAULT)$", re.IGNORECASE)


def sql_literal(value: str) -> str:
    """A value for ``SET GLOBAL``: a number or ON / OFF / TRUE / FALSE / DEFAULT as they are,
    anything else as a quoted string."""
    text = str(value).strip()
    if _NUMBER.match(text) or _KEYWORD.match(text):
        return text.upper() if _KEYWORD.match(text) else text
    escaped = text.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


class PendingWrite(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    kind: WriteKind
    statement: str
    impact: str  # shown to the user before they confirm
    target: str  # variable or table
    before: dict[str, Any] = Field(default_factory=dict)
    rollback: str | None = None  # SET GLOBAL only
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class WriteResult(BaseModel):
    pending_id: str
    kind: WriteKind
    statement: str
    confirmed: bool
    executed: bool
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    rollback: str | None = None
    at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def _rows(result: Any) -> list[dict[str, Any]]:
    return [dict(zip(result.columns, row)) for row in result.rows]


class WriteTools:
    """Builds proposals; reads the current state through the read-only tools."""

    def __init__(self, reader: ReadOnlyTools) -> None:
        self.reader = reader

    def _variable(self, name: str) -> str | None:
        rows = self.reader.system_variables(name).rows
        exact = [r for r in rows if str(r[0]).lower() == name.lower()]
        return None if not exact else str(exact[0][1])

    def propose_set_global(self, name: str, value: str) -> PendingWrite:
        check_identifier(name)
        current = self._variable(name)
        if current is None:
            raise UnsafeSQL(f"没有名为 {name} 的系统变量")
        return PendingWrite(
            kind="set_global",
            target=name,
            statement=f"SET GLOBAL {name} = {sql_literal(value)}",
            impact=(
                f"把系统变量 {name} 的全局值从 {current} 改为 {value}。GLOBAL 作用域的修改对新建立的"
                "会话生效（持久化到集群的变量重启后仍然有效），已有会话保持原来的会话值。"
            ),
            before={"value": current},
            rollback=f"SET GLOBAL {name} = {sql_literal(current)}",
        )

    def propose_analyze(self, table: str) -> PendingWrite:
        stats = self.reader.table_stats(table)
        return PendingWrite(
            kind="analyze_table",
            target=table,
            statement=f"ANALYZE TABLE {check_identifier(table)}",
            impact=(
                f"重新收集表 {table} 的统计信息。会扫描整张表并占用一定的 CPU 和 IO，"
                "完成后优化器按新的统计信息选择执行计划。不修改表数据。"
            ),
            before={"stats": _rows(stats)},
        )

    def propose_create_index(
        self, table: str, columns: list[str], name: str | None = None
    ) -> PendingWrite:
        check_identifier(table)
        if not columns:
            raise UnsafeSQL("至少需要一列")
        cols = [check_identifier(c) for c in columns]
        index = check_identifier(name or "idx_" + "_".join(cols))
        schema = self.reader.table_schema(table)
        return PendingWrite(
            kind="create_index",
            target=table,
            statement=f"CREATE INDEX {index} ON {table} ({', '.join(cols)})",
            impact=(
                f"在表 {table} 的 ({', '.join(cols)}) 上在线创建索引 {index}。建索引期间会回填现有数据，"
                "占用 CPU 和 IO；之后每次写入这张表都要多维护一个索引。不修改表数据。"
            ),
            before={"create_table": schema.rows[0][1] if schema.rows else None},
        )


class WriteExecutor:
    """Runs confirmed proposals on the admin connection; everything goes to the audit log."""

    def __init__(self, admin: Database, reader: ReadOnlyTools, audit_file: str | Path) -> None:
        self.admin = admin
        self.reader = reader
        self.audit_file = Path(audit_file)

    def _state(self, pending: PendingWrite) -> dict[str, Any]:
        if pending.kind == "set_global":
            rows = self.reader.system_variables(pending.target).rows
            return {"value": str(rows[0][1]) if rows else None}
        if pending.kind == "analyze_table":
            return {"stats": _rows(self.reader.table_stats(pending.target))}
        schema = self.reader.table_schema(pending.target)
        return {"create_table": schema.rows[0][1] if schema.rows else None}

    def _log(self, result: WriteResult) -> WriteResult:
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.model_dump(), ensure_ascii=False, default=str) + "\n")
        return result

    def execute(self, pending: PendingWrite, confirmed: bool) -> WriteResult:
        """Run ``pending`` if ``confirmed``; a rejected proposal is logged and never runs."""
        base = WriteResult(
            pending_id=pending.id,
            kind=pending.kind,
            statement=pending.statement,
            confirmed=confirmed,
            executed=False,
            before=pending.before,
            rollback=pending.rollback,
        )
        if not confirmed:
            return self._log(base)
        try:
            self.admin.execute(pending.statement)
        except DatabaseError as e:  # refused (privileges, syntax...): reported, not raised
            return self._log(base.model_copy(update={"error": f"{type(e).__name__}: {e}"}))
        return self._log(base.model_copy(update={"executed": True, "after": self._state(pending)}))

    def rollback(self, result: WriteResult) -> WriteResult:
        """Restore a system variable changed by ``result`` (the user's one-click rollback)."""
        if result.kind != "set_global" or not result.executed or result.rollback is None:
            raise ValueError("只有已执行的系统变量修改可以回滚")
        name = result.statement.split()[2]
        undo = PendingWrite(
            kind="set_global",
            target=name,
            statement=result.rollback,
            impact=f"回滚：{result.statement}",
            before=result.after,
            rollback=None,
        )
        return self.execute(undo, confirmed=True)
