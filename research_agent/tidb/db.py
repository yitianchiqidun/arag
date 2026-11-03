"""Connections to TiDB: a small protocol the tools depend on, and its PyMySQL implementation.

The read-only tools connect with an account that only has query privileges (``setup``), so a
write is refused by the database itself; the statement check in ``readonly`` is a second
line. Writes after a confirmation go through a separate admin connection.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class TiDBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = Field(default_factory=lambda: os.environ.get("TIDB_HOST", "127.0.0.1"))
    port: int = 4000
    database: str = "shop"
    readonly_user: str = "agent_ro"
    readonly_password_env: str = "TIDB_RO_PASSWORD"  # passwords only in .env
    admin_user: str = "root"  # used for confirmed writes and for setup
    admin_password_env: str = "TIDB_ADMIN_PASSWORD"  # empty for a fresh playground
    max_rows: int = Field(default=200, ge=1)  # rows a read-only query returns at most
    timeout_ms: int = Field(default=10_000, ge=100)  # max_execution_time of read-only queries


class DatabaseError(RuntimeError):
    """The database refused or failed a statement (privileges, syntax, timeout, ...)."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool = False  # more rows than the limit
    elapsed_ms: float = 0.0

    def as_text(self, max_cell: int = 200) -> str:
        """A compact table for the model: header, then one tab-separated line per row."""
        lines = ["\t".join(self.columns)]
        for row in self.rows:
            cells = ["NULL" if v is None else str(v) for v in row]
            lines.append("\t".join(c if len(c) <= max_cell else c[:max_cell] + "…" for c in cells))
        if self.truncated:
            lines.append(f"(只显示前 {len(self.rows)} 行)")
        return "\n".join(lines)


class Database(Protocol):
    def query(
        self, sql: str, params: Sequence[Any] | None = None, max_rows: int | None = None
    ) -> QueryResult: ...

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> int:
        """Run a statement that returns no rows; the affected row count. Failures raise
        :class:`DatabaseError`."""
        ...


class PyMySQLDatabase:
    """One lazily opened connection. ``session_sql`` runs right after connecting."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str | None = None,
        session_sql: Sequence[str] = (),
        autocommit: bool = True,
    ) -> None:
        self._kwargs: dict[str, Any] = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "autocommit": autocommit,
            "charset": "utf8mb4",
            "connect_timeout": 10,
        }
        self._session_sql = list(session_sql)
        self._conn: Any = None

    def _connection(self) -> Any:
        if self._conn is None:
            import pymysql

            self._conn = pymysql.connect(**self._kwargs)
            with self._conn.cursor() as cur:
                for sql in self._session_sql:
                    cur.execute(sql)
        return self._conn

    def query(
        self, sql: str, params: Sequence[Any] | None = None, max_rows: int | None = None
    ) -> QueryResult:
        import pymysql

        start = time.perf_counter()
        try:
            return self._query(sql, params, max_rows, start)
        except pymysql.err.MySQLError as e:
            raise DatabaseError(str(e), e.args[0] if e.args else None) from e

    def _query(
        self, sql: str, params: Sequence[Any] | None, max_rows: int | None, start: float
    ) -> QueryResult:
        with self._connection().cursor() as cur:
            cur.execute(sql, params)
            columns = [d[0] for d in cur.description or []]
            if max_rows is None:
                rows = list(cur.fetchall())
                truncated = False
            else:
                rows = list(cur.fetchmany(max_rows + 1))
                truncated = len(rows) > max_rows
                rows = rows[:max_rows]
        elapsed = (time.perf_counter() - start) * 1000
        return QueryResult(
            columns=columns,
            rows=[list(r) for r in rows],
            truncated=truncated,
            elapsed_ms=round(elapsed, 1),
        )

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> int:
        import pymysql

        try:
            with self._connection().cursor() as cur:
                return int(cur.execute(sql, params))
        except pymysql.err.MySQLError as e:
            raise DatabaseError(str(e), e.args[0] if e.args else None) from e

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def readonly_database(cfg: TiDBConfig) -> PyMySQLDatabase:
    """The agent's read-only connection, with a per-query execution time limit."""
    return PyMySQLDatabase(
        cfg.host,
        cfg.port,
        cfg.readonly_user,
        os.environ.get(cfg.readonly_password_env, ""),
        cfg.database,
        session_sql=[f"SET SESSION max_execution_time = {int(cfg.timeout_ms)}"],
    )


def admin_database(cfg: TiDBConfig, database: str | None = None) -> PyMySQLDatabase:
    """The connection for setup and for writes a user has confirmed."""
    return PyMySQLDatabase(
        cfg.host,
        cfg.port,
        cfg.admin_user,
        os.environ.get(cfg.admin_password_env, ""),
        database if database is not None else cfg.database,
    )
