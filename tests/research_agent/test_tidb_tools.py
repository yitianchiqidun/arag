from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from research_agent.tidb import cluster, sample_data, setup
from research_agent.tidb.db import DatabaseError, QueryResult, TiDBConfig
from research_agent.tidb.readonly import (
    ReadOnlyTools,
    UnsafeSQL,
    check_readonly_sql,
    stats_healthy,
)
from research_agent.tidb.write import WriteExecutor, WriteTools, sql_literal


class FakeDatabase:
    """Records statements; answers queries from ``responses`` (first matching substring)."""

    def __init__(
        self, responses: dict[str, QueryResult] | None = None, refuse: bool = False
    ) -> None:
        self.responses = responses or {}
        self.refuse = refuse
        self.queries: list[tuple[str, list[Any] | None, int | None]] = []
        self.executed: list[tuple[str, list[Any] | None]] = []

    def query(
        self, sql: str, params: Sequence[Any] | None = None, max_rows: int | None = None
    ) -> QueryResult:
        self.queries.append((sql, list(params) if params is not None else None, max_rows))
        for key, result in self.responses.items():
            if key in sql:
                return result
        return QueryResult(columns=["x"], rows=[])

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> int:
        if self.refuse:
            raise DatabaseError("Access denied", 1142)
        self.executed.append((sql, list(params) if params is not None else None))
        return 1


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders WHERE status = 'paid' LIMIT 10;",
        "select /*+ USE_INDEX(orders, idx_user) */ id from orders",
        "WITH t AS (SELECT 1) SELECT * FROM t",
        "SHOW CREATE TABLE orders",
        "SHOW GLOBAL VARIABLES LIKE 'tidb_mem%'",
        "EXPLAIN SELECT * FROM orders",
        "EXPLAIN ANALYZE SELECT count(*) FROM orders",
        "DESC orders",
        "SELECT 'drop table x; delete' AS text",  # keywords inside a literal are fine
        "SELECT update_time, offset_value FROM t",  # words containing keywords are fine
    ],
)
def test_readonly_sql_allowed(sql: str) -> None:
    assert check_readonly_sql(sql) == sql.strip().rstrip(";").strip()


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("DELETE FROM orders", "只允许"),
        ("SELECT 1; DROP TABLE orders", "一次只能执行一条语句"),
        ("WITH t AS (SELECT 1) DELETE FROM orders", "DELETE"),
        ("SELECT * FROM orders INTO OUTFILE '/tmp/x'", "OUTFILE"),
        ("SELECT * FROM orders WHERE id = 1 FOR UPDATE", "UPDATE"),
        ("SELECT * FROM orders LOCK IN SHARE MODE", "LOCK"),
        ("EXPLAIN ANALYZE DELETE FROM orders", "EXPLAIN ANALYZE"),
        ("SET GLOBAL tidb_mem_quota_query = 1", "只允许"),
        ("/* only a comment */", "空语句"),
    ],
)
def test_readonly_sql_refused(sql: str, message: str) -> None:
    with pytest.raises(UnsafeSQL, match=message):
        check_readonly_sql(sql)


def test_readonly_tools_queries() -> None:
    db = FakeDatabase()
    tools = ReadOnlyTools(db, max_rows=50)
    tools.run_sql("SELECT * FROM orders")
    tools.system_variables("tidb_mem_quota_query")
    tools.slow_queries(limit=500, min_seconds=1.0)
    tools.lock_waits()
    tools.statement_summary()
    assert db.queries[0] == ("SELECT * FROM orders", None, 50)
    assert db.queries[1] == ("SHOW GLOBAL VARIABLES LIKE %s", ["tidb_mem_quota_query"], 50)
    assert "cluster_slow_query" in db.queries[2][0] and db.queries[2][1] == [1.0, 50]
    assert "data_lock_waits" in db.queries[3][0] and "cluster_tidb_trx" in db.queries[3][0]
    assert "statements_summary" in db.queries[4][0]
    with pytest.raises(UnsafeSQL, match="SELECT"):
        tools.explain("SHOW TABLES")
    tools.explain("SELECT * FROM orders", analyze=True)
    assert db.queries[-1][0] == "EXPLAIN ANALYZE SELECT * FROM orders"
    with pytest.raises(UnsafeSQL, match="不合法"):
        tools.table_schema("orders; DROP TABLE x")


def test_table_stats_computes_the_health() -> None:
    meta = QueryResult(
        columns=["Db_name", "Table_name", "Modify_count", "Row_count", "Last_analyze_time"],
        rows=[["shop", "orders", 18000, 30000, "t"], ["shop", "users", 0, 10, None]],
    )
    db = FakeDatabase({"STATS_META": meta})
    stats = ReadOnlyTools(db).table_stats("shop.orders")
    assert stats.columns[-1] == "Healthy"
    assert [row[-1] for row in stats.rows] == [40, 0]  # never analyzed: 0
    assert db.queries[0][1] == ["orders"] and len(db.queries) == 1


@pytest.mark.parametrize(
    ("modified", "rows", "expected"), [(0, 0, 100), (0, 500, 100), (100, 400, 75), (500, 400, 0)]
)
def test_stats_healthy(modified: int, rows: int, expected: int) -> None:
    assert stats_healthy(modified, rows) == expected


def test_query_result_as_text() -> None:
    result = QueryResult(columns=["a", "b"], rows=[[1, None], ["x" * 5, 2]], truncated=True)
    assert result.as_text(max_cell=4) == "a\tb\n1\tNULL\nxxxx…\t2\n(只显示前 2 行)"


def test_wide_results_name_every_value() -> None:
    columns = [
        "Table",
        "Key_name",
        "Null",
        "Comment",
        "Index_comment",
        "Visible",
        "Expression",
        "Clustered",
        "Global",
    ]
    result = QueryResult(
        columns=columns, rows=[["orders", "idx_user", "", "", "", "YES", None, "NO", "NO"]]
    )
    assert result.as_text() == (
        "第 1 行：Table=orders，Key_name=idx_user，Null=''，Comment=''，Index_comment=''，"
        "Visible=YES，Expression=NULL，Clustered=NO，Global=NO"
    )
    assert QueryResult(columns=columns, rows=[]).as_text() == "（没有数据）"
    assert QueryResult(columns=["a", "b"], rows=[["", 1]]).as_text() == "a\tb\n''\t1"


VARIABLES = QueryResult(
    columns=["Variable_name", "Value"], rows=[["tidb_mem_quota_query", "1073741824"]]
)


def test_write_proposals_do_not_execute() -> None:
    db = FakeDatabase(
        {
            "VARIABLES": VARIABLES,
            "SHOW CREATE TABLE": QueryResult(
                columns=["Table", "Create Table"], rows=[["orders", "CREATE TABLE orders (...)"]]
            ),
        }
    )
    tools = WriteTools(ReadOnlyTools(db))
    pending = tools.propose_set_global("tidb_mem_quota_query", "2147483648")
    assert pending.statement == "SET GLOBAL tidb_mem_quota_query = 2147483648"
    assert pending.rollback == "SET GLOBAL tidb_mem_quota_query = 1073741824"
    assert pending.before == {"value": "1073741824"} and "1073741824" in pending.impact
    index = tools.propose_create_index("orders", ["status", "created_at"])
    assert index.statement == "CREATE INDEX idx_status_created_at ON orders (status, created_at)"
    assert tools.propose_analyze("orders").statement == "ANALYZE TABLE orders"
    assert db.executed == []  # proposing never writes
    with pytest.raises(UnsafeSQL):
        tools.propose_set_global("no_such_variable", "1")
    with pytest.raises(UnsafeSQL):
        tools.propose_create_index("orders", ["status) ; DROP TABLE x --"])


def test_sql_literal() -> None:
    assert sql_literal("2147483648") == "2147483648"
    assert sql_literal("on") == "ON"
    assert sql_literal("it's") == "'it\\'s'"


def test_executor_requires_confirmation(tmp_path: Path) -> None:
    reader = ReadOnlyTools(FakeDatabase({"VARIABLES": VARIABLES}))
    admin = FakeDatabase()
    executor = WriteExecutor(admin, reader, tmp_path / "audit.jsonl")
    pending = WriteTools(reader).propose_set_global("tidb_mem_quota_query", "1")

    rejected = executor.execute(pending, confirmed=False)
    assert (rejected.executed, admin.executed) == (False, [])
    done = executor.execute(pending, confirmed=True)
    assert done.executed and admin.executed == [("SET GLOBAL tidb_mem_quota_query = 1", None)]
    assert done.after == {"value": "1073741824"}  # what the (fake) reader says afterwards
    undo = executor.rollback(done)
    assert undo.executed and admin.executed[-1][0] == "SET GLOBAL tidb_mem_quota_query = 1073741824"
    with pytest.raises(ValueError, match="回滚"):
        executor.rollback(rejected)
    audit = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert [(a["confirmed"], a["executed"]) for a in audit] == [
        (False, False),
        (True, True),
        (True, True),
    ]

    refused = WriteExecutor(FakeDatabase(refuse=True), reader, tmp_path / "audit.jsonl")
    failed = refused.execute(pending, confirmed=True)
    assert not failed.executed and failed.error is not None and "Access denied" in failed.error


def test_sample_data_is_deterministic() -> None:
    size = sample_data.DataSize(users=50, products=20, orders=100)
    a, b = sample_data.generate(size, seed=1), sample_data.generate(size, seed=1)
    assert a == b and sample_data.generate(size, seed=2) != a
    assert [len(a[t]) for t in ("users", "products", "orders")] == [50, 20, 100]
    assert 100 <= len(a["order_items"]) <= 300
    assert {o[2] for o in a["orders"]} <= set(sample_data.STATUSES)
    statements = list(sample_data.insert_statements(a, batch=40))
    assert sum(len(p) for _, p in statements) == sum(len(r) * len(r[0]) for r in a.values())
    db = FakeDatabase()
    counts = sample_data.load(db, size, seed=1)
    assert counts["users"] == 50
    executed = [sql for sql, _ in db.executed]
    assert executed[0].startswith("CREATE TABLE IF NOT EXISTS users")
    assert executed[-1] == "ANALYZE TABLE order_items"
    assert "KEY idx_phone (phone)" in sample_data.SCHEMA[0]
    order_keys = [
        line for line in sample_data.SCHEMA[2].splitlines() if line.strip().startswith("KEY")
    ]
    assert order_keys == ["    KEY idx_user (user_id)"]  # no index on status or created_at


def test_cluster_command_and_memory(tmp_path: Path) -> None:
    cfg = cluster.ClusterConfig(tiup="/opt/tiup")
    cmd = cluster.playground_command(cfg)
    assert cmd[:3] == ["/opt/tiup", "playground", "v8.5.8"]
    assert cmd[cmd.index("--tiflash") + 1] == "0" and "--without-monitor" in cmd
    for pid, name, rss in [
        ("10", "/home/u/.tiup/components/tidb/tidb-server", 512000),
        ("11", "tikv-server", 1048576),
        ("12", "python", 99999),
    ]:
        proc = tmp_path / pid
        proc.mkdir()
        (proc / "cmdline").write_bytes(name.encode() + b"\0--flag\0")
        (proc / "status").write_text(f"Name:\tx\nVmRSS:\t{rss} kB\n")
    (tmp_path / "self").mkdir()
    assert cluster.component_memory(str(tmp_path)) == {
        "tidb-server": 500,
        "tikv-server": 1024,
        "pd-server": 0,
    }
    assert cluster.stop(cluster.ClusterConfig(state_dir=str(tmp_path / "state"))) is False


def test_readonly_grants_and_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = TiDBConfig()
    assert setup.readonly_grants(cfg) == [
        "GRANT SELECT ON shop.* TO 'agent_ro'@'%'",
        "GRANT PROCESS ON *.* TO 'agent_ro'@'%'",
        "GRANT SELECT ON mysql.stats_meta TO 'agent_ro'@'%'",
    ]
    monkeypatch.setattr(setup, "readonly_database", lambda c: FakeDatabase(refuse=True))
    result = setup.verify_readonly(cfg)
    assert len(result) == len(setup.WRITE_PROBES) and all(result.values())
    monkeypatch.setattr(setup, "readonly_database", lambda c: FakeDatabase())
    assert all(v is None for v in setup.verify_readonly(cfg).values())  # a writable account


class LoginRefused(FakeDatabase):
    def query(self, sql: str, params: Any = None, max_rows: int | None = None) -> QueryResult:
        raise DatabaseError("Access denied for user 'agent_ro'", 1045)

    def execute(self, sql: str, params: Any = None) -> int:
        raise DatabaseError("Access denied for user 'agent_ro'", 1045)


class OtherError(FakeDatabase):
    def execute(self, sql: str, params: Any = None) -> int:
        raise DatabaseError("Table doesn't exist", 1146)


def test_verification_needs_a_working_login_and_privilege_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = TiDBConfig()
    monkeypatch.setattr(setup, "readonly_database", lambda c: LoginRefused())
    with pytest.raises(RuntimeError, match="cannot log in"):
        setup.verify_readonly(cfg)
    monkeypatch.setattr(setup, "readonly_database", lambda c: OtherError())
    assert all(v is None for v in setup.verify_readonly(cfg).values())  # proves nothing


def test_readonly_user_statements_pass_the_host_wildcard_as_a_parameter() -> None:
    admin = FakeDatabase()
    setup.create_readonly_user(admin, TiDBConfig(), "pw")
    assert admin.executed[:2] == [
        ("CREATE USER IF NOT EXISTS %s@%s IDENTIFIED BY %s", ["agent_ro", "%", "pw"]),
        ("ALTER USER %s@%s IDENTIFIED BY %s", ["agent_ro", "%", "pw"]),
    ]
    # with parameters, the driver formats the text: no '%' besides the placeholders
    assert all("%" not in sql.replace("%s", "") for sql, params in admin.executed if params)
