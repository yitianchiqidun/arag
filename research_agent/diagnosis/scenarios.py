"""Fault scenarios for the diagnosis evaluation (stage 8, D34).

Each scenario injects one fault into the sample database of the local cluster with the admin
account, runs the workload that shows the symptom, and carries the user's description of the
symptom (without the cause) and the reference answer: root cause, fix and related documents.
The runner (``research_agent.diagnosis.run``) asks the assistant, approves only a proposal
that matches the fix, scores the answer and always cleans up.

The sample data is small (30,000 orders), so every query takes milliseconds: the missing
index scenario lowers ``tidb_slow_log_threshold`` to 10 ms for its run so that the slow query
log has entries, and restores the default afterwards.
"""

from __future__ import annotations

import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from research_agent.tidb.db import Database, DatabaseError

Connect = Callable[[], Database]
ORDERS_INDEXES = {"PRIMARY", "idx_user"}  # the sample schema's indexes on orders


def _columns(statement: str) -> list[str]:
    match = re.search(r"\(([^)]*)\)\s*$", statement)
    return [c.strip().strip("`") for c in match.group(1).split(",")] if match else []


def _int_value(statement: str) -> int | None:
    match = re.search(r"=\s*'?(-?\d+)'?\s*$", statement)
    return int(match.group(1)) if match else None


def _close(db: Database) -> None:
    close = getattr(db, "close", None)
    if close is not None:
        close()


class Scenario(ABC):
    name: str
    title: str
    symptom: str  # what the user says; never the cause
    root_cause: str
    fix: str
    expected_write: str | None  # the write that fixes it, or None when no write tool does
    related_docs: tuple[str, ...]  # substrings of document paths that count as related
    # Every group needs one of its keywords in the answer (case-insensitive).
    cause_keywords: tuple[tuple[str, ...], ...]

    def __init__(self, connect: Connect) -> None:
        self.connect = connect
        self.cleanup_errors: list[str] = []

    @abstractmethod
    def inject(self) -> None: ...

    def workload(self) -> None:
        """Statements an application would have run before the user noticed."""

    def accepts(self, pending: dict[str, Any]) -> bool:
        """Whether a simulated user approves this proposal (only the reference fix)."""
        return False

    def keywords(self) -> list[tuple[str, ...]]:
        return list(self.cause_keywords)

    @abstractmethod
    def cleanup(self) -> None: ...

    def _run_all(self, db: Database, statements: list[str]) -> None:
        for sql in statements:
            try:
                db.execute(sql)
            except DatabaseError as e:
                self.cleanup_errors.append(f"{sql}: {e}")


class MissingIndex(Scenario):
    name = "missing_index"
    title = "缺少索引导致慢查询"
    QUERIES = (
        (
            "SELECT id, user_id, amount, created_at FROM orders "
            "WHERE status = 'paid' AND created_at >= '2026-06-01'"
        ),
        "SELECT COUNT(*), SUM(amount) FROM orders WHERE status = 'refunded'",
    )
    symptom = (
        "订单管理后台按状态筛选订单的页面（比如查最近已付款的订单）越来越慢，其他页面都正常。"
        "帮我看看是什么原因。"
    )
    root_cause = "orders 表的 status 列没有索引，按状态筛选的查询每次都全表扫描（TableFullScan）。"
    fix = "在 orders(status) 上建索引（或以 status 开头的组合索引）。"
    expected_write = "CREATE INDEX … ON orders (status …)"
    related_docs = (
        "explain-",
        "identify-slow-queries",
        "analyze-slow-queries",
        "sql-tuning",
        "tune-",
    )
    cause_keywords = (("status",), ("索引", "index"), ("全表扫描", "TableFullScan", "全表"))

    def inject(self) -> None:
        admin = self.connect()
        self._drop_extra_indexes(admin)
        admin.execute("SET GLOBAL tidb_slow_log_threshold = 10")

    def workload(self) -> None:
        app = self.connect()  # a new session sees the lowered threshold
        for _ in range(20):
            for sql in self.QUERIES:
                app.query(sql)
        _close(app)

    def accepts(self, pending: dict[str, Any]) -> bool:
        columns = _columns(str(pending.get("statement", "")))
        return (
            pending.get("kind") == "create_index"
            and str(pending.get("target", "")).split(".")[-1] == "orders"
            and bool(columns)
            and columns[0] == "status"
        )

    def _drop_extra_indexes(self, admin: Database) -> None:
        rows = admin.query("SHOW INDEX FROM orders").rows
        for name in sorted({str(r[2]) for r in rows} - ORDERS_INDEXES):
            self._run_all(admin, [f"DROP INDEX `{name}` ON orders"])

    def cleanup(self) -> None:
        admin = self.connect()
        self._drop_extra_indexes(admin)
        self._run_all(admin, ["SET GLOBAL tidb_slow_log_threshold = 300"])


class MemoryQuota(Scenario):
    name = "memory_quota"
    title = "单条 SQL 内存上限过小"
    REPORT = (
        "SELECT o.user_id, COUNT(*) AS orders, SUM(i.quantity * i.price) AS spent "
        "FROM orders o JOIN order_items i ON i.order_id = o.id "
        "GROUP BY o.user_id ORDER BY spent DESC LIMIT 10"
    )
    QUOTA = 1 << 20  # 1 MiB; the default is 1 GiB
    symptom = (
        "每天跑的用户消费排行报表从今天开始报错：ERROR 8175 (HY000): Your query has been cancelled "
        "due to exceeding the allowed memory limit for a single SQL query. 昨天还是好的，数据量也"
        f"没怎么变。SQL 是：{REPORT}"
    )
    root_cause = (
        "tidb_mem_quota_query 的全局值被改成了 1 MiB（1048576，默认 1 GiB），报表 SQL 的内存用量"
        "超过了单条 SQL 的内存上限，被取消。"
    )
    fix = "把 tidb_mem_quota_query 恢复为默认的 1073741824（或其他足够大的值），并查清是谁改的。"
    expected_write = "SET GLOBAL tidb_mem_quota_query = ≥ 64 MiB"
    related_docs = ("system-variables", "configure-memory-usage", "error-codes", "troubleshoot")
    cause_keywords = (("tidb_mem_quota_query",), ("1048576", "1 MiB", "1MiB", "1 MB", "1MB"))

    def inject(self) -> None:
        self.connect().execute(f"SET GLOBAL tidb_mem_quota_query = {self.QUOTA}")

    def workload(self) -> None:
        app = self.connect()
        for _ in range(3):
            try:
                app.query(self.REPORT)
            except DatabaseError:
                pass  # error 8175: the symptom
        _close(app)

    def accepts(self, pending: dict[str, Any]) -> bool:
        value = _int_value(str(pending.get("statement", "")))
        return (
            pending.get("kind") == "set_global"
            and pending.get("target") == "tidb_mem_quota_query"
            and value is not None
            and value >= 64 << 20
        )

    def cleanup(self) -> None:
        self._run_all(self.connect(), ["SET GLOBAL tidb_mem_quota_query = 1073741824"])


class LockWait(Scenario):
    name = "lock_wait"
    title = "悲观锁等待超时"
    ROW = 100
    symptom = (
        "更新订单状态的接口最近一直失败：请求卡住十几秒，然后报 Lock wait timeout exceeded; "
        "try restarting transaction。别的接口都正常，帮我查一下原因。"
    )
    root_cause = (
        "一个会话（root）开启的悲观事务更新了 orders 表 id = 100 的行后一直没有提交，持有行锁；"
        "更新订单状态的事务排队等这把锁，超过 innodb_lock_wait_timeout 后报错。"
    )
    fix = (
        "找到持有锁的会话，让应用提交或回滚这个事务，必要时 KILL 该会话；调大 "
        "innodb_lock_wait_timeout 不能解决问题。"
    )
    expected_write = None
    related_docs = ("lock-conflicts", "pessimistic-transaction", "data-lock-waits", "tidb-trx")
    cause_keywords = (
        ("未提交", "没有提交", "没提交", "未结束", "长事务", "空闲", "idle"),
        ("持有", "阻塞", "holding", "占用"),
    )
    WAIT_S = 15

    def __init__(self, connect: Connect) -> None:
        super().__init__(connect)
        self.holder: Database | None = None
        self.holder_session: int | None = None
        self.stop = threading.Event()
        self.waiter: threading.Thread | None = None

    def inject(self) -> None:
        self.holder = self.connect()
        self.holder_session = int(self.holder.query("SELECT CONNECTION_ID()").rows[0][0])
        self.holder.execute("BEGIN PESSIMISTIC")
        self.holder.execute(f"UPDATE orders SET amount = amount + 0 WHERE id = {self.ROW}")
        self.waiter = threading.Thread(target=self._wait_for_lock, daemon=True)
        self.waiter.start()

    def _wait_for_lock(self) -> None:
        """The application retrying the update until the scenario ends."""
        while not self.stop.is_set():
            app = self.connect()
            try:
                app.execute(f"SET SESSION innodb_lock_wait_timeout = {self.WAIT_S}")
                app.execute("BEGIN PESSIMISTIC")
                app.execute(f"UPDATE orders SET status = 'shipped' WHERE id = {self.ROW}")
            except DatabaseError:
                pass  # error 1205: the symptom
            finally:
                try:
                    app.execute("ROLLBACK")
                except DatabaseError:
                    pass
                _close(app)

    def workload(self) -> None:
        time.sleep(2)  # the first update is waiting by now

    def keywords(self) -> list[tuple[str, ...]]:
        """The answer must also name the session that holds the lock."""
        return [*self.cause_keywords, (str(self.holder_session),)]

    def cleanup(self) -> None:
        self.stop.set()
        if self.holder is not None:
            self._run_all(self.holder, ["ROLLBACK"])  # releases the lock; the waiter ends
            _close(self.holder)
        if self.waiter is not None:
            self.waiter.join(timeout=self.WAIT_S + 10)


class StaleStats(Scenario):
    name = "stale_stats"
    title = "统计信息过期导致执行计划变差"
    USER = 7
    QUERY = (
        f"SELECT status, COUNT(*), SUM(amount) FROM orders WHERE user_id = {USER} GROUP BY status"
    )
    OFFSET = 1_000_000  # ids of the imported rows
    symptom = (
        f"昨天把用户 {USER} 的一批历史订单迁移进了 orders 表，之后查用户 {USER} 订单统计的 SQL 变慢了"
        f"（以前很快）：{QUERY}。帮我看看为什么。"
    )
    root_cause = (
        "导入 6 万行后统计信息没有更新：自动 ANALYZE 被关闭（tidb_enable_auto_analyze = OFF），"
        "orders 的修改行数 60000 / 总行数 90000，健康度约 33。优化器估算 user_id = 7 只有约 9 行，"
        "实际 60005 行，选择了 IndexLookUp。"
    )
    fix = "ANALYZE TABLE orders，并重新打开 tidb_enable_auto_analyze。"
    expected_write = "ANALYZE TABLE orders"
    related_docs = ("statistics", "explain-", "sql-tuning", "system-variables")
    cause_keywords = (
        ("统计信息", "statistics"),
        ("过期", "陈旧", "不准", "未更新", "没有更新", "过时", "健康度", "stale", "outdated"),
    )
    MAX_WAIT_S = 150.0

    def inject(self) -> None:
        admin = self.connect()
        admin.execute("SET GLOBAL tidb_enable_auto_analyze = OFF")
        for shift in (self.OFFSET, self.OFFSET + 30_000):  # two copies of the 30,000 orders
            admin.execute(
                "INSERT INTO orders (id, user_id, status, amount, created_at) "
                f"SELECT id + {shift}, {self.USER}, status, amount, created_at FROM orders "
                "WHERE id <= 30000"
            )
        deadline = time.monotonic() + self.MAX_WAIT_S
        while time.monotonic() < deadline:  # TiDB writes the modify count back periodically
            rows = admin.query("SHOW STATS_META WHERE Table_name = 'orders'").rows
            if rows and int(rows[0][4]) >= 60_000:
                return
            time.sleep(3)
        raise TimeoutError("the modify count of orders did not reach 60000")

    def workload(self) -> None:
        app = self.connect()
        for _ in range(20):
            app.query(self.QUERY)
        _close(app)

    def accepts(self, pending: dict[str, Any]) -> bool:
        if pending.get("kind") == "analyze_table":
            return str(pending.get("target", "")).split(".")[-1] == "orders"
        statement = str(pending.get("statement", "")).upper()
        return (
            pending.get("kind") == "set_global"
            and pending.get("target") == "tidb_enable_auto_analyze"
            and ("'ON'" in statement or "= 1" in statement or "= ON" in statement)
        )

    def cleanup(self) -> None:
        self._run_all(
            self.connect(),
            [
                f"DELETE FROM orders WHERE id > {self.OFFSET}",
                "ANALYZE TABLE orders",
                "SET GLOBAL tidb_enable_auto_analyze = ON",
            ],
        )


SCENARIOS: tuple[type[Scenario], ...] = (MissingIndex, MemoryQuota, LockWait, StaleStats)
