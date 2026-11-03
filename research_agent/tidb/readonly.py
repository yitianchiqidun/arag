"""Read-only access for the agent: checked SQL and the usual diagnosis queries.

The database account is the real barrier (it only has query privileges, see ``setup``); the
statement check here is a second line that also gives the model a clear refusal:

- one statement only, starting with SELECT / SHOW / EXPLAIN / DESC / DESCRIBE / WITH;
- no write or administrative keyword anywhere outside string literals and comments
  (``WITH ... DELETE``, ``SELECT ... INTO OUTFILE``, ``SELECT ... FOR UPDATE`` are refused);
- ``EXPLAIN ANALYZE`` runs the statement, so it is allowed for a SELECT only;
- SHOW / DESC statements cannot write and are not scanned (``SHOW CREATE TABLE``).

Every query returns at most ``max_rows`` rows, and the connection sets
``max_execution_time`` (``db.readonly_database``). The diagnosis helpers use the system
tables documented in the corpus (``information_schema.cluster_slow_query``,
``data_lock_waits``, ``cluster_tidb_trx``, ``cluster_deadlocks``, ``statements_summary``).
"""

from __future__ import annotations

import re

from research_agent.tidb.db import Database, QueryResult

ALLOWED_FIRST = ("SELECT", "SHOW", "EXPLAIN", "DESC", "DESCRIBE", "WITH")
FORBIDDEN = frozenset(
    [
        "INSERT",
        "UPDATE",
        "DELETE",
        "REPLACE",
        "CREATE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "RENAME",
        "GRANT",
        "REVOKE",
        "SET",
        "LOAD",
        "CALL",
        "LOCK",
        "UNLOCK",
        "ANALYZE",
        "ADMIN",
        "SPLIT",
        "FLASHBACK",
        "RECOVER",
        "IMPORT",
        "KILL",
        "OUTFILE",
        "DUMPFILE",
        "BATCH",
        "HANDLER",
        "DO",
        "PREPARE",
        "EXECUTE",
        "DEALLOCATE",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT",
    ]
)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*)?$")
_LITERALS = re.compile(r"'(?:[^'\\]|\\.|'')*'|\"(?:[^\"\\]|\\.|\"\")*\"|`[^`]*`")
_COMMENTS = re.compile(r"/\*.*?\*/|--[^\n]*|#[^\n]*", re.DOTALL)
_WORDS = re.compile(r"[A-Za-z_]+")


class UnsafeSQL(ValueError):
    """The statement is not a single read-only query."""


def _skeleton(sql: str) -> str:
    """The statement with string literals, quoted names and comments blanked out."""
    return _COMMENTS.sub(" ", _LITERALS.sub("''", sql))


def check_readonly_sql(sql: str) -> str:
    """The statement without a trailing semicolon, or :class:`UnsafeSQL`."""
    text = sql.strip().rstrip(";").strip()
    skeleton = _skeleton(text)
    if not skeleton.strip():
        raise UnsafeSQL("空语句")
    if ";" in skeleton:
        raise UnsafeSQL("一次只能执行一条语句")
    words = [w.upper() for w in _WORDS.findall(skeleton)]
    first, rest = words[0], words[1:]
    if first not in ALLOWED_FIRST:
        raise UnsafeSQL(f"只允许 {' / '.join(ALLOWED_FIRST)} 语句，收到的是 {first}")
    if first in ("SHOW", "DESC", "DESCRIBE"):  # read-only by definition (SHOW CREATE TABLE)
        return text
    if first == "EXPLAIN" and rest and rest[0] == "ANALYZE":
        rest = rest[1:]
        target = [w for w in rest if w != "FORMAT"]
        if not target or target[0] not in ("SELECT", "WITH"):
            raise UnsafeSQL("EXPLAIN ANALYZE 会真正执行语句，只允许用于 SELECT")
    bad = sorted(set(rest) & FORBIDDEN)
    if bad:
        raise UnsafeSQL(f"语句中包含不允许的关键字：{', '.join(bad)}")
    if "FOR" in rest and "UPDATE" in rest:
        raise UnsafeSQL("不允许加锁读")
    return text


def check_identifier(name: str) -> str:
    """A table or variable name (optionally ``db.table``), or :class:`UnsafeSQL`."""
    if not _IDENT.match(name):
        raise UnsafeSQL(f"不合法的名称：{name!r}")
    return name


def stats_healthy(modify_count: int, row_count: int, analyzed: bool = True) -> int:
    """TiDB's statistics health (``SHOW STATS_HEALTHY``): 100 × (1 − modified / rows); 0 once
    as many rows were modified as the table has, or when it was never analyzed. TiDB divides
    by the row count of the last ANALYZE, ``STATS_META`` only has the current one, so this
    is close but not always equal."""
    if not analyzed:
        return 0
    if modify_count == 0:
        return 100
    if modify_count < row_count:
        return int((1.0 - modify_count / row_count) * 100.0)
    return 0


class ReadOnlyTools:
    def __init__(self, db: Database, max_rows: int = 200) -> None:
        self.db = db
        self.max_rows = max_rows

    def run_sql(self, sql: str) -> QueryResult:
        """Any single read-only statement (checked), at most ``max_rows`` rows."""
        return self.db.query(check_readonly_sql(sql), max_rows=self.max_rows)

    def system_variables(self, like: str) -> QueryResult:
        """Global values of the system variables matching ``like`` (SQL LIKE pattern)."""
        return self.db.query("SHOW GLOBAL VARIABLES LIKE %s", [like], max_rows=self.max_rows)

    def slow_queries(self, limit: int = 10, min_seconds: float = 0.0) -> QueryResult:
        """The most recent slow queries of every TiDB instance (internal SQL left out)."""
        sql = (
            "SELECT Time, Query_time, Mem_max, Process_keys, Total_keys, Index_names, DB, "
            "Digest, Plan_digest, Succ, Query FROM information_schema.cluster_slow_query "
            "WHERE Is_internal = 0 AND Query_time >= %s ORDER BY Time DESC LIMIT %s"
        )
        return self.db.query(sql, [min_seconds, min(limit, self.max_rows)], self.max_rows)

    def explain(self, sql: str, analyze: bool = False) -> QueryResult:
        """The execution plan of a SELECT; ``analyze`` runs it and adds actual row counts."""
        statement = check_readonly_sql(sql)
        if _WORDS.findall(_skeleton(statement))[0].upper() not in ("SELECT", "WITH"):
            raise UnsafeSQL("只能查看 SELECT 语句的执行计划")
        prefix = "EXPLAIN ANALYZE " if analyze else "EXPLAIN "
        return self.db.query(prefix + statement, max_rows=self.max_rows)

    def lock_waits(self) -> QueryResult:
        """Pessimistic lock waits now: the waiting and the holding transaction and their SQL."""
        sql = (
            "SELECT w.KEY, w.TRX_ID AS waiting_trx, w.SQL_DIGEST_TEXT AS waiting_sql, "
            "w.CURRENT_HOLDING_TRX_ID AS holding_trx, h.SESSION_ID AS holding_session, "
            "h.START_TIME AS holding_since, h.STATE AS holding_state, h.USER AS holding_user, "
            "h.CURRENT_SQL_DIGEST_TEXT AS holding_current_sql "
            "FROM information_schema.data_lock_waits w "
            "LEFT JOIN information_schema.cluster_tidb_trx h ON w.CURRENT_HOLDING_TRX_ID = h.ID"
        )
        return self.db.query(sql, max_rows=self.max_rows)

    def transactions(self) -> QueryResult:
        """Running transactions on every TiDB instance, the oldest first."""
        sql = (
            "SELECT INSTANCE, ID, START_TIME, STATE, WAITING_START_TIME, SESSION_ID, USER, DB, "
            "MEM_BUFFER_KEYS, CURRENT_SQL_DIGEST_TEXT FROM information_schema.cluster_tidb_trx "
            "ORDER BY START_TIME"
        )
        return self.db.query(sql, max_rows=self.max_rows)

    def deadlocks(self, limit: int = 10) -> QueryResult:
        sql = (
            "SELECT DEADLOCK_ID, OCCUR_TIME, RETRYABLE, TRY_LOCK_TRX_ID, TRX_HOLDING_LOCK, "
            "CURRENT_SQL_DIGEST_TEXT, KEY_INFO FROM information_schema.cluster_deadlocks "
            "ORDER BY OCCUR_TIME DESC LIMIT %s"
        )
        return self.db.query(sql, [min(limit, self.max_rows)], self.max_rows)

    def cluster_info(self) -> QueryResult:
        """Every component of the cluster with its version and uptime."""
        sql = "SELECT TYPE, INSTANCE, VERSION, START_TIME, UPTIME FROM information_schema.cluster_info"
        return self.db.query(sql, max_rows=self.max_rows)

    def statement_summary(self, limit: int = 10) -> QueryResult:
        """The statements that took the most total time in the current summary window."""
        sql = (
            "SELECT SCHEMA_NAME, STMT_TYPE, EXEC_COUNT, SUM_LATENCY, AVG_LATENCY, MAX_LATENCY, "
            "AVG_MEM, PLAN_DIGEST, DIGEST_TEXT, QUERY_SAMPLE_TEXT "
            "FROM information_schema.statements_summary ORDER BY SUM_LATENCY DESC LIMIT %s"
        )
        return self.db.query(sql, [min(limit, self.max_rows)], self.max_rows)

    def table_stats(self, table: str) -> QueryResult:
        """Statistics of a table (``SHOW STATS_META``): rows, rows modified since the last
        ANALYZE, and the health derived from them, to tell whether they are outdated.

        ``SHOW STATS_HEALTHY`` would need privileges on the whole ``mysql`` schema, which
        holds the password hashes; the read-only account only gets ``mysql.stats_meta``.
        """
        name = check_identifier(table).split(".")[-1]
        meta = self.db.query("SHOW STATS_META WHERE Table_name = %s", [name], self.max_rows)
        col = {c: i for i, c in enumerate(meta.columns)}
        rows = []
        for row in meta.rows:
            analyzed = (
                row[col["Last_analyze_time"]] is not None if "Last_analyze_time" in col else True
            )
            healthy = stats_healthy(
                int(row[col["Modify_count"]]), int(row[col["Row_count"]]), analyzed
            )
            rows.append([*row, healthy])
        return QueryResult(columns=[*meta.columns, "Healthy"], rows=rows, truncated=meta.truncated)

    def table_schema(self, table: str) -> QueryResult:
        """``SHOW CREATE TABLE``: columns, types and indexes."""
        return self.db.query(f"SHOW CREATE TABLE {check_identifier(table)}", max_rows=1)

    def bindings(self) -> QueryResult:
        """Global SQL bindings (plan baselines) that may force a plan."""
        return self.db.query("SHOW GLOBAL BINDINGS", max_rows=self.max_rows)
