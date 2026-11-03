"""Tests against the local TiDB cluster; skipped unless ``TIDB_LIVE=1``.

python -m research_agent.tidb start && python -m research_agent.tidb setup
TIDB_LIVE=1 .venv/bin/pytest tests/research_agent/test_tidb_live.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from research_agent.tidb.db import TiDBConfig, admin_database, readonly_database
from research_agent.tidb.readonly import ReadOnlyTools
from research_agent.tidb.setup import verify_readonly
from research_agent.tidb.write import WriteExecutor, WriteTools
from research_agent.upstream import load_env

pytestmark = pytest.mark.skipif(os.environ.get("TIDB_LIVE") != "1", reason="needs TIDB_LIVE=1")


@pytest.fixture(scope="module")
def cfg() -> TiDBConfig:
    load_env()
    return TiDBConfig()


def test_readonly_account_cannot_write(cfg: TiDBConfig) -> None:
    result = verify_readonly(cfg)
    accepted = [sql for sql, error in result.items() if error is None]
    assert accepted == [], f"the read-only account ran: {accepted}"


def test_readonly_tools_read_real_data(cfg: TiDBConfig) -> None:
    tools = ReadOnlyTools(readonly_database(cfg))
    assert tools.run_sql("SELECT COUNT(*) FROM orders").rows[0][0] > 0
    assert tools.system_variables("tidb_mem_quota_query").rows
    assert {r[0] for r in tools.cluster_info().rows} >= {"tidb", "tikv", "pd"}
    plan = tools.explain("SELECT * FROM orders WHERE status = 'paid'")
    assert any("TableFullScan" in " ".join(map(str, row)) for row in plan.rows)


def test_unconfirmed_write_changes_nothing(cfg: TiDBConfig, tmp_path: Path) -> None:
    reader = ReadOnlyTools(readonly_database(cfg))
    executor = WriteExecutor(admin_database(cfg), reader, tmp_path / "audit.jsonl")
    pending = WriteTools(reader).propose_set_global("tidb_mem_quota_query", "2147483648")
    before = reader.system_variables("tidb_mem_quota_query").rows[0][1]
    result = executor.execute(pending, confirmed=False)
    assert not result.executed
    assert reader.system_variables("tidb_mem_quota_query").rows[0][1] == before


def test_every_readonly_tool_runs(cfg: TiDBConfig) -> None:
    """Catches privileges the read-only account lacks (e.g. SHOW STATS_HEALTHY needs the whole
    mysql schema, so table_stats computes the health from mysql.stats_meta instead)."""
    tools = ReadOnlyTools(readonly_database(cfg))
    tools.slow_queries(5)
    tools.lock_waits()
    tools.transactions()
    tools.deadlocks(5)
    tools.statement_summary(5)
    tools.bindings()
    assert tools.table_stats("orders").rows[0][-1] is not None
    assert "CREATE TABLE" in tools.table_schema("orders").rows[0][1]
