from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from research_agent.diagnosis import scenarios as sc
from research_agent.diagnosis.run import (
    answer_docs,
    comparison_markdown,
    load_results,
    missing_keywords,
    run_all,
    run_scenario,
    summary_markdown,
    unapproved_writes,
)
from research_agent.eval.citations import ChunkRef
from research_agent.tidb.db import QueryResult

from .test_tidb_tools import FakeDatabase

META = QueryResult(
    columns=["Db_name", "Table_name", "Partition_name", "Update_time", "Modify_count", "Row_count"],
    rows=[["shop", "orders", "", "t", 60000, 90000]],
)
INDEXES = QueryResult(
    columns=["Table", "Non_unique", "Key_name"],
    rows=[["orders", 0, "PRIMARY"], ["orders", 1, "idx_user"], ["orders", 1, "idx_status"]],
)


@pytest.fixture
def db() -> FakeDatabase:
    return FakeDatabase(
        {
            "STATS_META": META,
            "SHOW INDEX": INDEXES,
            "CONNECTION_ID": QueryResult(columns=["id"], rows=[[42]]),
            "COUNT(*) FROM orders": QueryResult(columns=["n"], rows=[[90000]]),
            "EXPLAIN": QueryResult(columns=["id"], rows=[["HashAgg_7"], ["IndexLookUp_16"]]),
        }
    )


def statements(db: FakeDatabase) -> list[str]:
    return [sql for sql, _ in db.executed]


def test_missing_index_drops_added_indexes_and_restores_the_threshold(db: FakeDatabase) -> None:
    s = sc.MissingIndex(lambda: db)
    s.inject()
    assert statements(db) == [
        "DROP INDEX `idx_status` ON orders",
        "SET GLOBAL tidb_slow_log_threshold = 10",
    ]
    s.workload()
    assert sum(q[0] in sc.MissingIndex.QUERIES for q in db.queries) == 40
    db.executed.clear()
    s.cleanup()
    assert statements(db) == [
        "DROP INDEX `idx_status` ON orders",
        "SET GLOBAL tidb_slow_log_threshold = 300",
    ]
    assert s.accepts(
        {
            "kind": "create_index",
            "target": "orders",
            "statement": "CREATE INDEX i ON orders (status, created_at)",
        }
    )
    assert not s.accepts(
        {
            "kind": "create_index",
            "target": "orders",
            "statement": "CREATE INDEX i ON orders (created_at, status)",
        }
    )
    assert not s.accepts(
        {"kind": "analyze_table", "target": "orders", "statement": "ANALYZE TABLE orders"}
    )


def test_memory_quota_accepts_only_a_large_enough_quota(db: FakeDatabase) -> None:
    s = sc.MemoryQuota(lambda: db)
    s.inject()
    s.cleanup()
    assert statements(db) == [
        "SET GLOBAL tidb_mem_quota_query = 1048576",
        "SET GLOBAL tidb_mem_quota_query = 1073741824",
    ]
    ok = {
        "kind": "set_global",
        "target": "tidb_mem_quota_query",
        "statement": "SET GLOBAL tidb_mem_quota_query = '1073741824'",
    }
    assert s.accepts(ok)
    assert not s.accepts(ok | {"statement": "SET GLOBAL tidb_mem_quota_query = '2097152'"})
    assert not s.accepts(ok | {"target": "tidb_mem_oom_action"})


def test_stale_stats_imports_rows_and_cleans_up(db: FakeDatabase) -> None:
    s = sc.StaleStats(lambda: db)
    s.inject()
    inserted = statements(db)
    assert inserted[:2] == ["ANALYZE TABLE orders", "SET GLOBAL tidb_enable_auto_analyze = OFF"]
    assert [sql.split("SELECT id + ")[1].split(",")[0] for sql in inserted[2:]] == [
        "1000000",
        "1030000",
    ]
    db.executed.clear()
    s.cleanup()
    assert statements(db) == [
        "DELETE FROM orders WHERE id > 1000000",
        "ANALYZE TABLE orders",
        "SET GLOBAL tidb_enable_auto_analyze = ON",
    ]
    assert s.cleanup_errors == []
    assert s.accepts(
        {"kind": "analyze_table", "target": "shop.orders", "statement": "ANALYZE TABLE shop.orders"}
    )
    assert s.accepts(
        {
            "kind": "set_global",
            "target": "tidb_enable_auto_analyze",
            "statement": "SET GLOBAL tidb_enable_auto_analyze = 'ON'",
        }
    )
    assert not s.accepts(
        {
            "kind": "set_global",
            "target": "tidb_enable_auto_analyze",
            "statement": "SET GLOBAL tidb_enable_auto_analyze = 'OFF'",
        }
    )


def test_stale_stats_refuses_to_run_without_a_visible_fault(db: FakeDatabase) -> None:
    db.responses["EXPLAIN"] = QueryResult(columns=["id"], rows=[["TableReader_12"]])
    with pytest.raises(RuntimeError, match="not visible"):
        sc.StaleStats(lambda: db).inject()


def test_lock_wait_holds_a_row_lock_until_cleanup(db: FakeDatabase) -> None:
    s = sc.LockWait(lambda: db)
    s.inject()
    assert statements(db)[:2] == [
        "BEGIN PESSIMISTIC",
        "UPDATE orders SET amount = amount + 0 WHERE id = 100",
    ]
    assert s.keywords()[-1] == ("42",)  # the holder's session must be named
    s.cleanup()
    assert s.waiter is not None and not s.waiter.is_alive()
    assert "ROLLBACK" in statements(db) and not s.accepts({"kind": "set_global"})


def test_missing_keywords_and_unapproved_writes() -> None:
    groups = [("status",), ("索引", "index"), ("全表扫描", "TableFullScan")]
    assert missing_keywords("STATUS 列没有 Index，导致 tablefullscan", groups) == []
    assert missing_keywords("没有索引", groups) == [["status"], ["全表扫描", "TableFullScan"]]
    lines = [
        json.dumps({"confirmed": True, "executed": True}),
        json.dumps({"confirmed": False, "executed": False}),
        json.dumps({"confirmed": False, "executed": True}),
    ]
    assert unapproved_writes(lines) == 1


REFS = {
    "7": ChunkRef(doc="explain-overview.md"),
    "9": ChunkRef(doc="system-variables.md"),
}


class FakeScenario(sc.Scenario):
    name = "fake"
    title = "假场景"
    symptom = "查询变慢了"
    root_cause = fix = ""
    expected_write = "CREATE INDEX"
    related_docs = ("explain-",)
    cause_keywords = (("status",),)

    def __init__(self, fail: bool = False) -> None:
        super().__init__(lambda: FakeDatabase())
        self.fail = fail
        self.log: list[str] = []

    def inject(self) -> None:
        self.log.append("inject")
        if self.fail:
            raise RuntimeError("injection failed")

    def accepts(self, pending: dict[str, Any]) -> bool:
        return "status" in pending["statement"]

    def cleanup(self) -> None:
        self.log.append("cleanup")


class FakeApp:
    def __init__(self, proposals: list[str]) -> None:
        self.proposals = proposals
        self.decisions: list[bool] = []

    def ask(self, session_id: str, question: str) -> list[dict[str, Any]]:
        return [
            {"type": "route", "intent": "diagnosis"},
            {"type": "tool_call", "tool": "slow_queries"},
            {"type": "tool_call", "tool": "explain"},
            {"type": "done", "latency_s": 10.0},
        ]

    def pending(self, session_id: str) -> dict[str, Any] | None:
        if len(self.decisions) < len(self.proposals):
            return {"kind": "create_index", "statement": self.proposals[len(self.decisions)]}
        return None

    def confirm(self, session_id: str, approved: bool) -> list[dict[str, Any]]:
        self.decisions.append(approved)
        return [
            {
                "type": "answer",
                "text": "根因：status 没有索引（Chunk 7）",
                "cost": 0.01,
                "citations": ["explain-overview.md › 执行计划"],
            },
            {"type": "done", "latency_s": 2.5},
        ]


def test_run_scenario_scores_and_approves_only_the_fix(tmp_path: Path) -> None:
    app = FakeApp(["CREATE INDEX a ON orders (created_at)", "CREATE INDEX b ON orders (status)"])
    scenario = FakeScenario()
    r = run_scenario(app, scenario, "s", tmp_path / "audit.jsonl", tmp_path / "e.jsonl", REFS)
    assert app.decisions == [False, True]
    assert (r.diagnosed, r.fix_proposed, r.related_doc_cited) == (True, True, True)
    assert r.answer_docs == ["explain-overview.md"]
    assert (r.intent, r.tool_calls, r.latency_s, r.cost) == ("diagnosis", 2, 15.0, 0.02)
    assert r.cited_docs == ["explain-overview.md"] and r.unapproved_writes == 0
    assert scenario.log == ["inject", "cleanup"]
    assert (
        "✓ CREATE INDEX a ON orders (created_at)（拒绝）；CREATE INDEX b ON orders (status)（批准）"
        in summary_markdown([r])
    )


def test_run_scenario_cleans_up_after_an_error(tmp_path: Path) -> None:
    scenario = FakeScenario(fail=True)
    r = run_scenario(FakeApp([]), scenario, "s", tmp_path / "audit.jsonl", tmp_path / "e.jsonl")
    assert scenario.log == ["inject", "cleanup"]
    assert r.error == "RuntimeError: injection failed" and not r.diagnosed


def test_answer_docs_from_chunk_ids_and_file_names() -> None:
    answer = "依据：Chunk 9 和 chunk 7；另见 system-variables.md 与 troubleshoot-lock-conflicts.md"
    assert answer_docs(answer, REFS) == ["system-variables.md", "explain-overview.md"]
    assert answer_docs("没有引用", REFS) == []


def test_comparison_rescores_earlier_runs(tmp_path: Path) -> None:
    first = run_scenario(
        FakeApp(["CREATE INDEX b ON orders (status)"]),
        FakeScenario(),
        "s",
        tmp_path / "a.jsonl",
        tmp_path / "e1.jsonl",
    )
    old = tmp_path / "old"
    old.mkdir()
    (old / "results.jsonl").write_text(first.model_dump_json() + "\n", "utf-8")
    before = load_results([old], REFS)  # the fake scenario has no related docs registered
    assert before[0].answer_docs == ["explain-overview.md"] and not before[0].related_doc_cited
    second = first.model_copy(update={"diagnosed": False})
    table = comparison_markdown(before, [second])
    assert (
        "| 假场景 | 修改前 | diagnosis | ✓ |" in table
        and "| 假场景 | 修改后 | diagnosis | ✗ |" in table
    )
    assert "修改前：诊断正确 1 / 1" in table and "修改后：诊断正确 0 / 1" in table
    assert "回答引用文档 1 / 1" in table


def test_run_all_stops_at_the_cost_limit(tmp_path: Path) -> None:
    scenarios = [FakeScenario(), FakeScenario()]
    app = FakeApp(["CREATE INDEX b ON orders (status)"])
    results = run_all(app, scenarios, tmp_path, tmp_path / "audit.jsonl", refs=REFS, max_cost=0.005)
    assert len(results) == 1 and scenarios[1].log == []
    assert "回答引用文档 1 / 1" in (tmp_path / "summary.md").read_text("utf-8")


def test_account_locked_creates_and_drops_the_locked_user(db: FakeDatabase) -> None:
    s = sc.AccountLocked(lambda: db)
    s.inject()
    s.cleanup()
    assert statements(db) == [
        "CREATE USER IF NOT EXISTS %s@%s IDENTIFIED BY %s",
        "GRANT SELECT ON shop.* TO 'report'@'%'",
        "ALTER USER 'report'@'%' ACCOUNT LOCK",
        "DROP USER IF EXISTS 'report'@'%'",
    ]
    assert s.expected_write is None and not s.accepts({"kind": "set_global"})
    assert (
        missing_keywords("账号 report 被锁定，需要 ALTER USER ... ACCOUNT UNLOCK", s.keywords())
        == []
    )
