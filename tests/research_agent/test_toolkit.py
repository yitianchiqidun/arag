from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from research_agent.assistant.report import ReportWriter
from research_agent.tidb.config_reader import ConfigReader
from research_agent.tidb.db import QueryResult
from research_agent.tidb.logs import LogSearch
from research_agent.tidb.readonly import ReadOnlyTools
from research_agent.tidb.write import WriteTools
from research_agent.toolkit.harness import ToolHarness, TurnBudget, validate
from research_agent.toolkit.local import InProcessTools, ToolOutput
from research_agent.toolkit.specs import SPEC_BY_NAME, SPECS, exposed

from .test_assistant_graph import FakeDocs
from .test_tidb_tools import FakeDatabase


def log_dir(tmp_path: Path) -> Path:
    (tmp_path / "tidb-0").mkdir(parents=True)
    lines = [
        f'[2026/11/03 10:00:0{i}] [WARN] [server.go] ["handshake"] user=report try={i}'
        for i in range(5)
    ]
    lines += ["[INFO] CRUCIAL OPERATION alter user report account lock " + "x" * 600]
    (tmp_path / "tidb-0" / "tidb.log").write_text("\n".join(lines) + "\n", "utf-8")
    return tmp_path


def test_log_search_returns_the_newest_matches(tmp_path: Path) -> None:
    logs = LogSearch(log_dir(tmp_path), max_line_chars=100)
    text = logs.search("tidb", "HANDSHAKE", max_lines=2)
    assert text.splitlines()[0] == "tidb-0/tidb.log：匹配 5 行，显示最新的 2 行"
    assert text.splitlines()[1:] == [
        l for l in text.splitlines()[1:] if "try=3" in l or "try=4" in l
    ]
    long = logs.search("tidb", "crucial", 5).splitlines()[1]
    assert len(long) == 101 and long.endswith("…")
    assert "没有找到 pd 的日志" in logs.search("pd", "x")
    with pytest.raises(ValueError, match="组件"):
        logs.search("mysql", "x")


def test_config_reader_sends_one_fixed_statement() -> None:
    db = FakeDatabase(
        {
            "SHOW CONFIG": QueryResult(
                columns=["Type", "Name", "Value"], rows=[["tikv", "quota.x", "0"]]
            )
        }
    )
    out = ConfigReader(db).show_config("tikv", "quota")
    assert out.rows == [["tikv", "quota.x", "0"]]
    assert db.queries[0][:2] == (
        "SHOW CONFIG WHERE type = %s AND name LIKE %s",
        ["tikv", "%quota%"],
    )
    with pytest.raises(ValueError):
        ConfigReader(db).show_config("tikv", "x'; SET CONFIG")
    with pytest.raises(ValueError):
        ConfigReader(db).show_config("mysql")


def test_report_has_every_section_and_cuts_long_fields(tmp_path: Path) -> None:
    msg = ReportWriter(tmp_path / "r").export(
        title="内存上限过小",
        symptom="报 8175",
        evidence=["变量为 1048576"] * 30,
        root_cause="x" * 3000,
        actions=["恢复 1 GiB"],
        citations=[],
    )
    path = Path(msg.split("：", 1)[1].split("（")[0])
    text = path.read_text("utf-8")
    for section in (
        "# 内存上限过小",
        "## 症状",
        "## 证据",
        "## 根因",
        "## 处理",
        "## 引用\n\n- （无）",
    ):
        assert section in text
    assert text.count("- 变量为 1048576") == 20 and "x" * 1500 + "…" in text


def services(tmp_path: Path, reader: bool = True) -> SimpleNamespace:
    db = FakeDatabase(
        {
            "GLOBAL VARIABLES": QueryResult(
                columns=["Variable_name", "Value"], rows=[["tidb_mem_quota_query", "1048576"]]
            )
        }
    )
    tools = ReadOnlyTools(db)
    return SimpleNamespace(
        reader=tools if reader else None,
        writes=WriteTools(tools) if reader else None,
        docs=FakeDocs(),
        logs=LogSearch(log_dir(tmp_path)),
        configs=None,
        reports=ReportWriter(tmp_path / "reports"),
    )


def test_in_process_tools(tmp_path: Path) -> None:
    tools = InProcessTools(services(tmp_path))
    assert tools.call("show_variables", {"pattern": "tidb_mem%"}).evidence == 1
    docs = tools.call("search_docs", {"question": "默认值？"})
    assert docs.evidence == 1 and docs.cost == 0.01 and "Chunk 408" in docs.text
    proposal = tools.call(
        "propose_set_global",
        {"name": "tidb_mem_quota_query", "value": "1073741824", "reason": "太小"},
    )
    assert (
        proposal.pending is not None
        and proposal.pending["statement"] == "SET GLOBAL tidb_mem_quota_query = 1073741824"
    )
    assert proposal.pending["impact"].endswith("原因：太小")
    assert tools.call("search_logs", {"component": "tidb", "keyword": "lock"}).evidence == 1
    assert "读不到组件配置" in tools.call("show_config", {"component": "tikv"}).text
    assert tools.call("run_sql", {"sql": "DELETE FROM orders"}).error
    down = InProcessTools(services(tmp_path / "b", reader=False))
    assert down.call("lock_waits", {}).cluster_down


class SlowBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def call(self, name: str, args: dict[str, Any]) -> ToolOutput:
        self.calls.append(name)
        if name == "cluster_info":
            time.sleep(0.5)
        if name == "propose_analyze":
            return ToolOutput("ANALYZE TABLE orders", pending={"statement": "ANALYZE TABLE orders"})
        return ToolOutput("r" * 5000 if name == "run_sql" else f"{name} ok", evidence=1)


def test_validate() -> None:
    spec = SPEC_BY_NAME["slow_queries"]
    assert validate(spec, {"limit": "5", "bogus": 1}) == {"limit": 5, "min_seconds": 0}
    for bad, message in (
        ({"limit": 0}, "范围"),
        ({"limit": "a"}, "integer"),
        ({"limit": True}, "integer"),
    ):
        with pytest.raises(ValueError, match=message):
            validate(spec, bad)
    with pytest.raises(ValueError, match="缺少参数"):
        validate(SPEC_BY_NAME["run_sql"], {})
    with pytest.raises(ValueError, match="只能是"):
        validate(SPEC_BY_NAME["search_logs"], {"component": "mysql"})


def test_harness_permissions_budget_duplicates_timeout_and_audit(tmp_path: Path) -> None:
    backend = SlowBackend()
    audit = tmp_path / "audit.jsonl"
    harness = ToolHarness(backend, audit, TurnBudget(max_calls=5, max_doc_searches=1))
    assert harness.call("execute_write", {"statement": "x"}).text.startswith("拒绝")
    assert harness.call("drop_everything", {}).error
    assert harness.call("lock_waits", {}, intent="param_lookup").text.startswith("拒绝")
    assert harness.call("run_sql", {}).text.startswith("参数错误")
    assert backend.calls == []  # none of these reached the backend
    long = harness.call("run_sql", {"sql": "SELECT 1"})
    assert long.text.endswith("…（结果共 5000 字，已截断）") and len(long.text) < 4100
    again = harness.call("run_sql", {"sql": "SELECT 1"})
    assert again.text.startswith("（与本轮第 1 次调用相同") and backend.calls == ["run_sql"]
    assert harness.call("propose_analyze", {"table": "orders", "reason": "r"}).pending is not None
    assert (
        "不再重复提出" in harness.call("propose_analyze", {"table": "orders", "reason": "r"}).text
    )
    assert not harness.call("search_docs", {"question": "q"}).error
    assert "文档检索次数" in harness.call("search_docs", {"question": "q2"}).text
    slow_spec = SPEC_BY_NAME["cluster_info"]
    harness.specs = {
        **harness.specs,
        "cluster_info": type(slow_spec)(**{**slow_spec.__dict__, "timeout_s": 0.1}),
    }
    assert harness.call("cluster_info", {}).text.startswith("执行超时")
    assert harness.call("lock_waits", {}).text == "lock_waits ok"  # the 5th call that ran
    assert "预算" in harness.call("transactions", {}).text
    statuses = [json.loads(l)["status"] for l in audit.read_text("utf-8").splitlines()]
    assert statuses[:4] == ["refused", "refused", "refused", "invalid"]
    assert {"duplicate", "timeout", "over_budget", "ok"} <= set(statuses)
    harness.start_turn()
    assert harness.call("lock_waits", {}).text == "lock_waits ok"


def test_tools_are_exposed_by_intent() -> None:
    names = {
        i: [s.name for s in exposed(i)]
        for i in ("param_lookup", "research", "diagnosis", "out_of_scope")
    }
    assert names["param_lookup"] == names["research"] == ["search_docs"]
    assert names["out_of_scope"] == []
    assert "execute_write" not in names["diagnosis"] and len(names["diagnosis"]) == len(SPECS) - 1
    assert ToolHarness(SlowBackend()).schemas("out_of_scope") == []
