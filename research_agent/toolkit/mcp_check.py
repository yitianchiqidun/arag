"""Calls every tool of the MCP server once through an MCP client (stage 10 check):

    python -m research_agent.toolkit.mcp_check [--config configs/assistant.yaml] [--out FILE]

Needs the local cluster. The propose_* calls must come back with a pending write and leave the
cluster unchanged; a call to the forbidden ``execute_write`` must be refused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research_agent.toolkit.specs import SPECS

SAMPLES: dict[str, dict[str, Any]] = {
    "search_docs": {"question": "tidb_mem_quota_query 的默认值是多少？"},
    "run_sql": {"sql": "SELECT status, COUNT(*) FROM orders GROUP BY status"},
    "show_variables": {"pattern": "tidb_mem_quota_query"},
    "slow_queries": {"limit": 3},
    "explain": {"sql": "SELECT * FROM orders WHERE status = 'paid'"},
    "lock_waits": {},
    "transactions": {},
    "deadlocks": {"limit": 3},
    "statement_summary": {"limit": 3},
    "table_stats": {"table": "orders"},
    "table_schema": {"table": "orders"},
    "bindings": {},
    "cluster_info": {},
    "search_logs": {"component": "tidb", "keyword": "CRUCIAL OPERATION", "max_lines": 3},
    "show_config": {"component": "tikv", "name": "quota.foreground"},
    "export_report": {
        "title": "MCP 检查",
        "symptom": "检查 export_report 能否通过 MCP 调用",
        "evidence": ["show_variables 返回 tidb_mem_quota_query = 1073741824"],
        "root_cause": "无（只是检查）",
        "actions": ["无"],
        "citations": ["system-variables.md"],
    },
    "propose_set_global": {
        "name": "tidb_mem_quota_query",
        "value": "2147483648",
        "reason": "MCP 检查",
    },
    "propose_analyze": {"table": "orders", "reason": "MCP 检查"},
    "propose_create_index": {"table": "orders", "columns": ["status"], "reason": "MCP 检查"},
    "execute_write": {"statement": "SET GLOBAL tidb_mem_quota_query = 1"},
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m research_agent.toolkit.mcp_check")
    parser.add_argument("--config", default="configs/assistant.yaml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    from research_agent.assistant.app import AssistantConfig, mcp_command
    from research_agent.tidb.db import TiDBConfig, admin_database
    from research_agent.toolkit.mcp_client import McpTools
    from research_agent.upstream import load_env

    load_env()
    cfg = AssistantConfig.from_yaml(args.config).model_copy(
        update={"mcp_server_config": args.config}
    )
    admin = admin_database(TiDBConfig())

    def cluster_state() -> list[Any]:
        variable = admin.query("SHOW GLOBAL VARIABLES LIKE 'tidb_mem_quota_query'").rows
        indexes = sorted(str(r[2]) for r in admin.query("SHOW INDEX FROM orders").rows)
        return [variable, indexes]

    before = cluster_state()
    client = McpTools(mcp_command(cfg))
    listed = [t.name for t in client.list_tools()]
    rows = []
    for spec in SPECS:
        out = client.call(spec.name, SAMPLES[spec.name])
        rows.append(
            {
                "tool": spec.name,
                "listed": spec.name in listed,
                "permission": spec.permission,
                "ok": not out.error,
                "chars": len(out.text),
                "pending": (out.pending or {}).get("statement"),
                "cost": round(out.cost, 4),
                "head": out.text[:120].replace("\n", " | "),
            }
        )
    client.close()
    after = cluster_state()
    report = {"listed_tools": len(listed), "calls": rows, "cluster_unchanged": before == after}
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    for r in rows:
        print(
            f"{r['tool']:22s} listed={r['listed']!s:5s} ok={r['ok']!s:5s} chars={r['chars']:5d} "
            f"pending={r['pending']} | {str(r['head'])[:70]}"
        )
    print(f"listed {len(listed)} tools; cluster unchanged: {before == after}")


if __name__ == "__main__":
    main()
