from __future__ import annotations

import sys
from pathlib import Path

from research_agent.toolkit.mcp_client import McpTools
from research_agent.toolkit.specs import SPECS

SERVER = str(Path(__file__).with_name("mcp_fake_server.py"))


def test_tools_round_trip_over_stdio() -> None:
    client = McpTools([sys.executable, SERVER], start_timeout_s=60)
    try:
        names = [t.name for t in client.list_tools()]
        assert len(names) == len(SPECS) - 1 and "execute_write" not in names
        out = client.call("table_stats", {"table": "orders"})
        assert (out.text, out.evidence, out.error) == ('table_stats:{"table": "orders"}', 1, False)
        proposal = client.call("propose_analyze", {"table": "orders", "reason": "r"})
        assert proposal.pending == {"statement": "ANALYZE TABLE orders"}
        refused = client.call("execute_write", {"statement": "SET GLOBAL x = 1"})
        assert refused.error and refused.text.startswith("拒绝")
    finally:
        client.close()
