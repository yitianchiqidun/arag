"""A stdio MCP server over a fake tool backend, started by test_mcp.py as a subprocess."""

from __future__ import annotations

import json
from typing import Any

import anyio
from mcp.server.stdio import stdio_server

from research_agent.toolkit.local import ToolOutput
from research_agent.toolkit.mcp_server import make_server


class Backend:
    def call(self, name: str, args: dict[str, Any]) -> ToolOutput:
        if name.startswith("propose_"):
            return ToolOutput("ANALYZE TABLE orders", pending={"statement": "ANALYZE TABLE orders"})
        return ToolOutput(f"{name}:{json.dumps(args, sort_keys=True)}", evidence=1)


async def main() -> None:
    async with stdio_server() as (read, write):
        server = make_server(Backend())
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(main)
