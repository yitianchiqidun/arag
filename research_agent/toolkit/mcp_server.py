"""MCP server of the assistant's tools (stage 10, D37), over stdio only:

    python -m research_agent.toolkit.mcp_server --config configs/assistant.yaml

Exposes every tool of the registry except the forbidden ones: document search (A-RAG), the
cluster reads, logs, configuration, the local report, and the three ``propose_*`` tools,
which only return a statement and its impact (the PendingWrite in ``structuredContent``).
Nothing here can execute a write: the server opens no admin connection, and execution stays
inside the agent, after the user confirms through ``interrupt()``.

The heavy setup (embedding model, index, cluster connections) runs after the stdio transport
has diverted fd 1 to stderr, so stray output cannot corrupt the protocol stream.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from research_agent.toolkit.local import InProcessTools, ToolBackend
from research_agent.toolkit.specs import SPECS

EXPOSED = [s for s in SPECS if s.permission != "forbidden"]


def build_backend(config_path: str) -> InProcessTools:
    from research_agent.assistant.app import AssistantConfig, attach_tools
    from research_agent.assistant.retrieval import DocRetriever
    from research_agent.config import RunConfig
    from research_agent.upstream import load_env

    load_env()
    cfg = AssistantConfig.from_yaml(config_path)
    run = RunConfig.from_yaml(cfg.run_config)
    if cfg.embedding_device:
        run = run.model_copy(
            update={"embedding": run.embedding.model_copy(update={"device": cfg.embedding_device})}
        )
    tools = SimpleNamespace(
        reader=None, writes=None, configs=None, logs=None, reports=None, docs=DocRetriever(run)
    )
    attach_tools(tools, cfg)
    return InProcessTools(tools)


def make_server(backend: Any) -> Server[Any]:
    """``backend`` is a ToolBackend, or a zero-argument callable returning one (built lazily)."""
    tools = [
        types.Tool(
            name=s.name,
            description=s.description,
            input_schema=s.input_schema(),
            annotations=types.ToolAnnotations(
                read_only_hint=s.category != "report", destructive_hint=False
            ),
        )
        for s in EXPOSED
    ]
    names = {s.name for s in EXPOSED}
    state: dict[str, ToolBackend] = {}

    def get_backend() -> ToolBackend:
        if "backend" not in state:
            state["backend"] = (
                backend() if callable(backend) and not hasattr(backend, "call") else backend
            )
        return state["backend"]

    async def list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        if params.name not in names:
            text = f"拒绝：{params.name} 不通过 MCP 开放（写操作只能在 Agent 内经用户确认后执行）。"
            return types.CallToolResult(content=[types.TextContent(text=text)], is_error=True)
        backend = await anyio.to_thread.run_sync(get_backend)
        out = await anyio.to_thread.run_sync(
            backend.call, params.name, dict(params.arguments or {})
        )
        return types.CallToolResult(
            content=[types.TextContent(text=out.text)],
            structured_content=out.to_dict(),
            is_error=out.error,
        )

    return Server("tidb-ops-tools", on_list_tools=list_tools, on_call_tool=call_tool)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m research_agent.toolkit.mcp_server")
    parser.add_argument("--config", default="configs/assistant.yaml")
    args = parser.parse_args()

    async def serve() -> None:
        async with stdio_server() as (read, write):
            # Built before serving (the client's initialize waits), with fd 1 already diverted.
            backend = await anyio.to_thread.run_sync(build_backend, args.config)
            server = make_server(backend)
            await server.run(read, write, server.create_initialization_options())

    anyio.run(serve)


if __name__ == "__main__":
    main()
