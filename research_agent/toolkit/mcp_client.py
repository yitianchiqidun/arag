"""MCP tool backend of the agent (stage 10): the same ``call(name, args) -> ToolOutput`` as the
in-process backend, served by ``research_agent.toolkit.mcp_server`` over stdio.

The MCP SDK is asynchronous and the graph is not: one background thread runs an event loop
that keeps the stdio session open, and calls are handed to it.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from research_agent.toolkit.local import ToolOutput


class McpTools:
    def __init__(
        self, command: list[str], start_timeout_s: float = 180.0, cwd: str | None = None
    ) -> None:
        self.params = StdioServerParameters(command=command[0], args=command[1:], cwd=cwd)
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._session: ClientSession | None = None
        self._stop: asyncio.Event | None = None
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="mcp-client"
        )
        self._thread.start()
        self._task = asyncio.run_coroutine_threadsafe(self._run(), self._loop)
        if not self._ready.wait(start_timeout_s):
            raise TimeoutError("MCP server did not start in time")
        if self._error is not None:
            raise RuntimeError(f"MCP server failed to start: {self._error}") from self._error

    async def _run(self) -> None:
        self._stop = asyncio.Event()
        try:
            async with (
                stdio_client(self.params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                self._session = session
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:  # noqa: BLE001 - surfaced to the constructor
            self._error = e
            self._ready.set()

    def _submit(self, coro: Any, timeout_s: float | None = None) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout_s)

    def list_tools(self) -> list[Any]:
        assert self._session is not None
        return list(self._submit(self._session.list_tools()).tools)

    def call(self, name: str, args: dict[str, Any]) -> ToolOutput:
        assert self._session is not None
        result = self._submit(self._session.call_tool(name, args))
        data = getattr(result, "structured_content", None)
        if isinstance(data, dict) and "text" in data:
            return ToolOutput(
                **{k: v for k, v in data.items() if k in ToolOutput.__dataclass_fields__}
            )
        text = "\n".join(getattr(c, "text", "") for c in getattr(result, "content", []) or [])
        return ToolOutput(text, error=bool(getattr(result, "is_error", False)))

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        try:
            self._task.result(timeout=10)
        except Exception:  # noqa: BLE001, S110 - closing anyway
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
