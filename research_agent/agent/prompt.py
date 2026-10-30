"""System prompt for A-RAG ablations, derived from the upstream prompt file.

The upstream prompt lists all three tools and a "search → read → ..." strategy. When a tool
is ablated, its line is dropped so the prompt does not advertise a tool that is not in the
schema list. Every edit asserts that the text it targets exists, so a change to the upstream
prompt fails loudly instead of silently producing a different prompt.
"""

from __future__ import annotations

from research_agent.config import ALL_TOOLS, AgentConfig
from research_agent.upstream import REPO_ROOT

UPSTREAM_PROMPT_FILE = REPO_ROOT / "src/arag/agent/prompts/default.txt"

SNIPPET_STRATEGY = "search → read → evaluate → search → read → ... → answer"
FULL_CHUNK_STRATEGY = "search → evaluate → search → ... → answer"


def load_upstream_prompt() -> str:
    # Same read as scripts/batch_runner.py, so the Full config sends byte-identical text.
    return UPSTREAM_PROMPT_FILE.read_text()


def build_system_prompt(base: str, agent: AgentConfig) -> str:
    """Return ``base`` unchanged for A-RAG (Full); otherwise drop/adjust the affected lines."""
    lines = base.splitlines(keepends=True)
    for tool in ALL_TOOLS:
        if tool in agent.tools:
            continue
        prefix = f"- **{tool}**:"
        matches = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"expected one line starting with {prefix!r} in the base prompt")
        del lines[matches[0]]
    prompt = "".join(lines)
    if agent.search_result == "full_chunk":
        if prompt.count(SNIPPET_STRATEGY) != 1:
            raise ValueError(f"strategy sentence {SNIPPET_STRATEGY!r} not found in the base prompt")
        prompt = prompt.replace(SNIPPET_STRATEGY, FULL_CHUNK_STRATEGY)
    return prompt
