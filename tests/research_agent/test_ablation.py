from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from arag.core.context import AgentContext
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.read_chunk import ReadChunkTool
from research_agent.agent.prompt import (
    FULL_CHUNK_STRATEGY,
    SNIPPET_STRATEGY,
    build_system_prompt,
    load_upstream_prompt,
)
from research_agent.agent.tools import FullChunkSearch, build_tool_registry, parse_chunk_ids
from research_agent.config import AgentConfig, RunConfig


def run_config(corpus: dict[str, Path], **agent: object) -> RunConfig:
    return RunConfig.model_validate(
        {
            "llm": {"model": "m"},
            "judge": {"model": "m"},
            "agent": agent,
            "data": {
                "chunks_file": str(corpus["chunks"]),
                "index_dir": str(corpus["root"] / "no_index"),
            },
        }
    )


@pytest.mark.parametrize(
    ("tools", "search_result", "error"),
    [
        (["read_chunk"], "snippet", "search tool"),
        (["keyword_search", "keyword_search", "read_chunk"], "snippet", "duplicate"),
        (["keyword_search", "read_chunk"], "full_chunk", "replaces read_chunk"),
        (["keyword_search"], "snippet", "only sees snippets"),
    ],
)
def test_agent_config_rejects_meaningless_ablations(
    tools: list[str], search_result: str, error: str
) -> None:
    with pytest.raises(ValidationError, match=error):
        AgentConfig.model_validate({"tools": tools, "search_result": search_result})


def test_full_config_keeps_upstream_prompt_byte_identical() -> None:
    base = load_upstream_prompt()
    assert build_system_prompt(base, AgentConfig()) == base


def test_ablated_tools_are_removed_from_prompt() -> None:
    base = load_upstream_prompt()
    prompt = build_system_prompt(base, AgentConfig(tools=["semantic_search", "read_chunk"]))
    assert "**keyword_search**" not in prompt and "**semantic_search**" in prompt
    assert len(prompt.splitlines()) == len(base.splitlines()) - 1

    no_read = build_system_prompt(
        base, AgentConfig(tools=["keyword_search", "semantic_search"], search_result="full_chunk")
    )
    assert "**read_chunk**" not in no_read
    assert FULL_CHUNK_STRATEGY in no_read and SNIPPET_STRATEGY not in no_read


def test_prompt_edit_fails_loudly_on_unexpected_base() -> None:
    with pytest.raises(ValueError, match="keyword_search"):
        build_system_prompt(
            "no tool list here", AgentConfig(tools=["semantic_search", "read_chunk"])
        )


def test_registry_contains_exactly_the_configured_tools(tiny_corpus: dict[str, Path]) -> None:
    cfg = run_config(tiny_corpus, tools=["keyword_search", "read_chunk"])
    assert build_tool_registry(cfg).list_tools() == ["keyword_search", "read_chunk"]

    full_chunk = run_config(tiny_corpus, tools=["keyword_search"], search_result="full_chunk")
    registry = build_tool_registry(full_chunk)
    assert registry.list_tools() == ["keyword_search"]
    assert isinstance(registry.get("keyword_search"), FullChunkSearch)


def test_missing_semantic_index_is_an_error_not_a_silent_ablation(
    tiny_corpus: dict[str, Path],
) -> None:
    with pytest.raises(FileNotFoundError, match="sentence_index.pkl"):
        build_tool_registry(run_config(tiny_corpus))


def test_parse_chunk_ids_handles_both_upstream_formats() -> None:
    keyword = "Chunk ID: 12, Matched keywords in chunk: ... a ...\n\nChunk ID: 3, Matched keywords"
    semantic = "Chunk ID: 7 (Similarity: 0.812)\nMatched: ...\n\nChunk ID: 70 (Similarity: -0.1)"
    assert parse_chunk_ids(keyword + " in chunk: x") == ["12", "3"]
    assert parse_chunk_ids(semantic) == ["7", "70"]


def full_chunk_keyword(corpus: dict[str, Path]) -> FullChunkSearch:
    chunks = str(corpus["chunks"])
    return FullChunkSearch(KeywordSearchTool(chunks), ReadChunkTool(chunks))


def test_full_chunk_search_returns_whole_chunks_and_dedups(tiny_corpus: dict[str, Path]) -> None:
    tool = full_chunk_keyword(tiny_corpus)
    context = AgentContext()

    text, log = tool.execute(context, keywords=["paris"])
    assert "alice was born in paris. she studied physics." in text  # full text, not a snippet
    assert "paris is the capital of france. it hosts the louvre." in text
    assert log["chunks_found"] == 2 and log["new_chunks_count"] == 2
    assert log["retrieved_tokens"] == context.total_retrieved_tokens > 0

    again, log2 = tool.execute(context, keywords=["louvre"])
    assert "(This chunk has been shown before)" in again and "hosts the louvre" not in again
    assert (log2["new_chunks_count"], log2["already_read_count"], log2["retrieved_tokens"]) == (
        0,
        1,
        0,
    )


def test_full_chunk_search_without_hits(tiny_corpus: dict[str, Path]) -> None:
    text, log = full_chunk_keyword(tiny_corpus).execute(AgentContext(), keywords=["tokyo"])
    assert text.startswith("No results") and log == {"retrieved_tokens": 0, "chunks_found": 0}


def test_full_chunk_schema_no_longer_points_to_read_chunk(tiny_corpus: dict[str, Path]) -> None:
    description = full_chunk_keyword(tiny_corpus).get_schema()["function"]["description"]
    assert "read_chunk" not in description and "abbreviated" not in description
    assert "full text" in description
    # The wrapped upstream schema is untouched.
    assert (
        "read_chunk"
        in KeywordSearchTool(str(tiny_corpus["chunks"])).get_schema()["function"]["description"]
    )
