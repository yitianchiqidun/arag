"""Tool registry construction for A-RAG and its ablations.

Upstream ``BatchRunner._init_shared_tools`` always registers all tools and silently drops
``semantic_search`` when the index is missing. Here the tool set comes from
``AgentConfig.tools`` and a missing index is an error, because a silently smaller toolkit
would turn the full model into an unnoticed ablation.
"""

from __future__ import annotations

import copy
import re
import threading
from pathlib import Path
from typing import Any

from arag.core.context import AgentContext
from arag.tools.base import BaseTool
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.tools.registry import ToolRegistry
from arag.tools.semantic_search import SemanticSearchTool
from research_agent.config import DataConfig, EmbeddingConfig, RunConfig
from research_agent.ingest.sentences import snippet_splitter

# Both upstream search tools start each hit with "Chunk ID: <id>" followed by a fixed marker.
_CHUNK_ID_LINE = re.compile(
    r"^Chunk ID: (.+?)(?:, Matched keywords in chunk:| \(Similarity: -?[\d.]+\))", re.MULTILINE
)

_FULL_CHUNK_RETURNS = (
    "RETURNS: The full text of each matching chunk, ready to answer from. "
    "Chunks already shown earlier are marked instead of repeated."
)


class PlainTextEncoding:
    """A tiktoken encoding whose ``encode`` treats special-token text as plain text.

    Upstream tools and BaseAgent count tokens with ``encode(text)``, which raises on text such
    as ``<|endoftext|>``. The benchmark corpora never contain it, but documentation about
    tokenizers and chat templates does, and the whole question then fails. Only the token
    count of such text changes; everything else is delegated to the wrapped encoding.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        tokens: list[int] = self._inner.encode(text, disallowed_special=())
        return tokens

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def plain_text_tokens(obj: Any) -> None:
    """Swap ``obj.tokenizer`` for a :class:`PlainTextEncoding` (idempotent)."""
    if not isinstance(obj.tokenizer, PlainTextEncoding):
        obj.tokenizer = PlainTextEncoding(obj.tokenizer)


_semantic_cache: dict[tuple[str, str, str, str | None], SemanticSearchTool] = {}
_semantic_lock = threading.Lock()


def shared_semantic_tool(data: DataConfig, embedding: EmbeddingConfig) -> SemanticSearchTool:
    """One SemanticSearchTool (embedding model + sentence index) per process and setting.

    A suite runs several variants in one process; reloading the model for each would cost
    time and, on a 6 GB GPU, risk running out of memory.
    """
    key = (data.chunks_file, data.index_dir, embedding.model, embedding.device)
    with _semantic_lock:
        if key not in _semantic_cache:
            index_file = Path(data.index_dir) / "sentence_index.pkl"
            if not index_file.exists():
                raise FileNotFoundError(
                    f"{index_file} not found; build it with scripts/build_index.py "
                    "or remove semantic_search from agent.tools"
                )
            tool = SemanticSearchTool(
                chunks_file=data.chunks_file,
                index_dir=data.index_dir,
                model_name=embedding.model,
                device=embedding.device,  # type: ignore[arg-type]  # upstream: implicit Optional
            )
            plain_text_tokens(tool)
            _semantic_cache[key] = tool
        return _semantic_cache[key]


def parse_chunk_ids(search_result: str) -> list[str]:
    """Chunk ids, in rank order, from an upstream keyword/semantic search result."""
    return _CHUNK_ID_LINE.findall(search_result)


class FullChunkSearch(BaseTool):
    """Wraps an upstream search tool so it returns whole chunks instead of snippets.

    Ranking is delegated to the wrapped tool, so the only difference from A-RAG (Full) is
    who picks the chunks whose full text enters the context: the search ranking here, the
    model (via read_chunk) there. Chunks shown before are marked, like upstream read_chunk
    does, so de-duplication is not a second difference between the two variants.
    """

    def __init__(self, inner: BaseTool, reader: ReadChunkTool) -> None:
        self.inner = inner
        self.reader = reader

    @property
    def name(self) -> str:
        return self.inner.name

    def get_schema(self) -> dict[str, Any]:
        schema = copy.deepcopy(self.inner.get_schema())
        func = schema["function"]
        description: str = func["description"]
        if description.count("RETURNS:") != 1:
            raise ValueError(f"unexpected {self.name} description; cannot rewrite RETURNS")
        description = description.split("RETURNS:")[0] + _FULL_CHUNK_RETURNS
        func["description"] = description.replace(
            "Returns chunk IDs and abbreviated sentence snippets where the keywords appear.",
            "Returns the full text of chunks where the keywords appear.",
        )
        return schema

    def execute(self, context: AgentContext, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        # A scratch context keeps the snippet tokens out of the real retrieval statistics.
        snippet_result, inner_log = self.inner.execute(AgentContext(), **kwargs)
        found = int(inner_log.get("chunks_found", 0))
        chunk_ids = parse_chunk_ids(snippet_result)
        if len(chunk_ids) != found:
            raise RuntimeError(
                f"parsed {len(chunk_ids)} chunk ids but {self.name} reported {found}; "
                "the upstream result format changed"
            )
        if not chunk_ids:
            context.add_retrieval_log(tool_name=self.name, tokens=0, metadata=dict(kwargs))
            return snippet_result, {"retrieved_tokens": 0, "chunks_found": 0}

        parts: list[str] = []
        new_ids: list[str] = []
        seen_ids: list[str] = []
        tokens = 0
        for cid in chunk_ids:
            parts.append("=" * 80)
            parts.append(f"[Chunk {cid}]")
            if context.is_chunk_read(cid):
                seen_ids.append(cid)
                parts.append("(This chunk has been shown before)")
                continue
            text = self.reader.chunks_dict[cid]
            parts += ["-" * 80, text]
            tokens += len(self.reader.tokenizer.encode(text))
            context.mark_chunk_as_read(cid)
            new_ids.append(cid)
        parts.append("=" * 80)

        context.add_retrieval_log(
            tool_name=self.name,
            tokens=tokens,
            metadata={**kwargs, "chunk_ids": chunk_ids, "new_chunks_read": new_ids},
        )
        log = {
            "retrieved_tokens": tokens,
            "chunks_found": found,
            "new_chunks_count": len(new_ids),
            "already_read_count": len(seen_ids),
        }
        return "\n".join(parts), log


def build_tool_registry(cfg: RunConfig) -> ToolRegistry:
    """Registry with exactly the tools in ``cfg.agent.tools`` (in upstream registration order)."""
    agent, data = cfg.agent, cfg.data
    tools = ToolRegistry()
    reader = ReadChunkTool(chunks_file=data.chunks_file)
    plain_text_tokens(reader)
    searches: dict[str, BaseTool] = {}
    if "keyword_search" in agent.tools:
        keyword = KeywordSearchTool(chunks_file=data.chunks_file)
        plain_text_tokens(keyword)
        split = snippet_splitter(data.sentence_split)
        if split is not None:
            keyword._split_sentences = split  # type: ignore[method-assign, assignment]
        searches["keyword_search"] = keyword
    if "semantic_search" in agent.tools:
        searches["semantic_search"] = shared_semantic_tool(data, cfg.embedding)
    if agent.search_result == "full_chunk":
        searches = {name: FullChunkSearch(tool, reader) for name, tool in searches.items()}

    if "keyword_search" in searches:
        tools.register(searches["keyword_search"])
    if "read_chunk" in agent.tools:
        tools.register(reader)
    if "semantic_search" in searches:
        tools.register(searches["semantic_search"])
    return tools
