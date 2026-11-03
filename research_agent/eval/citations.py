"""Answer citations: which chunks a final answer cites, and which document section each is.

The upstream system prompt already asks the agent to "cite the specific chunks that support
your answer", and answers do ("Source: Chunk 58", "(chunk 54, 55)"). Instead of changing the
prompt (which would make A-RAG (Full) differ from the paper's setup), the cited chunk ids are
parsed from the answer afterwards and resolved to the document path and heading path stored
with each chunk. Each citation is also checked against what the agent actually saw:

- ``read``: the chunk's full text was in the context (read_chunk, a full-chunk search, or a
  Naive RAG retrieval)
- ``snippet``: only a search snippet of it was shown
- ``unseen``: an existing chunk the agent never saw (a guessed id)
- ``unknown``: no chunk with this id

A chunk usually holds several sections; its label names all of them (chunks.json keeps their
heading paths in ``sections``), not only the first.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from research_agent.agent.tools import parse_chunk_ids
from research_agent.ingest.chunker import merge_heading_paths

CitationStatus = Literal["read", "snippet", "unseen", "unknown"]

# "Chunk 58", "chunks 54, 55 and 60", "[Chunk 12]", "Chunk ID: 7", "chunk #3".
_CHUNK_REF = re.compile(
    r"\bchunks?\b(?:\s*ids?)?[\s#:]*(\d+(?:\s*(?:,|/|&|\band\b|\bor\b)\s*(?:chunks?\s*#?\s*)?\d+)*)",
    re.IGNORECASE,
)
_FULL_CHUNK_HEADER = re.compile(r"^\[Chunk ([^\]\s]+)\]$", re.MULTILINE)
_SEARCH_TOOLS = ("keyword_search", "semantic_search")


class ChunkRef(BaseModel):
    doc: str | None = None  # None for corpora without metadata (the benchmark chunks)
    headings: list[str] = Field(default_factory=list)  # path of the chunk's first section
    sections: list[list[str]] = Field(default_factory=list)  # paths of all its sections


class Citation(BaseModel):
    chunk_id: str
    status: CitationStatus
    doc: str | None = None
    headings: list[str] = Field(default_factory=list)
    sections: list[list[str]] = Field(default_factory=list)

    def label(self) -> str:
        """``docs/x.md › A › B、C`` with every section of the chunk (shared parents written
        once), or just the chunk id when the corpus has no metadata."""
        if self.doc is None:
            return f"Chunk {self.chunk_id}"
        merged = merge_heading_paths(self.sections or [self.headings], sep=" › ")
        return f"{self.doc} › {merged}" if merged else self.doc


def load_chunk_refs(chunks_file: str | Path) -> dict[str, ChunkRef]:
    """Chunk id -> source, for both chunk file formats upstream accepts."""
    with open(chunks_file, encoding="utf-8") as f:
        data = json.load(f)
    refs: dict[str, ChunkRef] = {}
    for item in data:
        if isinstance(item, dict):
            refs[str(item["id"])] = ChunkRef(
                doc=item.get("doc"),
                headings=item.get("headings", []),
                sections=item.get("sections", []),
            )
        elif isinstance(item, str) and ":" in item:
            refs[item.split(":", 1)[0]] = ChunkRef()
    return refs


def cited_chunk_ids(answer: str) -> list[str]:
    """Chunk ids referenced in ``answer``, in order of first mention."""
    ids: list[str] = []
    for match in _CHUNK_REF.finditer(answer):
        for cid in re.findall(r"\d+", match.group(1)):
            if cid not in ids:
                ids.append(cid)
    return ids


def shown_chunk_ids(results: Iterable[tuple[str, str]]) -> tuple[set[str], set[str]]:
    """``(full text shown, snippet shown)`` from ``(tool name, tool result)`` pairs."""
    full: set[str] = set()
    snippet: set[str] = set()
    for tool, result in results:
        if tool == "dense_retrieval":  # Naive RAG puts the whole chunks in the prompt
            full.update(str(hit["chunk_id"]) for hit in json.loads(result))
            continue
        full.update(_FULL_CHUNK_HEADER.findall(result))  # read_chunk and full-chunk searches
        if tool in _SEARCH_TOOLS:
            snippet.update(parse_chunk_ids(result))
    return full, snippet - full


def resolve_citations(
    answer: str, refs: dict[str, ChunkRef], full: set[str], snippet: set[str]
) -> list[Citation]:
    citations: list[Citation] = []
    for cid in cited_chunk_ids(answer):
        ref = refs.get(cid)
        status: CitationStatus
        if ref is None:
            status = "unknown"
        elif cid in full:
            status = "read"
        elif cid in snippet:
            status = "snippet"
        else:
            status = "unseen"
        citations.append(
            Citation(
                chunk_id=cid,
                status=status,
                doc=ref.doc if ref else None,
                headings=ref.headings if ref else [],
                sections=ref.sections if ref else [],
            )
        )
    return citations


def record_citations(record: dict[str, Any], refs: dict[str, ChunkRef]) -> list[Citation]:
    """Citations of one prediction record (``pred_answer`` + upstream ``trajectory``)."""
    steps = record.get("trajectory") or []
    full, snippet = shown_chunk_ids((s["tool_name"], str(s.get("tool_result", ""))) for s in steps)
    return resolve_citations(str(record.get("pred_answer") or ""), refs, full, snippet)


class CitationStats(BaseModel):
    answers_citing: float  # share of answers citing at least one existing chunk
    cited: int  # citations of existing chunks
    cited_read: float | None  # share of those whose full text the agent had seen
    cited_snippet: float | None  # share of those the agent only saw as a search snippet
    cited_unseen: int  # citations of existing chunks never shown to the agent
    evidence_n: int  # questions with gold evidence chunks
    evidence_cited: float | None  # share of those whose answer cites a gold evidence chunk


def citation_stats(records: list[dict[str, Any]], refs: dict[str, ChunkRef]) -> CitationStats:
    citing = cited = read = snippet = unseen = evidence_n = evidence_hit = 0
    for r in records:
        citations = [c for c in record_citations(r, refs) if c.status != "unknown"]
        citing += bool(citations)
        cited += len(citations)
        read += sum(c.status == "read" for c in citations)
        snippet += sum(c.status == "snippet" for c in citations)
        unseen += sum(c.status == "unseen" for c in citations)
        evidence = r.get("evidence_chunks")
        if evidence:
            evidence_n += 1
            evidence_hit += any(c.chunk_id in evidence for c in citations)
    return CitationStats(
        answers_citing=citing / len(records) if records else 0.0,
        cited=cited,
        cited_read=read / cited if cited else None,
        cited_snippet=snippet / cited if cited else None,
        cited_unseen=unseen,
        evidence_n=evidence_n,
        evidence_cited=evidence_hit / evidence_n if evidence_n else None,
    )
