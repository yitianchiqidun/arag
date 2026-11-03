"""Heading-aware chunking with a token limit.

A document is cut at every heading (outside code fences) into sections that remember their
heading path. Consecutive sections of the same document are then packed greedily into chunks
of at most ``max_tokens``, so short sections do not become tiny chunks; a section longer than
the limit is split at blank lines, and a single block that is still too long is split by
lines (tables repeat their header row, code blocks are re-fenced).

Each chunk's text starts with one header line, ``[docs/path.md] Heading > Subheading``, so the
agent sees where the text comes from (and can cite it) without a separate metadata lookup.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from research_agent.ingest.loaders import Document
from research_agent.ingest.markdown import FENCE, fence_update

TokenCounter = Callable[[str], int]
HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
TABLE_SEPARATOR = re.compile(r"^\|?[ \t]*:?-{3,}")


class ChunkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(default=1000, ge=50)  # of the body, excluding the header line
    merge_sections: bool = True
    source_header: bool = True
    tokenizer: str = "o200k_base"  # tiktoken encoding used to count tokens


class Chunk(BaseModel):
    doc: str
    title: str
    position: int  # 0-based index within the document
    headings: list[str]  # heading path of the first section in the chunk
    text: str  # what the tools return: header line + body
    tokens: int  # of ``text``


@dataclass(frozen=True)
class Section:
    level: int  # 0 for text before the first heading
    path: tuple[str, ...]
    text: str  # heading line + body


def tiktoken_counter(encoding: str = "o200k_base") -> TokenCounter:
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    return lambda text: len(enc.encode(text, disallowed_special=()))


def parse_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    path: list[tuple[int, str]] = []
    buf: list[str] = []
    level = 0
    fence: str | None = None

    def flush() -> None:
        body = "\n".join(buf).strip("\n")
        if body.strip():
            sections.append(Section(level, tuple(h for _, h in path), body))

    for line in text.splitlines():
        match = HEADING.match(line) if fence is None else None
        if match:
            flush()
            buf = [line]
            level = len(match.group(1))
            path = [(lv, h) for lv, h in path if lv < level] + [(level, match.group(2))]
        else:
            buf.append(line)
        fence = fence_update(line, fence)
    flush()
    return sections


def split_paragraphs(text: str) -> list[str]:
    """Blocks separated by blank lines; a fenced code block is never cut."""
    paragraphs: list[str] = []
    cur: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        if fence is None and not line.strip():
            if cur:
                paragraphs.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
        fence = fence_update(line, fence)
    if cur:
        paragraphs.append("\n".join(cur))
    return paragraphs


def _pack(
    items: list[str], limit: int, count: TokenCounter, sep: str, head: str = "", tail: str = ""
) -> list[str]:
    """Greedy packing of ``items`` joined by ``sep``, each piece wrapped in head/tail."""

    def wrap(xs: list[str]) -> str:
        return head + sep.join(xs) + tail

    pieces: list[str] = []
    acc: list[str] = []
    for item in items:
        if acc and count(wrap([*acc, item])) > limit:
            pieces.append(wrap(acc))
            acc = []
        acc.append(item)
    if acc:
        pieces.append(wrap(acc))
    return pieces


def _split_block(block: str, limit: int, count: TokenCounter) -> list[str]:
    """Split one paragraph that is longer than the limit."""
    lines = block.splitlines()
    if len(lines) >= 3 and FENCE.match(lines[0]) and FENCE.match(lines[-1]):
        head, tail = lines[0] + "\n", "\n" + lines[-1]
        return _pack(lines[1:-1], limit, count, "\n", head, tail)
    if len(lines) >= 3 and lines[0].lstrip().startswith("|") and TABLE_SEPARATOR.match(lines[1]):
        return _pack(lines[2:], limit, count, "\n", "\n".join(lines[:2]) + "\n")
    if len(lines) > 1:
        out: list[str] = []
        for piece in _pack(lines, limit, count, "\n"):
            out.extend(_split_block(piece, limit, count) if count(piece) > limit else [piece])
        return out
    # One long line: sentences, then words.
    parts = re.split(r"(?<=[.!?])\s+", block)
    if len(parts) == 1:
        parts = block.split(" ")
        if len(parts) == 1:
            return [block]  # an unbreakable token run; kept whole
    return _pack(parts, limit, count, " ")


def split_long(text: str, limit: int, count: TokenCounter) -> list[str]:
    pieces: list[str] = []
    for paragraph in split_paragraphs(text):
        if count(paragraph) > limit:
            pieces.extend(_split_block(paragraph, limit, count))
        else:
            pieces.append(paragraph)
    return _pack(pieces, limit, count, "\n\n")


def header_line(doc: Document, headings: list[str]) -> str:
    return f"[{doc.path}] " + (" > ".join(headings) if headings else doc.title)


def chunk_document(doc: Document, cfg: ChunkerConfig, count: TokenCounter) -> list[Chunk]:
    limit = cfg.max_tokens
    units: list[tuple[tuple[str, ...], str]] = []
    for section in parse_sections(doc.text):
        if count(section.text) <= limit:
            units.append((section.path, section.text))
        else:
            units.extend((section.path, piece) for piece in split_long(section.text, limit, count))

    groups: list[tuple[tuple[str, ...], list[str]]] = []
    for path, body in units:
        if cfg.merge_sections and groups and count("\n\n".join([*groups[-1][1], body])) <= limit:
            groups[-1][1].append(body)
        else:
            groups.append((path, [body]))

    chunks: list[Chunk] = []
    for position, (path, bodies) in enumerate(groups):
        body = "\n\n".join(bodies).strip()
        headings = list(path)
        text = f"{header_line(doc, headings)}\n\n{body}" if cfg.source_header else body
        chunks.append(
            Chunk(
                doc=doc.path,
                title=doc.title,
                position=position,
                headings=headings,
                text=text,
                tokens=count(text),
            )
        )
    return chunks
