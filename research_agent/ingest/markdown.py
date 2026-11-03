"""Markdown preprocessing shared by all corpora: front matter, code fences, layout markup.

``normalize_markdown`` keeps the text a reader sees and drops markup that carries no content
for retrieval: HTML comments, layout tags (the text between them is kept), images (their alt
text is kept) and ``{#anchor}`` attribute lists after headings.
"""

from __future__ import annotations

import re

import yaml

FENCE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})")
HEADING_ATTRS = re.compile(r"^(#{1,6}[ \t].*?)[ \t]*\{[^}]*\}[ \t]*$")
IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)(?:\{[^}]*\})?")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# Layout tags that carry no content; text between them is kept.
HTML_TAG = re.compile(
    r"</?(?:figure|figcaption|img|p|div|span|br|details|summary|center|picture|source|sup|sub"
    r"|iframe|video|a|b|i|strong|em)\b[^>]*>",
    re.IGNORECASE,
)


def split_front_matter(text: str) -> tuple[dict[str, object], str]:
    """``(metadata, body)`` for a document with an optional leading YAML block."""
    match = re.match(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", text, re.DOTALL)
    if not match:
        return {}, text
    meta = yaml.safe_load(match.group(1)) or {}
    return (meta if isinstance(meta, dict) else {}), text[match.end() :]


def fence_update(line: str, fence: str | None) -> str | None:
    """Track whether ``line`` opens or closes a fenced code block."""
    match = FENCE.match(line)
    if not match:
        return fence
    marker = match.group("fence")
    if fence is None:
        return marker
    rest = line.strip()
    if rest[0] == fence[0] and len(rest) >= len(fence) and set(rest) == {fence[0]}:
        return None
    return fence


def normalize_markdown(text: str) -> str:
    """Drop markup without content outside code blocks (see module docstring)."""
    text = HTML_COMMENT.sub("", text)
    out: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        if fence is None:
            heading = HEADING_ATTRS.match(line)
            if heading:
                line = heading.group(1)
            line = HTML_TAG.sub("", IMAGE.sub(lambda m: m.group(1), line))
            if not line.strip():
                line = ""
        out.append(line)
        fence = fence_update(line, fence)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"
