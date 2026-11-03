"""Markdown preprocessing shared by all corpora: front matter, code fences, layout markup.

``normalize_markdown`` keeps the text a reader sees and drops markup that carries no content
for retrieval. Outside code blocks it:

- drops HTML comments, Hugo shortcodes (``{{< copyable "sql" >}}``) and site components
  written as CamelCase tags (``<EmailSubscriptionWrapper />``);
- turns HTML tables into one line per row, cells joined by `` | ``, separated by blank lines
  so the chunker can split between rows; list items inside a table cell (HTML or Markdown
  table) are joined by ``；``;
- removes layout tags but keeps the text between them (``<span class="version-mark">从 v6.4.0
  版本开始引入</span>`` keeps the version note), turns ``<code>`` into backticks, keeps the
  alt text of images and drops ``{#anchor}`` attribute lists after headings;
- with ``link_targets=False``, keeps only the text of ``[text](target)`` links: the agent
  cannot follow a link, and targets (anchors, issue URLs) can be a sizeable share of tokens.

Unknown tags such as ``<host>:<port>`` placeholders are left alone.
"""

from __future__ import annotations

import html
import re

import yaml

FENCE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})")
HEADING_ATTRS = re.compile(r"^(#{1,6}[ \t].*?)[ \t]*\{[^}]*\}[ \t]*$")
IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)(?:\{[^}]*\})?")
# Link text may hold one level of brackets (``[`KILL [TIDB]`](...)``), a target one level of
# parentheses.
LINK = re.compile(
    r"(?<!!)\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\((?:[^()\s]|\([^()\s]*\))*(?:\s+\"[^\"]*\")?\)"
)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}")
COMPONENT = re.compile(r"</?[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b[^>]*>")
CODE_TAG = re.compile(r"</?code>", re.IGNORECASE)
# ``<ul><li>a</li><li>b</li></ul>`` inside a Markdown table cell -> ``a；b``.
LIST_ITEM_GAP = re.compile(r"</li>\s*<li\b[^>]*>", re.IGNORECASE)
LIST_TAG = re.compile(r"</?(?:ul|ol|li)\b[^>]*>", re.IGNORECASE)
# Layout tags that carry no content; text between them is kept.
HTML_TAG = re.compile(
    r"</?(?:figure|figcaption|img|p|div|span|br|details|summary|center|picture|source|sup|sub"
    r"|iframe|video|a|b|i|strong|em|kbd)\b[^>]*>",
    re.IGNORECASE,
)
# An HTML table block starts a line; "`... OF <table>`" in running text is not one.
TABLE_START = re.compile(r"^[ \t]*<table\b", re.IGNORECASE)
TABLE_END = re.compile(r"</table\s*>", re.IGNORECASE)
ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr\s*>", re.IGNORECASE | re.DOTALL)
CELL = re.compile(r"<t[hd]\b[^>]*>(.*?)</t[hd]\s*>", re.IGNORECASE | re.DOTALL)


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


def _cell_text(cell: str) -> str:
    cell = re.sub(r"<li\b[^>]*>", "；", cell, flags=re.IGNORECASE)
    cell = re.sub(r"<[^>]+>", "", CODE_TAG.sub("`", cell))  # inside a table cell, all tags
    text = re.sub(r"\s*；\s*", "；", " ".join(html.unescape(cell).split()))
    return re.sub(r"([：:])；", r"\1", text).strip("； ")


def html_table_to_text(block: str) -> str:
    """An HTML table as one line per row, blank lines between rows."""
    rows = []
    for row in ROW.findall(block):
        cells = [c for c in (_cell_text(x) for x in CELL.findall(row)) if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n\n".join(rows)


def _clean_line(line: str, link_targets: bool) -> str:
    heading = HEADING_ATTRS.match(line)
    if heading:
        line = heading.group(1)
    line = SHORTCODE.sub("", COMPONENT.sub("", line))
    line = IMAGE.sub(lambda m: m.group(1), line)
    if not link_targets:
        line = LINK.sub(lambda m: m.group(1), line)
    line = LIST_TAG.sub("", LIST_ITEM_GAP.sub("；", line))
    line = HTML_TAG.sub("", CODE_TAG.sub("`", line))
    return line if line.strip() else ""


def normalize_markdown(text: str, link_targets: bool = True) -> str:
    """Drop markup without content outside code blocks (see module docstring)."""
    text = HTML_COMMENT.sub("", text)
    out: list[str] = []
    fence: str | None = None
    table: list[str] | None = None  # lines of an HTML table being collected
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if table is not None:
            table.append(line)
            if TABLE_END.search(line):
                out += ["", html_table_to_text("\n".join(table)), ""]
                table = None
            continue
        if fence is None:
            if TABLE_START.search(line) and any(TABLE_END.search(x) for x in lines[i:]):
                table = [line]
                if TABLE_END.search(line):
                    out += ["", html_table_to_text(line), ""]
                    table = None
                continue
            line = _clean_line(line, link_targets)
        out.append(line)
        fence = fence_update(line, fence)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"
