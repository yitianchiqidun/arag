"""Turn the selected files of a corpus source into normalized Markdown documents."""

from __future__ import annotations

import hashlib
import re
from fnmatch import fnmatchcase

from pydantic import BaseModel, ConfigDict, Field

from research_agent.ingest.markdown import normalize_markdown, split_front_matter
from research_agent.ingest.sources import CorpusSource

MARKDOWN_SUFFIXES = (".md", ".markdown")


class LoaderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # fnmatch patterns on source-relative paths; ``*`` also matches ``/``.
    include: list[str] = Field(min_length=1)
    exclude: list[str] = Field(default_factory=list)


class Document(BaseModel):
    path: str
    title: str
    text: str  # normalized Markdown, the input of the chunker
    sha256: str  # of ``text``


class LoadReport(BaseModel):
    files: int  # all files in the source
    selected: int
    excluded: dict[str, int]  # pattern -> files it removed
    skipped_empty: list[str] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)


def select_files(files: list[str], cfg: LoaderConfig) -> tuple[list[str], dict[str, int]]:
    selected: list[str] = []
    excluded = {pattern: 0 for pattern in cfg.exclude}
    for path in files:
        if not any(fnmatchcase(path, p) for p in cfg.include):
            continue
        hit = next((p for p in cfg.exclude if fnmatchcase(path, p)), None)
        if hit is not None:
            excluded[hit] += 1
        else:
            selected.append(path)
    return selected, excluded


def _first_heading(text: str) -> str | None:
    match = re.search(r"^#[ \t]+(.+?)[ \t]*#*[ \t]*$", text, re.MULTILINE)
    return match.group(1) if match else None


def load_document(source: CorpusSource, path: str) -> Document | None:
    """One normalized document; ``None`` for a file that is not Markdown."""
    if not path.lower().endswith(MARKDOWN_SUFFIXES):
        return None
    meta, body = split_front_matter(source.read_text(path))
    text = normalize_markdown(body)
    meta_title = meta.get("title")
    title = str(meta_title) if meta_title else _first_heading(text)
    return Document(
        path=path,
        title=title or path.rsplit("/", 1)[-1],
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def load_corpus(source: CorpusSource, cfg: LoaderConfig) -> tuple[list[Document], LoadReport]:
    files = source.list_files()
    selected, excluded = select_files(files, cfg)
    report = LoadReport(files=len(files), selected=len(selected), excluded=excluded)
    docs: list[Document] = []
    for path in selected:
        doc = load_document(source, path)
        if doc is None:
            report.unsupported.append(path)
        elif not doc.text.strip():
            report.skipped_empty.append(path)
        else:
            docs.append(doc)
    return docs, report
