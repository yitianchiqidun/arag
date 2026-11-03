"""Sentence splitting for the sentence index (semantic_search) and keyword_search snippets.

Upstream splits at ``.``, ``!``, ``?`` and newlines (``scripts/build_index.py`` and
``KeywordSearchTool``), and the index drops sentences of 10 characters or fewer. On Chinese
documentation that goes wrong twice:

- ``。！？；`` are not split, so a paragraph of several sentences becomes one "sentence";
- every ``.`` is split, so ``v8.5.0`` becomes ``v8`` / ``5`` / ``0`` and ``1.5`` loses its
  decimals, and the short pieces are then dropped.

The multilingual splitter also splits at Chinese sentence punctuation, and splits at ``.!?``
only before whitespace or the end of the text. Its minimum length counts a CJK character as
two, roughly one English word, so a short Chinese sentence is kept where upstream would keep
an English one of similar content.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

from research_agent.upstream import load_script

SentenceSplit = Literal["upstream", "multilingual"]
# Recorded in index manifests, so a change of splitter is detected.
SPLITTER_IDS: dict[SentenceSplit, str] = {
    "upstream": "scripts/build_index.py:split_sentences",
    "multilingual": "research_agent/ingest/sentences.py:split_multilingual",
}

_BOUNDARY = re.compile(r"[。！？；!?;]*[。！？；]+|[.!?]+(?=\s|$)|\n+")
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
MIN_WEIGHT = 10  # upstream keeps sentences longer than 10 characters


def weight(text: str) -> int:
    """Length with each CJK character counted twice."""
    return len(text) + len(_CJK.findall(text))


def split_multilingual(text: str, min_weight: int = MIN_WEIGHT) -> list[str]:
    """Sentences of ``text`` whose :func:`weight` exceeds ``min_weight`` (0 keeps all)."""
    parts = (p.strip() for p in _BOUNDARY.split(text))
    return [p for p in parts if p and weight(p) > min_weight]


def index_splitter(name: SentenceSplit) -> Callable[[str], list[str]]:
    """The function that splits chunk texts into the sentences of the sentence index."""
    if name == "upstream":
        split: Callable[[str], list[str]] = load_script("build_index").split_sentences
        return split
    return split_multilingual


def snippet_splitter(name: SentenceSplit) -> Callable[[str], list[str]] | None:
    """Replacement for ``KeywordSearchTool._split_sentences`` (None keeps upstream's)."""
    if name == "upstream":
        return None
    return lambda text: split_multilingual(text, min_weight=0)
