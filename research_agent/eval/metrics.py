"""Per-question metrics computed from the tool trajectory and the question id."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from pydantic import BaseModel

SEARCH_TOOLS = frozenset({"keyword_search", "semantic_search"})
_HOP = re.compile(r"(?:^|_)(\d+)hop\d*__")


class RepeatCounts(BaseModel):
    """Raw counts, so rates can be pooled over questions (micro average)."""

    full_text_chunks: int = 0  # chunk ids whose full text was requested/returned
    repeated_chunks: int = 0  # ... of which had already been shown in this question
    searches: int = 0
    repeated_searches: int = 0  # identical (tool, normalized arguments) seen earlier


def _search_key(step: dict[str, Any]) -> str:
    args = dict(step.get("arguments") or {})
    if isinstance(args.get("keywords"), list):
        args["keywords"] = sorted(str(k).strip().lower() for k in args["keywords"])
    if isinstance(args.get("query"), str):
        args["query"] = " ".join(args["query"].lower().split())
    return json.dumps([step["tool_name"], args], sort_keys=True, ensure_ascii=False)


def repeat_counts(trajectory: list[dict[str, Any]]) -> RepeatCounts:
    """Count repeated reads and repeated searches within one question.

    A "read" is any step that hands full chunk text to the model: upstream ``read_chunk`` and
    the ``full_chunk`` search wrapper both log ``new_chunks_count``/``already_read_count``
    (already-read chunks are answered with a short marker instead of the text).
    """
    counts = RepeatCounts()
    seen_searches: set[str] = set()
    for step in trajectory:
        if "already_read_count" in step:
            new, again = int(step.get("new_chunks_count", 0)), int(step["already_read_count"])
            counts.full_text_chunks += new + again
            counts.repeated_chunks += again
        if step.get("tool_name") in SEARCH_TOOLS:
            key = _search_key(step)
            counts.searches += 1
            counts.repeated_searches += key in seen_searches
            seen_searches.add(key)
    return counts


def hop_group(qid: str) -> str | None:
    """Hop count from MuSiQue ids (``musique_2hop__…``, ``3hop1__…``); other ids give None."""
    match = _HOP.search(qid)
    return f"{match.group(1)}hop" if match else None


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs ``b`` and ``c``."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0**n
    return min(1.0, 2 * tail)
