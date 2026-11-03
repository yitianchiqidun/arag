"""Per-question metrics computed from the tool trajectory and the question id."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from pydantic import BaseModel

SEARCH_TOOLS = frozenset({"keyword_search", "semantic_search"})
_HOP = re.compile(r"(?:^|_)(\d+)hop\d*__")
_EVALSET_ID = re.compile(
    r"^[A-Za-z0-9_.-]+?__([A-Za-z][A-Za-z0-9-]*)__[^_]+$"
)  # dataset.question_record


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
    """The group a question is reported in: the hop count of MuSiQue ids
    (``musique_2hop__…``, ``3hop1__…``), or the question type of built evalsets
    (``tidb_params__direct__d001`` -> ``direct``); other ids give None."""
    match = _HOP.search(qid)
    if match:
        return f"{match.group(1)}hop"
    built = _EVALSET_ID.match(qid)
    return built.group(1) if built else None


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs ``b`` and ``c``."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0**n
    return min(1.0, 2 * tail)


def _wilson(x: int, n: int, z: float) -> tuple[float, float]:
    center = (x + z * z / 2) / (n + z * z)
    half = z * math.sqrt(x * (n - x) / n + z * z / 4) / (n + z * z)
    return center - half, center + half


def paired_diff_ci(
    both: int, run_only: int, ref_only: int, neither: int, z: float = 1.959964
) -> tuple[float, float]:
    """95% CI of ``acc(run) - acc(ref)`` on paired 0/1 outcomes (Newcombe 1998, method 10).

    Square-and-add of the two Wilson intervals, corrected for the correlation between the
    paired outcomes. Unlike the Wald interval it stays inside [-1, 1] and behaves when the
    discordant counts are small (a few to a dozen in a 100-question table).
    """
    n = both + run_only + ref_only + neither
    if n == 0:
        raise ValueError("no questions")
    p1, p2 = (both + run_only) / n, (both + ref_only) / n
    l1, u1 = _wilson(both + run_only, n, z)
    l2, u2 = _wilson(both + ref_only, n, z)
    margins = (both + run_only) * (ref_only + neither) * (both + ref_only) * (run_only + neither)
    cross: float = both * neither - run_only * ref_only
    if cross > 0:  # Newcombe's correction, which keeps phi from overstating the correlation
        cross = max(cross - n / 2, 0.0)
    phi = cross / math.sqrt(margins) if margins else 0.0
    delta = p1 - p2
    low = delta - math.sqrt(
        max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    )
    high = delta + math.sqrt(
        max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    )
    return max(-1.0, low), min(1.0, high)
