from __future__ import annotations

import pytest

from research_agent.eval.metrics import hop_group, mcnemar_exact_p, repeat_counts


def test_repeat_counts_reads_and_searches() -> None:
    trajectory = [
        {"tool_name": "keyword_search", "arguments": {"keywords": ["B", "a"]}},
        {"tool_name": "read_chunk", "new_chunks_count": 2, "already_read_count": 0},
        {"tool_name": "keyword_search", "arguments": {"keywords": ["a ", "b"]}},  # same search
        {"tool_name": "semantic_search", "arguments": {"query": "Who  is X"}},
        {"tool_name": "semantic_search", "arguments": {"query": "who is x", "top_k": 10}},
        {"tool_name": "read_chunk", "new_chunks_count": 1, "already_read_count": 1},
    ]
    c = repeat_counts(trajectory)
    assert (c.full_text_chunks, c.repeated_chunks) == (4, 1)
    assert (c.searches, c.repeated_searches) == (4, 1)  # a different top_k is a new search


def test_full_chunk_search_steps_count_as_reads() -> None:
    c = repeat_counts(
        [
            {
                "tool_name": "keyword_search",
                "arguments": {},
                "new_chunks_count": 3,
                "already_read_count": 2,
            }
        ]
    )
    assert (c.full_text_chunks, c.repeated_chunks, c.searches) == (5, 2, 1)


def test_hop_group() -> None:
    assert hop_group("2hop__13548_13529") == "2hop"
    assert hop_group("musique_4hop1__94201_642284") == "4hop"  # rag_test dataset ids
    assert hop_group("3hop1__9285_5188_23307") == "3hop"
    assert hop_group("4hop3__x") == "4hop"
    assert hop_group("q1") is None
    assert hop_group("shop__1") is None


def test_mcnemar_exact_p() -> None:
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(5, 5) == 1.0
    assert mcnemar_exact_p(6, 17) == pytest.approx(0.0347, abs=1e-4)  # stage 0 plus models
    assert mcnemar_exact_p(17, 6) == mcnemar_exact_p(6, 17)
    assert mcnemar_exact_p(0, 5) == pytest.approx(2 / 32)
