from __future__ import annotations

import pytest

from research_agent.eval.metrics import (
    hop_group,
    mcnemar_exact_p,
    paired_diff_ci,
    repeat_counts,
)


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


def test_paired_diff_ci_matches_published_example() -> None:
    # Airway hyper-responsiveness example (n11=1, n12=1, n21=7, n22=12) in Fagerland,
    # Lydersen & Laake (2014), Stat Med 33:2850: Newcombe square-and-add -0.507 to -0.026.
    low, high = paired_diff_ci(both=1, run_only=1, ref_only=7, neither=12)
    assert low == pytest.approx(-0.507, abs=5e-4)
    assert high == pytest.approx(-0.026, abs=5e-4)


def test_paired_diff_ci_properties() -> None:
    # Stage 1: arag_full_stage0 vs arag_full (64 both right, 4 / 7 discordant, 25 both wrong).
    low, high = paired_diff_ci(both=64, run_only=4, ref_only=7, neither=25)
    assert low < -0.03 < high and low < 0 < high
    # Swapping the roles mirrors the interval.
    mirrored = paired_diff_ci(both=64, run_only=7, ref_only=4, neither=25)
    assert mirrored == pytest.approx((-high, -low))
    # Identical runs: centred on 0 and narrow; extreme runs stay within [-1, 1].
    low0, high0 = paired_diff_ci(both=70, run_only=0, ref_only=0, neither=30)
    assert low0 < 0 < high0 and high0 - low0 < 0.08
    assert paired_diff_ci(both=0, run_only=0, ref_only=100, neither=0)[0] >= -1.0
    # More discordant pairs with the same difference -> a wider interval.
    narrow = paired_diff_ci(both=60, run_only=5, ref_only=8, neither=27)
    wide = paired_diff_ci(both=50, run_only=15, ref_only=18, neither=17)
    assert wide[1] - wide[0] > narrow[1] - narrow[0]
    with pytest.raises(ValueError):
        paired_diff_ci(0, 0, 0, 0)
