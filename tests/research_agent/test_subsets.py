from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_agent.eval.subsets import make_subset, sample_questions, sha256_file

QUESTIONS = [{"id": f"q{i}", "question": f"Q{i}?", "answer": str(i)} for i in range(50)]


def test_sampling_is_deterministic_and_keeps_order() -> None:
    a = sample_questions(QUESTIONS, 10, seed=42)
    assert a == sample_questions(QUESTIONS, 10, seed=42)
    assert a != sample_questions(QUESTIONS, 10, seed=43)
    indices = [QUESTIONS.index(q) for q in a]
    assert indices == sorted(indices) and len(set(indices)) == 10


def test_subset_larger_than_source_rejected() -> None:
    with pytest.raises(ValueError):
        sample_questions(QUESTIONS, 51, seed=0)


def test_make_subset_writes_questions_and_meta(tmp_path: Path) -> None:
    source = tmp_path / "questions.json"
    source.write_text(json.dumps(QUESTIONS))
    out = tmp_path / "subsets" / "s.json"

    meta = make_subset(source, out, n=5, seed=1)

    written = json.loads(out.read_text())
    assert [q["id"] for q in written] == meta.ids and len(written) == 5
    saved_meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert saved_meta["source_sha256"] == sha256_file(source)
    assert saved_meta["source_size"] == 50 and saved_meta["seed"] == 1


def test_sampling_without_exclusions_matches_original_algorithm() -> None:
    # Subsets made before ``exclude_ids`` existed must be reproducible unchanged.
    import random

    picked = set(random.Random(42).sample(range(len(QUESTIONS)), 10))
    expected = [q for i, q in enumerate(QUESTIONS) if i in picked]
    assert sample_questions(QUESTIONS, 10, seed=42) == expected


def test_excluded_questions_are_never_drawn() -> None:
    excluded = frozenset(f"q{i}" for i in range(0, 50, 2))
    subset = sample_questions(QUESTIONS, 25, seed=3, exclude_ids=excluded)
    assert {q["id"] for q in subset} == {f"q{i}" for i in range(1, 50, 2)}
    with pytest.raises(ValueError, match="pool"):
        sample_questions(QUESTIONS, 26, seed=3, exclude_ids=excluded)


def test_make_subset_excludes_earlier_subsets(tmp_path: Path) -> None:
    source = tmp_path / "questions.json"
    source.write_text(json.dumps(QUESTIONS))
    dev = tmp_path / "dev.json"
    make_subset(source, dev, n=20, seed=1)

    meta = make_subset(source, tmp_path / "test.json", n=30, seed=2, exclude=[dev])

    dev_ids = {q["id"] for q in json.loads(dev.read_text())}
    assert not dev_ids & set(meta.ids)
    assert meta.pool_size == 30 and meta.excluded == {str(dev): sha256_file(dev)}


def test_make_subset_rejects_exclusions_from_another_source(tmp_path: Path) -> None:
    source = tmp_path / "questions.json"
    source.write_text(json.dumps(QUESTIONS))
    other = tmp_path / "other.json"
    other.write_text(json.dumps([{"id": "x1", "question": "?", "answer": "a"}]))
    with pytest.raises(ValueError, match="not in"):
        make_subset(source, tmp_path / "test.json", n=5, seed=0, exclude=[other])
