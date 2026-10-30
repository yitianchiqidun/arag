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
