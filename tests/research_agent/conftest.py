from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def env_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAG_API_KEY", "test-key")
    monkeypatch.setenv("ARAG_BASE_URL", "https://example.invalid/v1")


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> dict[str, Path]:
    chunks = [
        "0:alice was born in paris. she studied physics.",
        "1:paris is the capital of france. it hosts the louvre.",
        "2:bob lives in berlin.",
    ]
    questions = [
        {"id": "q1", "question": "Where was Alice born?", "answer": "Paris"},
        {"id": "q2", "question": "Where does Bob live?", "answer": "Berlin"},
    ]
    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text(json.dumps(chunks))
    questions_file = tmp_path / "questions.json"
    questions_file.write_text(json.dumps(questions))
    return {"chunks": chunks_file, "questions": questions_file, "root": tmp_path}
