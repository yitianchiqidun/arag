"""Fixed, reproducible question subsets for development and benchmarking."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class SubsetMeta(BaseModel):
    source_file: str
    source_sha256: str
    source_size: int
    n: int
    seed: int
    created: str
    ids: list[str]


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_questions(questions: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Uniformly sample ``n`` questions with a fixed seed, keeping their original order."""
    if n > len(questions):
        raise ValueError(f"subset size {n} exceeds {len(questions)} questions")
    picked = set(random.Random(seed).sample(range(len(questions)), n))
    return [q for i, q in enumerate(questions) if i in picked]


def make_subset(source: str | Path, out: str | Path, n: int, seed: int) -> SubsetMeta:
    """Write ``out`` (same format as the source questions.json) and ``out.meta.json``."""
    with open(source, encoding="utf-8") as f:
        questions: list[dict[str, Any]] = json.load(f)
    subset = sample_questions(questions, n, seed)
    meta = SubsetMeta(
        source_file=str(source),
        source_sha256=sha256_file(source),
        source_size=len(questions),
        n=n,
        seed=seed,
        created=date.today().isoformat(),
        ids=[str(q["id"]) for q in subset],
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return meta
