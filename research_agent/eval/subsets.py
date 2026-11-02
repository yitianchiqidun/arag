"""Fixed, reproducible question subsets for development and benchmarking."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SubsetMeta(BaseModel):
    source_file: str
    source_sha256: str
    source_size: int
    n: int
    seed: int
    created: str
    # Subsets whose questions were removed from the pool before sampling (file -> sha256).
    excluded: dict[str, str] = Field(default_factory=dict)
    pool_size: int | None = None  # source_size minus the excluded questions
    ids: list[str]


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_questions(
    questions: list[dict[str, Any]],
    n: int,
    seed: int,
    exclude_ids: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Uniformly sample ``n`` questions with a fixed seed, keeping their original order.

    Questions in ``exclude_ids`` are removed from the pool first. Without exclusions the pool
    is ``range(len(questions))``, so subsets made before this option existed are unchanged.
    """
    pool = [i for i, q in enumerate(questions) if str(q["id"]) not in exclude_ids]
    if n > len(pool):
        raise ValueError(f"subset size {n} exceeds the {len(pool)} questions in the pool")
    picked = set(random.Random(seed).sample(pool, n))
    return [q for i, q in enumerate(questions) if i in picked]


def _load_ids(path: str | Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {str(q["id"]) for q in json.load(f)}


def make_subset(
    source: str | Path,
    out: str | Path,
    n: int,
    seed: int,
    exclude: list[str | Path] | None = None,
) -> SubsetMeta:
    """Write ``out`` (same format as the source questions.json) and ``out.meta.json``.

    ``exclude`` lists earlier subset files (same format) whose questions must not be drawn,
    e.g. a held-out test set that has to be disjoint from the development subsets.
    """
    with open(source, encoding="utf-8") as f:
        questions: list[dict[str, Any]] = json.load(f)
    source_ids = {str(q["id"]) for q in questions}
    exclude_ids: set[str] = set()
    for path in exclude or []:
        ids = _load_ids(path)
        if not ids <= source_ids:
            raise ValueError(f"{path} contains questions that are not in {source}")
        exclude_ids |= ids
    subset = sample_questions(questions, n, seed, frozenset(exclude_ids))
    meta = SubsetMeta(
        source_file=str(source),
        source_sha256=sha256_file(source),
        source_size=len(questions),
        n=n,
        seed=seed,
        created=date.today().isoformat(),
        excluded={str(path): sha256_file(path) for path in exclude or []},
        pool_size=len(questions) - len(exclude_ids),
        ids=[str(q["id"]) for q in subset],
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return meta
