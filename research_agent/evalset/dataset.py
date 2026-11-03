"""An evaluation set built from a document corpus, on disk.

A builder (rules over a reference page, an LLM rewrite, ...) writes the candidates; the
checks and the review add files next to them. Everything lives in one directory (committed
to the repository):

- ``candidates.jsonl``: the builder's output, one candidate per line (a rebuild rewrites it)
- ``meta.json``: which corpus build the chunk ids refer to
- ``closed_book.jsonl``: closed-book attempts per candidate (appended, resumable)
- ``prereview.jsonl``: LLM pre-reviews (appended, resumable; see review.py)
- ``review.json``: the final human decision per candidate (keep / drop, edits, reasons, note)
- ``review_first.json``: the first human decision per candidate, saved before the pre-review
  was shown and never changed afterwards (the agreement rate uses it)
- ``questions.json``: the final set, in the upstream question format plus evidence fields
- ``memorized.json``: candidates the agent model answered without documents (kept apart)

Check results and pre-reviews store the question and answer they were made for; after a
rebuild changes a candidate, the old result no longer applies and the check runs again.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from research_agent.config import LLMConfig, RunConfig
from research_agent.ingest.index import MANIFEST_FILE

CANDIDATES_FILE = "candidates.jsonl"
META_FILE = "meta.json"
CLOSED_BOOK_FILE = "closed_book.jsonl"
REVIEW_FILE = "review.json"
FIRST_JUDGMENTS_FILE = "review_first.json"
QUESTIONS_FILE = "questions.json"
MEMORIZED_FILE = "memorized.json"

# Why a question is dropped (or modified); ids are stored, labels are what the page shows.
DEFAULT_REASONS: dict[str, str] = {
    "answer_not_in_evidence": "答案不在原文",
    "answer_too_long": "答案过长",
    "unnatural_question": "问题不自然",
    "ambiguous": "有歧义",
    "other": "其他",
}
ReviewLabel = Literal["keep", "modify", "drop"]
Status = Literal["rejected", "pending", "closed_book", "partial_memory", "ready"]


class ScopeRule(BaseModel):
    """How the candidates of some groups are reviewed.

    With ``prereview`` an LLM reviews every candidate; a human reviews those it would modify
    or drop (or failed on) plus ``spot_check_n`` random ones it would keep. Without it, the
    builder's output is trusted and a human checks ``spot_check_n`` random candidates.
    """

    model_config = ConfigDict(extra="forbid")

    groups: list[str] = Field(min_length=1)
    prereview: bool = False
    spot_check_n: int = Field(default=0, ge=0)


class ReviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criteria: str  # numbered review criteria: shown on the review page and to the pre-review
    reasons: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_REASONS))
    rules: list[ScopeRule] = Field(min_length=1)
    llm: LLMConfig | None = None  # pre-review model; must differ from the agent model
    spot_check_seed: int = 2026
    summary_docs: list[str] = Field(default_factory=list)  # besides out_dir/README.md

    @model_validator(mode="after")
    def _consistent(self) -> ReviewConfig:
        groups = [g for r in self.rules for g in r.groups]
        if len(groups) != len(set(groups)):
            raise ValueError("a group is covered by more than one review rule")
        if any(r.prereview for r in self.rules) and self.llm is None:
            raise ValueError("a pre-review rule needs review.llm")
        if "other" not in self.reasons:
            raise ValueError('review.reasons needs an "other" reason')
        return self

    def rule_for(self, group: str) -> ScopeRule:
        rule = next((r for r in self.rules if group in r.groups), None)
        if rule is None:
            raise ValueError(f"no review rule covers group {group!r}")
        return rule


class ClosedBookConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: int = Field(default=3, ge=0)  # extra tries after a wrong answer at temperature 0
    temperature: float = 0.7


class EvalsetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source: str  # "source" field of the questions
    subject: str  # what the documents are about, for the prompts ("TiDB")
    run_config: str  # agent model, judge and the corpus build (data.chunks_file)
    out_dir: str
    groups: dict[str, str] = Field(min_length=1)  # question group id -> label
    closed_book: ClosedBookConfig = ClosedBookConfig()
    review: ReviewConfig | None = None

    @model_validator(mode="after")
    def _known_groups(self) -> EvalsetConfig:
        if self.review is not None:
            unknown = {g for r in self.review.rules for g in r.groups} - set(self.groups)
            if unknown:
                raise ValueError(f"review rules name unknown groups: {sorted(unknown)}")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalsetConfig:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    @property
    def out(self) -> Path:
        return Path(self.out_dir)

    def load_run_config(self) -> RunConfig:
        return RunConfig.from_yaml(self.run_config)


class Evidence(BaseModel):
    doc: str
    chunk_id: str | None = None  # in the build recorded in meta.json
    quote: str  # verbatim text that supports the answer


class Candidate(BaseModel):
    candidate_id: str
    group: str
    question: str
    answer: str
    answer_aliases: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)
    origin: dict[str, Any] = Field(default_factory=dict)  # how the builder made it
    reject_reason: str | None = None  # a builder check failed; kept for the statistics
    cost: float = 0.0  # of building it (an LLM rewrite), not of the checks
    # Shown to the pre-review and on the review page, e.g. similar parameters to rule out.
    review_notes: str = ""


class Attempt(BaseModel):
    temperature: float
    reply: str
    correct: bool


class ClosedBookResult(BaseModel):
    candidate_id: str
    question: str
    answer: str
    model: str
    attempts: list[Attempt] = Field(default_factory=list)
    error: str | None = None  # the check failed; a rerun retries it
    cost: float = 0.0


class Decision(BaseModel):
    decision: Literal["keep", "drop"]
    question: str | None = None  # edited text; None keeps the candidate's
    answer: str | None = None
    reasons: list[str] = Field(default_factory=list)  # ReviewConfig.reasons ids
    note: str = ""
    updated_at: str

    @property
    def label(self) -> ReviewLabel:
        """keep / modify (kept with edits) / drop, comparable with a pre-review decision."""
        if self.decision == "drop":
            return "drop"
        return "modify" if self.question or self.answer else "keep"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(tmp, 0o644)  # mkstemp creates 0600; these files are committed and shared
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_chunks(cfg: EvalsetConfig) -> dict[str, dict[str, Any]]:
    path = Path(cfg.load_run_config().data.chunks_file)
    return {c["id"]: c for c in json.loads(path.read_text(encoding="utf-8"))}


def write_candidates(cfg: EvalsetConfig, candidates: list[Candidate]) -> None:
    """Replace the candidates and record the corpus build their chunk ids refer to."""
    ids = [c.candidate_id for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate ids")
    unknown = {c.group for c in candidates} - set(cfg.groups)
    if unknown:
        raise ValueError(f"candidates in unknown groups: {sorted(unknown)}")
    chunks_path = Path(cfg.load_run_config().data.chunks_file)
    manifest_path = chunks_path.parent / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    meta = {
        "chunks_file": str(chunks_path),
        "chunks_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        "corpus_source": manifest.get("source"),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "groups": dict(Counter(c.group for c in candidates)),
    }
    lines = "".join(c.model_dump_json() + "\n" for c in candidates)
    _atomic_write(cfg.out / CANDIDATES_FILE, lines)
    _atomic_write(cfg.out / META_FILE, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")


def load_candidates(cfg: EvalsetConfig) -> list[Candidate]:
    return [Candidate.model_validate(r) for r in _read_jsonl(cfg.out / CANDIDATES_FILE)]


def check_build(cfg: EvalsetConfig) -> None:
    """Raise if the chunks file changed since the candidates were written (ids would not match)."""
    meta = json.loads((cfg.out / META_FILE).read_text())
    chunks_path = Path(meta["chunks_file"])
    if hashlib.sha256(chunks_path.read_bytes()).hexdigest() != meta["chunks_sha256"]:
        raise RuntimeError(f"{chunks_path} changed since the candidates were built")


class _MadeFor(Protocol):
    question: str
    answer: str


R = TypeVar("R", bound=_MadeFor)


def current(rows: dict[str, R], candidates: list[Candidate]) -> dict[str, R]:
    """Results (closed book, pre-review) made for each candidate's present question and answer."""
    by_id = {c.candidate_id: c for c in candidates}
    return {
        cid: row
        for cid, row in rows.items()
        if cid in by_id and (row.question, row.answer) == (by_id[cid].question, by_id[cid].answer)
    }


def load_closed_book(cfg: EvalsetConfig) -> dict[str, ClosedBookResult]:
    rows = [ClosedBookResult.model_validate(r) for r in _read_jsonl(cfg.out / CLOSED_BOOK_FILE)]
    return {r.candidate_id: r for r in rows}  # a rerun replaces the earlier line


def status(cand: Candidate, closed_book: ClosedBookResult | None) -> Status:
    if cand.reject_reason is not None:
        return "rejected"
    if closed_book is None or closed_book.error is not None or not closed_book.attempts:
        return "pending"
    if closed_book.attempts[0].correct:
        return "closed_book"
    if any(a.correct for a in closed_book.attempts):
        return "partial_memory"
    return "ready"


def statuses(cfg: EvalsetConfig, candidates: list[Candidate]) -> dict[str, Status]:
    results = current(load_closed_book(cfg), candidates)
    return {c.candidate_id: status(c, results.get(c.candidate_id)) for c in candidates}


def _load_decisions(path: Path) -> dict[str, Decision]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: Decision.model_validate(v) for k, v in data.items()}


def _write_decisions(path: Path, decisions: dict[str, Decision]) -> None:
    data = {k: v.model_dump() for k, v in sorted(decisions.items())}
    _atomic_write(path, json.dumps(data, ensure_ascii=False, indent=1) + "\n")


def load_review(cfg: EvalsetConfig) -> dict[str, Decision]:
    """Final decisions (``review.json``): what finalize uses."""
    return _load_decisions(cfg.out / REVIEW_FILE)


def save_decision(cfg: EvalsetConfig, candidate_id: str, decision: Decision | None) -> None:
    """Re-reads the file before writing, so two open review pages do not lose decisions."""
    review = load_review(cfg)
    if decision is None:
        review.pop(candidate_id, None)
    else:
        review[candidate_id] = decision
    _write_decisions(cfg.out / REVIEW_FILE, review)


def load_first_judgments(cfg: EvalsetConfig) -> dict[str, Decision]:
    """First judgments (``review_first.json``): what the agreement rate uses."""
    return _load_decisions(cfg.out / FIRST_JUDGMENTS_FILE)


def record_first_judgment(cfg: EvalsetConfig, candidate_id: str, decision: Decision) -> bool:
    """Store the reviewer's first judgment of a candidate, made before the pre-review was shown.

    Write-once: a later save, a cleared decision or an adopted suggestion never changes it,
    so the agreement rate cannot be influenced by having seen the pre-review. Returns whether
    it was written (False if a first judgment already existed).
    """
    first = load_first_judgments(cfg)
    if candidate_id in first:
        return False
    first[candidate_id] = decision
    _write_decisions(cfg.out / FIRST_JUDGMENTS_FILE, first)
    return True


def question_record(
    cfg: EvalsetConfig,
    cand: Candidate,
    decision: Decision | None,
    closed_book: Status,
    review: str | None = None,
) -> dict[str, Any]:
    return {
        "id": f"{cfg.name}__{cand.group}__{cand.candidate_id}",
        "source": cfg.source,
        "question": decision.question if decision and decision.question else cand.question,
        "answer": decision.answer if decision and decision.answer else cand.answer,
        "answer_aliases": cand.answer_aliases,
        "question_type": cand.group,
        "evidence": "",  # upstream field (a string, empty in the benchmark files)
        "evidence_chunks": [e.chunk_id for e in cand.evidence if e.chunk_id is not None],
        "evidence_docs": list(dict.fromkeys(e.doc for e in cand.evidence)),
        "evidence_quotes": [e.quote for e in cand.evidence],
        "closed_book": closed_book,
        "review": review,  # "human", "prereview" or "auto"; None for memorized.json
    }


class Stats(BaseModel):
    candidates: int
    by_group_status: dict[str, dict[str, int]]
    reject_reasons: dict[str, int]
    build_cost: float
    closed_book_cost: float
    reviewed: int
    kept: int


def stats(cfg: EvalsetConfig) -> Stats:
    candidates = load_candidates(cfg)
    st = statuses(cfg, candidates)
    review = load_review(cfg)
    by_group: dict[str, Counter[str]] = {}
    for c in candidates:
        by_group.setdefault(c.group, Counter())[st[c.candidate_id]] += 1
    return Stats(
        candidates=len(candidates),
        by_group_status={k: dict(v) for k, v in sorted(by_group.items())},
        reject_reasons=dict(Counter(c.reject_reason for c in candidates if c.reject_reason)),
        build_cost=round(sum(c.cost for c in candidates), 4),
        closed_book_cost=round(sum(r.cost for r in load_closed_book(cfg).values()), 4),
        reviewed=len(review),
        kept=sum(d.decision == "keep" for d in review.values()),
    )
