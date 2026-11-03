"""LLM pre-review, the human review scope, agreement, and the final question set.

Each group of candidates is reviewed by one rule (``ReviewConfig.rules``):

- **With pre-review**: a model other than the agent model reviews every ready candidate
  against the criteria (keep / modify / drop, reasons, suggested fixes). A human reviews
  every candidate it would modify or drop (or failed on) and a seeded random sample of those
  it would keep; the rest keep the pre-review's "keep".
- **Without pre-review**: the builder's output is trusted (e.g. answers extracted by rules)
  and a human checks a seeded random sample.

The human judges before seeing the pre-review. That first judgment is stored write-once in
``review_first.json``, apart from the final decision in ``review.json``; the agreement rate
uses first judgments only, so the agreement on the random sample estimates how often the
pre-review is right where it is trusted alone, unaffected by having seen it. ``finalize``
writes the questions and a summary table into the evalset README (and ``summary_docs``).
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field

from research_agent.evalset.dataset import (
    MEMORIZED_FILE,
    QUESTIONS_FILE,
    Candidate,
    EvalsetConfig,
    ReviewConfig,
    ReviewLabel,
    Status,
    _atomic_write,
    _read_jsonl,
    check_build,
    current,
    load_candidates,
    load_chunks,
    load_first_judgments,
    load_review,
    question_record,
    statuses,
)
from research_agent.evalset.prompts import (
    PREREVIEW_PROMPT_VERSION,
    PREREVIEW_TEMPLATE,
    passages_block,
)
from research_agent.llm.client import AccountError, client_from_config
from research_agent.llm.pricing import PricingTable

PREREVIEW_FILE = "prereview.jsonl"
SUMMARY_FILE = "review_summary.json"
MARKER_START = "<!-- evalset-review:start -->"
MARKER_END = "<!-- evalset-review:end -->"


class PreReview(BaseModel):
    candidate_id: str
    question: str  # what was reviewed; a rebuilt candidate needs a new pre-review
    answer: str
    model: str
    decision: ReviewLabel | None = None  # None: the call or the reply failed
    reasons: list[str] = Field(default_factory=list)
    explanation: str = ""
    suggested_question: str | None = None
    suggested_answer: str | None = None
    error: str | None = None
    cost: float = 0.0
    prompt_version: int = PREREVIEW_PROMPT_VERSION


def review_config(cfg: EvalsetConfig) -> ReviewConfig:
    if cfg.review is None:
        raise ValueError(f"{cfg.name} has no review section")
    agent = cfg.load_run_config().llm.model
    if cfg.review.llm is not None and cfg.review.llm.model == agent:
        raise ValueError(f"the pre-review model must differ from the agent model ({agent})")
    return cfg.review


def ready_candidates(cfg: EvalsetConfig, candidates: list[Candidate]) -> list[Candidate]:
    st = statuses(cfg, candidates)
    return [c for c in candidates if st[c.candidate_id] == "ready"]


def prereview_candidates(cfg: EvalsetConfig) -> list[Candidate]:
    review = review_config(cfg)
    ready = ready_candidates(cfg, load_candidates(cfg))
    return [c for c in ready if review.rule_for(c.group).prereview]


def prereview_prompt(
    cfg: EvalsetConfig,
    cand: Candidate,
    chunks: dict[str, dict[str, Any]],
    same_doc_questions: list[str],
) -> str:
    review = review_config(cfg)
    passages = [
        (f"{e.doc}，块 {e.chunk_id}", chunks[e.chunk_id]["text"])
        if e.chunk_id is not None and e.chunk_id in chunks
        else (e.doc, e.quote)
        for e in cand.evidence
    ]
    return PREREVIEW_TEMPLATE.format(
        subject=cfg.subject,
        criteria=review.criteria.strip(),
        reasons=", ".join(f"{k}（{v}）" for k, v in review.reasons.items()),
        group=cfg.groups[cand.group],
        question=cand.question,
        answer=cand.answer,
        aliases="、".join(cand.answer_aliases) or "（无）",
        passages=passages_block(passages),
        others="\n".join(f"- {q}" for q in same_doc_questions) or "（无）",
        notes=cand.review_notes.strip() or "（无）",
    )


def prereview_prompts(cfg: EvalsetConfig) -> dict[str, str]:
    """Prompt per candidate to pre-review; other questions on the same documents help spot
    duplicates."""
    todo = prereview_candidates(cfg)
    if any(e.chunk_id is not None for c in todo for e in c.evidence):
        check_build(cfg)
    chunks = load_chunks(cfg)
    prompts: dict[str, str] = {}
    for cand in todo:
        docs = {e.doc for e in cand.evidence}
        others = [
            c.question
            for c in todo
            if c.candidate_id != cand.candidate_id and docs & {e.doc for e in c.evidence}
        ][:10]
        prompts[cand.candidate_id] = prereview_prompt(cfg, cand, chunks, others)
    return prompts


def parse_prereview(reply: str, cand: Candidate, model: str, reasons: dict[str, str]) -> PreReview:
    base = PreReview(
        candidate_id=cand.candidate_id, question=cand.question, answer=cand.answer, model=model
    )
    start, end = reply.find("{"), reply.rfind("}")
    try:
        data = json.loads(reply[start : end + 1]) if start >= 0 else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or data.get("decision") not in ("keep", "modify", "drop"):
        return base.model_copy(update={"error": f"bad reply: {reply[:200]}"})
    known = [r if r in reasons else "other" for r in data.get("reasons") or []]
    return base.model_copy(
        update={
            "decision": data["decision"],
            "reasons": list(dict.fromkeys(known)),
            "explanation": str(data.get("explanation") or ""),
            "suggested_question": data.get("suggested_question") or None,
            "suggested_answer": (
                str(data["suggested_answer"]) if data.get("suggested_answer") else None
            ),
        }
    )


def load_prereviews(cfg: EvalsetConfig) -> dict[str, PreReview]:
    """Pre-reviews of the present candidates (a rerun after an error replaces the old line)."""
    rows = [PreReview.model_validate(r) for r in _read_jsonl(cfg.out / PREREVIEW_FILE)]
    return current({r.candidate_id: r for r in rows}, load_candidates(cfg))


class CostEstimate(BaseModel):
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost: float
    currency: str


def estimate_prereview(
    cfg: EvalsetConfig, count_tokens: Any, output_tokens: int = 200, overhead: float = 1.15
) -> CostEstimate:
    """Cost of pre-reviewing every candidate that needs it, counted locally (no API calls).

    ``overhead`` covers the difference between the local tokenizer and the provider's.
    """
    review = review_config(cfg)
    assert review.llm is not None
    pricing = PricingTable.from_yaml(cfg.load_run_config().pricing_file)
    inputs = [round(count_tokens(p) * overhead) for p in prereview_prompts(cfg).values()]
    price = pricing.get(review.llm.model)
    return CostEstimate(
        model=pricing.resolve(review.llm.model),
        calls=len(inputs),
        input_tokens=sum(inputs),
        output_tokens=output_tokens * len(inputs),
        cost=round(sum(price.cost(n, output_tokens, 0) for n in inputs), 4),
        currency=pricing.currency,
    )


def run_prereview(cfg: EvalsetConfig, workers: int = 5) -> list[PreReview]:
    """Pre-review every candidate that needs it and has no successful pre-review yet."""
    review = review_config(cfg)
    assert review.llm is not None
    llm_cfg = review.llm
    pricing = PricingTable.from_yaml(cfg.load_run_config().pricing_file)
    done = {k for k, v in load_prereviews(cfg).items() if v.decision is not None}
    by_id = {c.candidate_id: c for c in prereview_candidates(cfg)}
    prompts = {k: v for k, v in prereview_prompts(cfg).items() if k not in done}

    def one(cand: Candidate, prompt: str) -> PreReview:
        client = client_from_config(llm_cfg, pricing)
        try:
            result = client.chat(messages=[{"role": "user", "content": prompt}])
        except AccountError:
            raise
        except (requests.RequestException, KeyError) as e:  # recorded; a rerun retries it
            return PreReview(
                candidate_id=cand.candidate_id,
                question=cand.question,
                answer=cand.answer,
                model=llm_cfg.model,
                error=f"{type(e).__name__}: {e}",
            )
        reply = str(result["message"].get("content") or "")
        parsed = parse_prereview(reply, cand, llm_cfg.model, review.reasons)
        parsed.cost = round(sum(r.cost for r in client.records), 6)
        return parsed

    new: list[PreReview] = []
    path = cfg.out / PREREVIEW_FILE
    with ThreadPoolExecutor(max_workers=workers) as pool, open(path, "a", encoding="utf-8") as f:
        futures = [pool.submit(one, by_id[k], v) for k, v in prompts.items()]
        for future in as_completed(futures):
            row = future.result()
            f.write(row.model_dump_json() + "\n")
            f.flush()
            new.append(row)
    return new


def manual_scope(
    ready: list[Candidate], prereviews: dict[str, PreReview], review: ReviewConfig
) -> dict[str, str]:
    """Candidate id -> why a human reviews it: prereview_modify/drop/error or spot_check."""
    scope: dict[str, str] = {}
    for rule in review.rules:
        trusted: list[str] = []  # pre-review "keep", or every candidate without a pre-review
        for c in ready:
            if c.group not in rule.groups:
                continue
            pre = prereviews.get(c.candidate_id)
            if not rule.prereview:
                trusted.append(c.candidate_id)
            elif pre is None:
                continue  # not pre-reviewed yet: no scope (finalize refuses)
            elif pre.decision is None:
                scope[c.candidate_id] = "prereview_error"
            elif pre.decision in ("modify", "drop"):
                scope[c.candidate_id] = f"prereview_{pre.decision}"
            else:
                trusted.append(c.candidate_id)
        trusted.sort()
        n = min(rule.spot_check_n, len(trusted))
        for cid in random.Random(review.spot_check_seed).sample(trusted, n):
            scope[cid] = "spot_check"
    return scope


class Agreement(BaseModel):
    n: int
    exact: float | None  # same keep / modify / drop
    usable: float | None  # same "keep or modify" vs "drop"
    pairs: dict[str, int]  # "human/prereview" -> count


def agreement(pairs: list[tuple[ReviewLabel, ReviewLabel]]) -> Agreement:
    n = len(pairs)

    def usable(label: str) -> bool:
        return label != "drop"

    return Agreement(
        n=n,
        exact=sum(h == p for h, p in pairs) / n if n else None,
        usable=sum(usable(h) == usable(p) for h, p in pairs) / n if n else None,
        pairs=dict(Counter(f"{h}/{p}" for h, p in pairs)),
    )


class RuleSummary(BaseModel):
    groups: list[str]
    prereview: bool
    ready: int
    prereview_prompt_versions: dict[str, int]  # "v1" -> count
    prereview_decisions: dict[str, int]
    prereview_reasons: dict[str, int]
    scope: dict[str, int]  # why reviewed by hand -> count
    spot_check_n: int
    # Everything below about human judgments uses first judgments only.
    spot_check_first: dict[str, int]  # first-judgment labels on the random sample
    agreement_spot_check: Agreement | None  # pre-review rules: the sample of pre-review keeps
    agreement_all_human: Agreement | None  # pre-review rules: every human-reviewed candidate
    changed_after_first: dict[str, int]  # "first -> final" label, for changed decisions
    kept: int
    kept_by_source: dict[str, int]  # human / prereview / auto
    kept_modified: int
    dropped: int
    drop_reasons: dict[str, int]  # final drops (human reasons)


class ReviewSummary(BaseModel):
    prereview_model: str | None
    spot_check_seed: int
    rules: list[RuleSummary]
    without_first_judgment: int  # human-reviewed but no first judgment: left out of the stats
    kept: int
    kept_by_group: dict[str, int]
    rejected: int  # by the builder's checks
    memorized: int
    memorized_by_group: dict[str, int]


def build_final(
    cfg: EvalsetConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], ReviewSummary]:
    review_cfg = review_config(cfg)
    candidates = load_candidates(cfg)
    st = statuses(cfg, candidates)
    pending = [cid for cid, s in st.items() if s == "pending"]
    if pending:
        raise ValueError(f"{len(pending)} candidates have no closed-book check yet")
    ready = [c for c in candidates if st[c.candidate_id] == "ready"]
    prereviews = load_prereviews(cfg)
    decisions = load_review(cfg)
    first = load_first_judgments(cfg)
    unreviewed = [
        c
        for c in ready
        if review_cfg.rule_for(c.group).prereview and c.candidate_id not in prereviews
    ]
    if unreviewed:
        raise ValueError(f"{len(unreviewed)} candidates have no pre-review yet")
    scope = manual_scope(ready, prereviews, review_cfg)
    missing = [cid for cid in scope if cid not in decisions]
    if missing:
        raise ValueError(f"{len(missing)} candidates in the human review scope are undecided")

    kept: list[dict[str, Any]] = []
    rules: list[RuleSummary] = []
    without_first = 0
    for rule in review_cfg.rules:
        mine = [c for c in ready if c.group in rule.groups]
        spot_first: Counter[str] = Counter()
        spot: list[tuple[ReviewLabel, ReviewLabel]] = []
        human: list[tuple[ReviewLabel, ReviewLabel]] = []
        changed: Counter[str] = Counter()
        sources: Counter[str] = Counter()
        drop_reasons: Counter[str] = Counter()
        modified = dropped = 0
        for c in mine:
            pre = prereviews.get(c.candidate_id)
            if c.candidate_id not in scope:
                sources["prereview" if rule.prereview else "auto"] += 1
                kept.append(
                    question_record(
                        cfg, c, None, "ready", "prereview" if rule.prereview else "auto"
                    )
                )
                continue
            d = decisions[c.candidate_id]
            f = first.get(c.candidate_id)
            if f is None:
                without_first += 1
            else:
                if f.label != d.label:
                    changed[f"{f.label} -> {d.label}"] += 1
                if scope[c.candidate_id] == "spot_check":
                    spot_first[f.label] += 1
                if rule.prereview and pre is not None and pre.decision is not None:
                    human.append((f.label, pre.decision))
                    if scope[c.candidate_id] == "spot_check":
                        spot.append((f.label, pre.decision))
            if d.decision == "drop":
                dropped += 1
                drop_reasons.update(d.reasons or ["other"])
                continue
            modified += d.label == "modify"
            sources["human"] += 1
            kept.append(question_record(cfg, c, d, "ready", "human"))
        valid = [
            p
            for c in mine
            if (p := prereviews.get(c.candidate_id)) is not None and p.decision is not None
        ]
        rules.append(
            RuleSummary(
                groups=rule.groups,
                prereview=rule.prereview,
                ready=len(mine),
                prereview_prompt_versions=dict(
                    sorted(Counter(f"v{p.prompt_version}" for p in valid).items())
                ),
                prereview_decisions=dict(Counter(p.decision for p in valid if p.decision)),
                prereview_reasons=dict(Counter(r for p in valid for r in p.reasons)),
                scope=dict(Counter(scope[c.candidate_id] for c in mine if c.candidate_id in scope)),
                spot_check_n=rule.spot_check_n,
                spot_check_first=dict(spot_first),
                agreement_spot_check=agreement(spot) if rule.prereview else None,
                agreement_all_human=agreement(human) if rule.prereview else None,
                changed_after_first=dict(changed.most_common()),
                kept=sum(sources.values()),
                kept_by_source=dict(sources),
                kept_modified=modified,
                dropped=dropped,
                drop_reasons=dict(drop_reasons.most_common()),
            )
        )
    memorized = [
        question_record(cfg, c, None, st[c.candidate_id])
        for c in candidates
        if st[c.candidate_id] in ("closed_book", "partial_memory")
    ]
    kept.sort(key=lambda q: q["id"])
    memorized.sort(key=lambda q: q["id"])
    summary = ReviewSummary(
        prereview_model=review_cfg.llm.model if review_cfg.llm else None,
        spot_check_seed=review_cfg.spot_check_seed,
        rules=rules,
        without_first_judgment=without_first,
        kept=len(kept),
        kept_by_group=dict(Counter(q["question_type"] for q in kept)),
        rejected=sum(s == "rejected" for s in st.values()),
        memorized=len(memorized),
        memorized_by_group=dict(Counter(q["question_type"] for q in memorized)),
    )
    return kept, memorized, summary


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


LABELS = {"keep": "保留", "modify": "修改", "drop": "丢弃"}
SCOPE_LABELS = {
    "prereview_modify": "预审判为修改",
    "prereview_drop": "预审判为丢弃",
    "prereview_error": "预审失败",
    "spot_check": "随机抽检",
}


def _counts(counts: dict[str, int], names: dict[str, str]) -> str:
    return "，".join(f"{names.get(k, k)} {v}" for k, v in counts.items()) or "—"


def summary_markdown(cfg: EvalsetConfig, s: ReviewSummary) -> str:
    reasons = review_config(cfg).reasons
    rows = ["| 题型 | 项目 | 结果 |", "| --- | --- | --- |"]
    for r in s.rules:
        name = "、".join(cfg.groups[g] for g in r.groups)

        def row(item: str, value: str, name: str = name) -> None:
            rows.append(f"| {name} | {item} | {value} |")

        if r.prereview:
            versions = _counts(r.prereview_prompt_versions, {})
            row("预审", f"`{s.prereview_model}`，提示词 {versions}")
            row("预审结论", _counts(r.prereview_decisions, LABELS))
            row("预审给出的原因", _counts(r.prereview_reasons, reasons))
        scope = _counts(r.scope, SCOPE_LABELS)
        row("人工复核范围", f"{scope}（抽检 {r.spot_check_n} 题，种子 {s.spot_check_seed}）")
        if r.prereview and r.agreement_spot_check and r.agreement_all_human:
            for item, a in (
                ("抽检一致率", r.agreement_spot_check),
                ("全部人工复核题的一致率", r.agreement_all_human),
            ):
                value = f"三分类 {_pct(a.exact)}；保留与丢弃 {_pct(a.usable)}"
                row(f"{item}（按第一次判断，n={a.n}）", value)
        else:
            row("抽检结果（按第一次判断）", _counts(r.spot_check_first, LABELS))
        changed = "，".join(
            "{} → {} {}".format(*(LABELS[x] for x in k.split(" -> ")), v)
            for k, v in r.changed_after_first.items()
        )
        n_changed = sum(r.changed_after_first.values())
        row("第一次判断之后改判", f"{n_changed} 题（{changed}）" if changed else "0 题")
        sources = _counts(
            r.kept_by_source, {"human": "人工决定", "prereview": "采用预审", "auto": "未复核"}
        )
        row("最终保留", f"{r.kept} / {r.ready} 题（{sources}；其中修改过 {r.kept_modified}）")
        row("丢弃", f"{r.dropped} 题：{_counts(r.drop_reasons, reasons)}")
    total = _counts(s.kept_by_group, cfg.groups)
    rows.append(f"| 合计 | 最终保留 | {s.kept} 题（{total}） |")
    memorized = _counts(s.memorized_by_group, cfg.groups)
    rows.append(f"| 合计 | 闭卷可答（`memorized.json`） | {s.memorized} 题（{memorized}） |")
    if s.without_first_judgment:
        rows.append(f"| 合计 | 没有第一次判断（不计入统计） | {s.without_first_judgment} 题 |")
    return "\n".join(rows)


def replace_block(text: str, block: str) -> str:
    pattern = re.compile(re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"marker {MARKER_START} not found")
    return pattern.sub(lambda _: f"{MARKER_START}\n{block}\n{MARKER_END}", text, count=1)


def finalize(cfg: EvalsetConfig) -> ReviewSummary:
    """Write questions.json, memorized.json, review_summary.json and the summary blocks."""
    kept, memorized, summary = build_final(cfg)
    dump = json.dumps
    _atomic_write(cfg.out / QUESTIONS_FILE, dump(kept, ensure_ascii=False, indent=1) + "\n")
    _atomic_write(cfg.out / MEMORIZED_FILE, dump(memorized, ensure_ascii=False, indent=1) + "\n")
    _atomic_write(cfg.out / SUMMARY_FILE, summary.model_dump_json(indent=1) + "\n")
    block = summary_markdown(cfg, summary)
    docs = [cfg.out / "README.md", *(Path(p) for p in review_config(cfg).summary_docs)]
    for path in docs:
        if path.exists():
            _atomic_write(path, replace_block(path.read_text(encoding="utf-8"), block))
    return summary


def human_scope(cfg: EvalsetConfig) -> tuple[dict[str, str], dict[str, Status]]:
    """(human review scope, status per candidate); the scope is empty without a review."""
    candidates = load_candidates(cfg)
    st = statuses(cfg, candidates)
    if cfg.review is None:
        return {}, st
    ready = [c for c in candidates if st[c.candidate_id] == "ready"]
    return manual_scope(ready, load_prereviews(cfg), cfg.review), st
