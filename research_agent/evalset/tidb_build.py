"""Build the stage 4 TiDB evalset: parameter questions (direct / descriptive) and release-notes
questions, from the rules in tidb_facts.py and tidb_questions.py.

    python -m research_agent.evalset.tidb_build configs/evalset/tidb_params_build.yaml draft
    python -m research_agent.evalset.tidb_build configs/evalset/tidb_params_build.yaml estimate
    python -m research_agent.evalset.tidb_build configs/evalset/tidb_params_build.yaml rewrite

``draft`` (no API calls) samples the facts, locates the evidence chunks and, for descriptive
questions, lists the most similar parameters for the pre-review; it writes ``drafts.jsonl``.
``rewrite`` has the rewriter model turn each descriptive draft into a question that does not
name the parameter, checks it by rules (one revision with feedback), and writes the
candidates. ``estimate`` counts the tokens of every API call the evalset still needs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
import yaml
from pydantic import BaseModel, ConfigDict, Field

from research_agent.config import LLMConfig
from research_agent.evalset.dataset import (
    Candidate,
    EvalsetConfig,
    Evidence,
    _atomic_write,
    _read_jsonl,
    load_candidates,
    write_candidates,
)
from research_agent.evalset.tidb_facts import (
    COMPONENT_LABELS,
    FIELDS,
    META_FIELDS,
    PARAM_DOCS,
    Component,
    ParamFact,
    param_facts,
    parse_entries,
    release_changes,
)
from research_agent.evalset.tidb_questions import (
    BASE_VERSION,
    TARGET_VERSION,
    DraftQuestion,
    ParamIndex,
    changed_names,
    locate_chunk,
    param_question,
    seeded_sample,
    stratified_sample,
    unique_changes,
    unique_param_facts,
    version_question,
)
from research_agent.ingest.corpus import CorpusConfig
from research_agent.ingest.loaders import Document
from research_agent.llm.client import AccountError, client_from_config
from research_agent.llm.pricing import PricingTable

DRAFTS_FILE = "drafts.jsonl"
BUILD_REPORT_FILE = "build_report.json"

REWRITE_TEMPLATE = """你在为 TiDB 文档问答评测集写题。下面是一个{label}的名称和文档里对它的说明。\
请写一个自然的中文问题，问这个参数的默认值。

要求：
1. 用说明里的信息描述这个参数的作用，让读过文档的人能确定是哪一个参数；
2. 问题里不能出现参数名 `{name}`，也不能出现它的任何一段（{segments}）；
3. 问题里不能出现默认值；
4. 写明它是{kind}；
5. 只问一个问题，不超过 60 个字。
如果说明太笼统，无法让人确定是哪一个参数，回复 {{"skip": "原因"}}。

参数：{label} `{name}`
说明：
{description}

只回复 JSON：{{"question": "..."}}"""

REVISION_TEMPLATE = """这个问题没有通过检查：{problems}。请按上面的要求重写，只回复 JSON。"""

KINDS: dict[Component, str] = {
    "system_variable": "TiDB 的一个系统变量",
    "tidb_config": "TiDB 配置文件中的一个配置项",
    "tikv_config": "TiKV 配置文件中的一个配置项",
    "pd_config": "PD 配置文件中的一个配置项",
}


class Composition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # (component, field) -> count; component "system_variable" etc., field as in FIELDS.
    direct: dict[str, int]
    descriptive: dict[str, int]  # component -> count (default values)
    # Ready descriptive questions wanted after the closed-book filter (for ``supplement``);
    # more than the 50 needed, as the pre-review and the human review drop some.
    descriptive_ready: dict[str, int] = Field(default_factory=dict)
    # Final descriptive questions per component (``select``), taken from those the pre-review
    # keeps; empty keeps all of them.
    descriptive_final: dict[str, int] = Field(default_factory=dict)
    version: dict[str, int]  # change type(s) joined by "/" -> count


class BuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evalset: str  # the evalset config the candidates belong to
    corpus: str  # corpus config (docs-cn)
    new_ref: str = "origin/release-8.5"
    old_ref: str = "origin/release-8.1"
    base_version: str = BASE_VERSION
    target_version: str = TARGET_VERSION
    seed: int = 2026
    max_per_group: int = 2  # release-notes questions per config section
    min_description_chars: int = 10
    similar_params: int = 5  # listed for the pre-review of descriptive questions
    composition: Composition
    rewriter: LLMConfig
    max_revisions: int = Field(default=1, ge=0)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BuildConfig:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    def load_evalset(self) -> EvalsetConfig:
        return EvalsetConfig.from_yaml(self.evalset)


class Draft(BaseModel):
    """A sampled question with its evidence located; descriptive drafts have no question yet."""

    candidate_id: str
    group: str  # direct / descriptive / version
    draft: DraftQuestion
    evidence: list[Evidence]
    description: str = ""  # descriptive: what the rewriter may use
    review_notes: str = ""


class BuildReport(BaseModel):
    pools: dict[str, int]
    version_drops: dict[str, int]  # why release-notes rows gave no question
    sampled: dict[str, int]
    evidence_missing: int


def _vtuple(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in version.lstrip("v").split("."))


def _description(fact: ParamFact, entries_text: dict[tuple[str, str], list[str]]) -> str:
    """All non-field bullets of the entry (what the parameter does), at most 400 characters."""
    bullets = entries_text.get((fact.component, fact.name), [])
    return "\n".join(f"- {b}" for b in bullets)[:400]


def _entry_descriptions(docs: list[Document]) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    for doc in docs:
        component = PARAM_DOCS.get(doc.path)
        if component is None:
            continue
        for entry in parse_entries(doc):
            keep = []
            for bullet in entry.bullets:
                key = re.split(r"[：:]", bullet, maxsplit=1)[0].strip()
                if key not in FIELDS and key not in META_FIELDS and key != "默认":
                    keep.append(bullet)
            out[(component, entry.name)] = keep
    return out


def _similar_notes(
    targets: list[DraftQuestion],
    pool: list[DraftQuestion],
    descriptions: dict[tuple[str, str], list[str]],
    k: int,
    embed: Any,
) -> list[str]:
    """For each target, the ``k`` parameters whose descriptions are most similar."""
    import numpy as np

    def text(q: DraftQuestion) -> str:
        bullets = descriptions.get((str(q.fact["component"]), str(q.fact["name"])), [])
        return " ".join(bullets)[:400] or str(q.fact["description"])

    vectors = embed([text(q) for q in pool])
    index = {q.source: i for i, q in enumerate(pool)}
    notes = []
    for q in targets:
        sims = vectors @ vectors[index[q.source]]
        order = [i for i in np.argsort(-sims) if pool[i].source != q.source][:k]
        rows = [
            f"- {COMPONENT_LABELS[pool[i].fact['component']]} `{pool[i].fact['name']}`"  # type: ignore[index]
            f"（相似度 {sims[i]:.2f}，默认值 {pool[i].answer}）：{text(pool[i])[:120]}"
            for i in order
        ]
        notes.append("说明最相似的其他参数（判断描述是否唯一指向本参数）：\n" + "\n".join(rows))
    return notes


def _param_pool(new_docs: list[Document]) -> list[DraftQuestion]:
    by_path = {d.path: d for d in new_docs}
    facts = unique_param_facts([f for p in PARAM_DOCS for f in param_facts(by_path[p])])
    return [q for f in facts if (q := param_question(f))]


def _direct_items(pool: list[DraftQuestion], spec: str) -> list[DraftQuestion]:
    """``spec`` is "component:field"; "范围" also takes "取值范围"."""
    component, field = spec.split(":")
    fields = ("范围", "取值范围") if field == "范围" else (field,)
    return [q for q in pool if q.fact["component"] == component and q.fact["field"] in fields]


def _descriptive_items(
    pool: list[DraftQuestion], component: str, min_chars: int
) -> list[DraftQuestion]:
    return [
        q
        for q in pool
        if q.fact["field"] == "默认值"
        and q.fact["component"] == component
        and len(str(q.fact["description"])) >= min_chars
    ]


def _to_draft(
    group: str,
    candidate_id: str,
    q: DraftQuestion,
    chunks: list[dict[str, Any]],
    descriptions: dict[tuple[str, str], list[str]],
) -> Draft:
    refs = [(q.doc, q.quote, q.anchor), *((e.doc, e.quote, e.anchor) for e in q.more_evidence)]
    evidence = []
    for doc, quote, anchor in refs:
        probe = q.model_copy(update={"doc": doc, "quote": quote, "anchor": anchor})
        evidence.append(Evidence(doc=doc, chunk_id=locate_chunk(chunks, probe), quote=quote))
    description = ""
    if group == "descriptive":
        description = _description(ParamFact.model_validate(q.fact), descriptions)
    return Draft(
        candidate_id=candidate_id, group=group, draft=q, evidence=evidence, description=description
    )


def _add_notes(
    cfg: BuildConfig,
    drafts: list[Draft],
    pool: list[DraftQuestion],
    descriptions: dict[tuple[str, str], list[str]],
    embed: Any,
) -> None:
    targets = [d for d in drafts if d.group == "descriptive"]
    if embed is None or not targets:
        return
    defaults = [q for q in pool if q.fact["field"] == "默认值"]
    notes = _similar_notes(
        [d.draft for d in targets], defaults, descriptions, cfg.similar_params, embed
    )
    for d, note in zip(targets, notes):
        d.review_notes = note


def build_drafts(cfg: BuildConfig, embed: Any | None = None) -> tuple[list[Draft], BuildReport]:
    corpus = CorpusConfig.from_yaml(cfg.corpus)
    evalset = cfg.load_evalset()
    _, new_docs, _ = corpus.load(cfg.new_ref)
    _, old_docs, _ = corpus.load(cfg.old_ref)
    chunks = json.loads(Path(evalset.load_run_config().data.chunks_file).read_text())
    pool = _param_pool(new_docs)
    pools: Counter[str] = Counter(f"{q.fact['component']}:{q.fact['field']}" for q in pool)

    direct: list[DraftQuestion] = []
    for spec, n in cfg.composition.direct.items():
        exclude = {q.source for q in direct}
        direct += seeded_sample(_direct_items(pool, spec), n, cfg.seed, exclude=exclude)

    descriptions = _entry_descriptions(new_docs)
    descriptive: list[DraftQuestion] = []
    for component, n in cfg.composition.descriptive.items():
        items = _descriptive_items(pool, component, cfg.min_description_chars)
        exclude = {q.source for q in direct} | {q.source for q in descriptive}
        descriptive += seeded_sample(items, n, cfg.seed, exclude=exclude)

    new, old = ParamIndex(new_docs), ParamIndex(old_docs)
    all_changes = [c for d in new_docs for c in release_changes(d)]
    base, target = _vtuple(cfg.base_version), _vtuple(cfg.target_version)
    # Changes on the base's own line after it (v8.1.1, v8.1.2): v8.1.0 had the older value.
    old_line = changed_names(
        [c for c in all_changes if _vtuple(c.version)[:2] == base[:2] and _vtuple(c.version) > base]
    )
    later = unique_changes(
        [
            c
            for c in all_changes
            if base[:2] < _vtuple(c.version)[:2] and _vtuple(c.version) <= target
        ]
    )
    results = [
        version_question(c, new, old, old_line, cfg.base_version, cfg.target_version) for c in later
    ]
    drops = Counter(f"{c.change}:{why}" for c, (q, why) in zip(later, results) if q is None)
    vpool = [q for q, _ in results if q is not None]
    pools.update(f"version:{q.rule}" for q in vpool)
    version: list[DraftQuestion] = []
    group_counts: dict[str, int] = {}  # one limit per config section across change types
    for spec, n in cfg.composition.version.items():
        rules = {f"release:{t}" for t in spec.split("/")}
        version += stratified_sample(
            [q for q in vpool if q.rule in rules],
            n,
            cfg.seed,
            cfg.max_per_group,
            exclude={q.source for q in version},
            group_counts=group_counts,
        )

    drafts = [
        _to_draft(group, f"{prefix}{i:03d}", q, chunks, descriptions)
        for group, prefix, questions in (
            ("direct", "d", direct),
            ("descriptive", "s", descriptive),
            ("version", "v", version),
        )
        for i, q in enumerate(questions, 1)
    ]
    missing = sum(e.chunk_id is None for d in drafts for e in d.evidence)
    _add_notes(cfg, drafts, pool, descriptions, embed)
    report = BuildReport(
        pools=dict(sorted(pools.items())),
        version_drops=dict(sorted(drops.items())),
        sampled=dict(Counter(d.group for d in drafts)),
        evidence_missing=missing,
    )
    return drafts, report


def write_drafts(cfg: BuildConfig, drafts: list[Draft], report: BuildReport) -> None:
    out = cfg.load_evalset().out
    _atomic_write(out / DRAFTS_FILE, "".join(d.model_dump_json() + "\n" for d in drafts))
    _atomic_write(out / BUILD_REPORT_FILE, report.model_dump_json(indent=1) + "\n")


def load_drafts(cfg: BuildConfig) -> list[Draft]:
    return [Draft.model_validate(r) for r in _read_jsonl(cfg.load_evalset().out / DRAFTS_FILE)]


def forbidden_terms(name: str) -> list[str]:
    """The parameter name and its last key segment (``log.slow-threshold`` -> both)."""
    last = name.rsplit(".", 1)[-1]
    return list(dict.fromkeys([name, last]))


def rewrite_failures(question: str, draft: Draft) -> list[str]:
    problems = []
    for term in forbidden_terms(str(draft.draft.fact["name"])):
        if term.lower() in question.lower():
            problems.append(f"问题里出现了参数名 `{term}`")
    for value in [draft.draft.answer, *draft.draft.answer_aliases]:
        pattern = rf"(?<![\w.]){re.escape(value)}(?![\w.])"
        if value and re.search(pattern, question, re.IGNORECASE):
            problems.append(f"问题里出现了默认值 {value}")
            break
    if len(question) > 80:
        problems.append("问题超过 80 个字")
    if "？" not in question and "?" not in question:
        problems.append("不是一个问句")
    return problems


def rewrite_prompt(draft: Draft) -> str:
    fact = draft.draft.fact
    component: Component = fact["component"]  # type: ignore[assignment]
    name = str(fact["name"])
    return REWRITE_TEMPLATE.format(
        label=COMPONENT_LABELS[component],
        name=name,
        segments="、".join(f"`{t}`" for t in forbidden_terms(name)),
        kind=KINDS[component],
        description=draft.description or str(fact["description"]),
    )


def _parse(reply: str) -> dict[str, Any]:
    start, end = reply.find("{"), reply.rfind("}")
    try:
        data = json.loads(reply[start : end + 1]) if start >= 0 else {}
    except json.JSONDecodeError:
        data = {}
    return data if isinstance(data, dict) else {}


def rewrite_one(cfg: BuildConfig, pricing: PricingTable, draft: Draft) -> Candidate:
    """Rewriter call(s) for one descriptive draft; a failure becomes a rejected candidate."""
    client = client_from_config(cfg.rewriter, pricing)
    messages: list[dict[str, Any]] = [{"role": "user", "content": rewrite_prompt(draft)}]
    question, reject = "", None
    history: list[dict[str, Any]] = []
    try:
        for attempt in range(cfg.max_revisions + 1):
            result = client.chat(messages=messages)
            reply = str(result["message"].get("content") or "")
            data = _parse(reply)
            if "skip" in data:
                question, reject = "", f"rewriter: {data['skip']}"
                break
            question = str(data.get("question") or "").strip()
            problems = rewrite_failures(question, draft) if question else ["没有给出问题"]
            history.append({"question": question, "problems": problems})
            if not problems:
                reject = None
                break
            reject = "rules: " + "；".join(problems)
            if attempt < cfg.max_revisions:
                messages += [
                    {"role": "assistant", "content": reply},
                    {
                        "role": "user",
                        "content": REVISION_TEMPLATE.format(problems="；".join(problems)),
                    },
                ]
    except AccountError:
        raise
    except (requests.RequestException, KeyError) as e:
        reject = f"api error: {type(e).__name__}: {e}"[:300]
    return _candidate(
        draft,
        question or "（未生成）",
        reject,
        history,
        round(sum(r.cost for r in client.records), 6),
    )


def _candidate(
    draft: Draft,
    question: str,
    reject: str | None = None,
    history: list[dict[str, Any]] | None = None,
    cost: float = 0.0,
) -> Candidate:
    q = draft.draft
    origin: dict[str, Any] = {"rule": q.rule, "fact": q.fact, "template_question": q.question}
    if history is not None:
        origin["rewrites"] = history
    return Candidate(
        candidate_id=draft.candidate_id,
        group=draft.group,
        question=question,
        answer=q.answer,
        answer_aliases=q.answer_aliases,
        evidence=draft.evidence,
        origin=origin,
        reject_reason=reject
        or (None if all(e.chunk_id for e in draft.evidence) else "evidence chunk not found"),
        cost=cost,
        review_notes=draft.review_notes,
    )


def build_candidates(cfg: BuildConfig, workers: int = 5) -> list[Candidate]:
    """Rule questions as they are, descriptive drafts rewritten (API calls)."""
    evalset = cfg.load_evalset()
    pricing = PricingTable.from_yaml(evalset.load_run_config().pricing_file)
    drafts = load_drafts(cfg)
    rule = [_candidate(d, d.draft.question) for d in drafts if d.group != "descriptive"]
    todo = [d for d in drafts if d.group == "descriptive"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rewritten = list(pool.map(lambda d: rewrite_one(cfg, pricing, d), todo))
    candidates = rule + rewritten
    write_candidates(evalset, candidates)
    return candidates


SUPPLEMENT_MARGIN = 1.2  # draw this much more than the expected need
MIN_READY_RATE = 0.25  # a cell's observed ready rate is not taken below this


def _cell(d: Draft) -> str:
    """Sampling cell of a parameter draft: "component:field" (direct) or component."""
    fact = d.draft.fact
    if d.group == "direct":
        field = "范围" if fact["field"] in ("范围", "取值范围") else fact["field"]
        return f"{fact['component']}:{field}"
    return str(fact["component"])


def supplement_plan(
    cfg: BuildConfig, drafts: list[Draft], status: dict[str, str]
) -> dict[tuple[str, str], int]:
    """(group, cell) -> drafts to add so that, at the observed ready rate of the cell, its
    ready questions reach the target (direct: the composition; descriptive: descriptive_ready).
    """
    plan: dict[tuple[str, str], int] = {}
    targets = [
        ("direct", cfg.composition.direct),
        ("descriptive", cfg.composition.descriptive_ready),
    ]
    for group, cells in targets:
        for cell, target in cells.items():
            mine = [d for d in drafts if d.group == group and _cell(d) == cell]
            ready = sum(status.get(d.candidate_id) == "ready" for d in mine)
            need = target - ready
            if need <= 0 or not mine:
                continue
            rate = max(ready / len(mine), MIN_READY_RATE)
            plan[(group, cell)] = math.ceil(need / rate * SUPPLEMENT_MARGIN)
    return plan


def supplement(
    cfg: BuildConfig, status: dict[str, str], embed: Any | None, workers: int = 5
) -> list[Candidate]:
    """Draw more parameter drafts where the closed-book filter left a cell short, in the same
    seeded order as the first draw (the next unused parameters), rewrite the descriptive ones
    and append everything to drafts.jsonl and candidates.jsonl (API calls for the rewrite)."""
    corpus = CorpusConfig.from_yaml(cfg.corpus)
    evalset = cfg.load_evalset()
    _, new_docs, _ = corpus.load(cfg.new_ref)
    chunks = json.loads(Path(evalset.load_run_config().data.chunks_file).read_text())
    pool = _param_pool(new_docs)
    descriptions = _entry_descriptions(new_docs)
    drafts = load_drafts(cfg)
    used = {d.draft.source for d in drafts}
    counters = Counter(d.candidate_id[0] for d in drafts)
    added: list[Draft] = []
    for (group, cell), n in supplement_plan(cfg, drafts, status).items():
        if group == "direct":
            items = _direct_items(pool, cell)
        else:
            items = _descriptive_items(pool, cell, cfg.min_description_chars)
        prefix = "d" if group == "direct" else "s"
        for q in seeded_sample(items, n, cfg.seed, exclude=used):
            used.add(q.source)
            counters[prefix] += 1
            cid = f"{prefix}{counters[prefix]:03d}"
            added.append(_to_draft(group, cid, q, chunks, descriptions))
    _add_notes(cfg, added, pool, descriptions, embed)
    out = evalset.out
    _atomic_write(out / DRAFTS_FILE, "".join(d.model_dump_json() + "\n" for d in drafts + added))
    pricing = PricingTable.from_yaml(evalset.load_run_config().pricing_file)
    todo = [d for d in added if d.group == "descriptive"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rewritten = list(executor.map(lambda d: rewrite_one(cfg, pricing, d), todo))
    new = [_candidate(d, d.draft.question) for d in added if d.group != "descriptive"] + rewritten
    write_candidates(evalset, load_candidates(evalset) + new)
    return new


def select(cfg: BuildConfig) -> dict[str, int]:
    """Leave out ready candidates beyond each cell's quota (in draw order) as spares.

    Direct cells use the composition; descriptive cells use ``descriptive_final`` and count
    only the candidates the pre-review keeps. Version questions are all kept. Spares get a
    ``spare:`` reject reason (finalize counts them apart); a rerun starts from scratch.
    Returns spares per group.
    """
    from research_agent.evalset.dataset import statuses
    from research_agent.evalset.review import SPARE, auto_decision, load_prereviews

    evalset = cfg.load_evalset()
    candidates = [
        c.model_copy(update={"reject_reason": None})
        if (c.reject_reason or "").startswith(SPARE)
        else c
        for c in load_candidates(evalset)
    ]
    write_candidates(evalset, candidates)
    status = statuses(evalset, candidates)
    prereviews = load_prereviews(evalset)
    drafts = {d.candidate_id: d for d in load_drafts(cfg)}
    quotas = {"direct": cfg.composition.direct, "descriptive": cfg.composition.descriptive_final}
    used: Counter[tuple[str, str]] = Counter()
    spares: Counter[str] = Counter()
    out = []

    def order(c: Candidate) -> tuple[str, int]:
        return c.candidate_id[0], int(c.candidate_id[1:])

    for c in sorted(candidates, key=order):
        quota = quotas.get(c.group)
        if quota and status[c.candidate_id] == "ready":
            counts = c.group != "descriptive" or (
                auto_decision(c, prereviews.get(c.candidate_id)).decision == "keep"
            )
            cell = _cell(drafts[c.candidate_id])
            if counts and used[(c.group, cell)] >= quota.get(cell, 0):
                reason = f"{SPARE} {cell} already has {quota.get(cell, 0)}"
                c = c.model_copy(update={"reject_reason": reason})
                spares[c.group] += 1
            elif counts:
                used[(c.group, cell)] += 1
        out.append(c)
    write_candidates(evalset, out)
    return dict(spares)


class CallEstimate(BaseModel):
    step: str
    model: str
    calls: int  # upper bound
    input_tokens: int
    output_tokens: int
    cost: float


def estimate(cfg: BuildConfig, count_tokens: Any, overhead: float = 1.15) -> list[CallEstimate]:
    """Upper-bound cost of the API calls still to make: rewrite, closed book, pre-review."""
    from research_agent.evalset.prompts import CLOSED_BOOK_TEMPLATE, PREREVIEW_TEMPLATE

    evalset = cfg.load_evalset()
    run = evalset.load_run_config()
    pricing = PricingTable.from_yaml(run.pricing_file)
    drafts = load_drafts(cfg)
    out: list[CallEstimate] = []

    def add(step: str, model: str, prompts: list[str], output: int, repeat: int = 1) -> None:
        tokens = [round(count_tokens(p) * overhead) for p in prompts] * repeat
        price = pricing.get(model)
        out.append(
            CallEstimate(
                step=step,
                model=pricing.resolve(model),
                calls=len(tokens),
                input_tokens=sum(tokens),
                output_tokens=output * len(tokens),
                cost=round(sum(price.cost(n, output, 0) for n in tokens), 4),
            )
        )

    descriptive = [d for d in drafts if d.group == "descriptive"]
    # Rewrite: every draft once, plus one revision each (the conversation grows by ~100).
    prompts = [rewrite_prompt(d) for d in descriptive]
    add("改写（含一次修订）", cfg.rewriter.model, prompts + [p + " " * 400 for p in prompts], 80)
    # Closed book: every candidate at temperature 0, plus 3 samples when wrong (upper bound).
    questions = [d.draft.question for d in drafts]
    cb = [CLOSED_BOOK_TEMPLATE.format(subject=evalset.subject, question=q) for q in questions]
    add(
        "闭卷作答（上限：每题 4 次）", run.llm.model, cb, 60, repeat=1 + evalset.closed_book.samples
    )
    judge = [
        "You are an expert evaluator. Please evaluate if the generated answer is correct by "
        "comparing it with the gold answer. Generated answer: xxxxxxxxxx Gold answer: "
        f"{d.draft.answer} The generated answer should be considered correct if it: 1. Contains "
        "the key information from the gold answer 2. Is factually accurate and consistent with "
        "the gold answer 3. Does not contain any contradicting information Respond with ONLY "
        "'correct' or 'incorrect'. Response:"
        for d in drafts
    ]
    add(
        "闭卷判分（上限：每题 4 次）",
        run.judge.model,
        judge,
        3,
        repeat=1 + evalset.closed_book.samples,
    )
    # Pre-review of the descriptive questions: criteria, evidence chunks, similar parameters.
    review = evalset.review
    if review is not None and review.llm is not None:
        chunks = {c["id"]: c["text"] for c in json.loads(Path(run.data.chunks_file).read_text())}
        prompts = [
            PREREVIEW_TEMPLATE.format(
                subject=evalset.subject,
                criteria=review.criteria,
                reasons=", ".join(review.reasons),
                group="描述问法",
                question="x" * 40,
                answer=d.draft.answer,
                aliases="",
                passages="\n".join(chunks.get(e.chunk_id or "", e.quote) for e in d.evidence),
                others="\n".join("- " + "x" * 40 for _ in range(5)),
                notes=d.review_notes,
            )
            for d in descriptive
        ]
        add("描述问法预审", review.llm.model, prompts, 200)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m research_agent.evalset.tidb_build")
    parser.add_argument("config")
    parser.add_argument("step", choices=["draft", "estimate", "rewrite", "supplement", "select"])
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    cfg = BuildConfig.from_yaml(args.config)
    if args.step == "draft":
        from research_agent.ingest.index import SentenceTransformerEmbedder

        corpus = CorpusConfig.from_yaml(cfg.corpus)
        model = SentenceTransformerEmbedder(
            corpus.embedding.model, corpus.embedding.device, corpus.embedding.batch_size
        )
        drafts, report = build_drafts(cfg, model.encode)
        write_drafts(cfg, drafts, report)
        print(report.model_dump_json(indent=1))
    elif args.step == "estimate":
        from research_agent.ingest.chunker import tiktoken_counter

        rows = estimate(cfg, tiktoken_counter())
        for r in rows:
            print(r.model_dump_json())
        print(f"total {sum(r.cost for r in rows):.4f}")
    elif args.step == "select":
        print(f"spares: {select(cfg)}")
    elif args.step == "supplement":
        from research_agent.evalset.dataset import statuses
        from research_agent.ingest.index import SentenceTransformerEmbedder
        from research_agent.upstream import load_env

        load_env()
        evalset = cfg.load_evalset()
        status: dict[str, str] = dict(statuses(evalset, load_candidates(evalset)))
        corpus = CorpusConfig.from_yaml(cfg.corpus)
        model = SentenceTransformerEmbedder(
            corpus.embedding.model, corpus.embedding.device, corpus.embedding.batch_size
        )
        new = supplement(cfg, status, model.encode, args.workers)
        by_group = dict(Counter(c.group for c in new))
        n_rejected = sum(bool(c.reject_reason) for c in new)
        new_cost = sum(c.cost for c in new)
        print(f"{len(new)} new: {by_group}, rejected {n_rejected}, rewrite cost {new_cost:.4f}")
    else:
        from research_agent.upstream import load_env

        load_env()
        candidates = build_candidates(cfg, args.workers)
        rejected = Counter(c.group for c in candidates if c.reject_reason)
        cost = sum(c.cost for c in candidates)
        print(f"{len(candidates)} candidates, rejected {dict(rejected)}, rewrite cost {cost:.4f}")


if __name__ == "__main__":
    main()
