"""Rules that turn the extracted TiDB facts into candidate questions (stage 4).

Parameter questions ("direct": the parameter is named) and release-notes questions
("version"). Answers are always taken from the document text; only the question wording comes
from a template. Facts are filtered so that each question has exactly one short answer:

- parameters: no internal / test-only / deprecated entries, no deprecated config sections,
  no entries shared by several config sections (``rocksdb.defaultcf | rocksdb.writecf``) or
  keys that occur twice, and a value that states the current setting only (no version
  history, no reference to another parameter, at most 40 characters);
- release notes: TiDB, TiKV and PD only (the subset leaves out TiFlash, TiCDC, BR, ...);
  "修改" rows only when the description states the old and new default; names that appear
  in several rows of the same kind are skipped (the answer would be ambiguous).

Sampling is seeded and ordered by parameter name, so the same documents give the same set.
"""

from __future__ import annotations

import random
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from research_agent.evalset.tidb_facts import (
    COMPONENT_LABELS,
    PARAM_DOCS,
    Component,
    Entry,
    ParamFact,
    ReleaseChange,
    parse_entries,
)
from research_agent.ingest.loaders import Document

QuestionKind = Literal["direct", "version"]

_SKIP_ENTRY = re.compile(r"内部测试|内部使用|已废弃|不再生效|不推荐使用|已被移除|已删除")
_VERSIONED = re.compile(r"v\d|版本")
_PAREN = re.compile(r"^(?P<main>.+?)\s*[（(](?P<alt>[^()（）]+)[)）]$")
CORE_COMPONENTS = {"TiDB": "TiDB 配置项", "TiKV": "TiKV 配置项", "PD": "PD 配置项"}
FIELD_TEMPLATES = {
    "默认值": "{label} `{name}` 的默认值是多少？",
    "作用域": "{label} `{name}` 的作用域是什么？",
    "取值范围": "{label} `{name}` 的取值范围是什么？",
    "范围": "{label} `{name}` 的取值范围是什么？",
}
BASE_VERSION = "v8.1.0"  # the old version of the comparison
TARGET_VERSION = "v8.5.8"  # newest v8.5 patch in docs-cn release-8.5 and on the tiup mirror
CHANGE_TEMPLATES = {
    # Compares the defaults of two versions: an upgraded cluster may keep its old value.
    "修改": "与 {base} 相比，{target} 中 {label} `{name}` 的默认值有什么变化？",
    "新增": "{label} `{name}` 是在 TiDB 哪个版本新增的？",
    "废弃": "{label} `{name}` 是在 TiDB 哪个版本被废弃的？",
    "删除": "{label} `{name}` 是在 TiDB 哪个版本被删除的？",
}
# Several release lines got the change at once ("从 v7.5.7 和 v8.5.4 开始"): ask within v8.
CHANGE_TEMPLATES_V8 = {
    "新增": "在 v8 系列中，{label} `{name}` 是从哪个版本开始引入的？",
    "废弃": "在 v8 系列中，{label} `{name}` 是从哪个版本开始废弃的？",
}
TABLE_COMPONENTS: dict[str, Component] = {
    "TiDB": "tidb_config",
    "TiKV": "tikv_config",
    "PD": "pd_config",
}
_DEPRECATED = re.compile(
    r"从 (v[\d.]+(?:(?:、|和| 和 )v[\d.]+)*) ?(?:版本)?(?:开始)?[，,]?[^。\n]{0,12}废弃"
)
_VERSION = re.compile(r"v\d+\.\d+(?:\.\d+)?")


class EvidenceRef(BaseModel):
    doc: str
    quote: str
    anchor: str | None = None


class DraftQuestion(BaseModel):
    """A rule-made question before it becomes an evalset candidate."""

    kind: QuestionKind
    question: str
    answer: str
    answer_aliases: list[str] = Field(default_factory=list)
    doc: str
    quote: str  # evidence, verbatim from the normalized document
    anchor: str | None = None  # the entry's heading, to find the chunk of a common quote
    source: str  # one question per source: the parameter, or the release-notes row
    more_evidence: list[EvidenceRef] = Field(default_factory=list)  # e.g. the parameter doc
    stratum: str = ""  # sampling stratum (release version and component)
    group: str | None = None  # config section; at most a few questions per section
    rule: str  # which template / field produced it
    fact: dict[str, object]  # the extracted fact, for the review page


def clean_value(value: str) -> str | None:
    """The current value as a short answer, or None when it cannot be one.

    Only the first sentence counts; "`168h`，即 7 天" becomes "168h (7 天)" so the
    explanation is accepted as an alias.
    """
    main = value.split("。")[0].strip().rstrip("，,；;").replace("`", "").strip()
    if "，即" in main:
        head, _, tail = main.partition("，即")
        main = f"{head.strip()} ({tail.strip()})"
    if not main or _VERSIONED.search(main):
        return None
    if len(main) > 40 or main in ('""', "''", "空") or re.search(r"与 .+ 相同|参考|详见", main):
        return None
    return main


def split_aliases(answer: str) -> tuple[str, list[str]]:
    """``1073741824 (1 GiB)`` -> (``1073741824 (1 GiB)``, [``1073741824``, ``1 GiB``])."""
    match = _PAREN.match(answer)
    if not match:
        return answer, []
    alt = match.group("alt").strip().removeprefix("即").strip()
    return answer, [match.group("main").strip(), alt]


def param_question(fact: ParamFact) -> DraftQuestion | None:
    template = FIELD_TEMPLATES.get(fact.field)
    if template is None or fact.field != "默认值" and fact.component != "system_variable":
        return None
    if _SKIP_ENTRY.search(fact.description) or any("已废弃" in p for p in fact.path):
        return None
    if "|" in fact.name:  # an entry shared by several config sections has no single key
        return None
    value = clean_value(fact.value)
    if value is None:
        return None
    if fact.field == "作用域":
        value = value.replace(" | ", " 和 ")
    answer, aliases = split_aliases(value)
    return DraftQuestion(
        kind="direct",
        question=template.format(label=COMPONENT_LABELS[fact.component], name=fact.name),
        answer=answer,
        answer_aliases=aliases,
        doc=fact.doc,
        quote=fact.quote,
        anchor=fact.path[-1],
        source=f"{fact.doc}#{fact.name}",
        rule=f"param:{fact.field}",
        fact=fact.model_dump(),
    )


def unique_param_facts(facts: list[ParamFact]) -> list[ParamFact]:
    """Drop configuration keys that occur in more than one entry (an ambiguous name)."""
    seen: dict[tuple[str, str, str], int] = {}
    for f in facts:
        key = (f.component, f.name, f.field)
        seen[key] = seen.get(key, 0) + 1
    return [f for f in facts if seen[(f.component, f.name, f.field)] == 1]


class ParamIndex:
    """Parameter entries of one docs version, looked up the way release notes name them."""

    def __init__(self, docs: list[Document]) -> None:
        self.entries: dict[Component, list[Entry]] = {}
        for doc in docs:
            component = PARAM_DOCS.get(doc.path)
            if component is not None:
                self.entries[component] = parse_entries(doc)

    def find(self, component: Component, name: str) -> list[Entry]:
        """Entries named ``name``, or whose full key ends with ``.name`` (notes often omit
        the section: ``region-compact-check-interval`` is ``raftstore.region-...``)."""
        entries = self.entries.get(component, [])
        exact = [e for e in entries if e.name == name]
        return exact or [e for e in entries if e.name.endswith("." + name)]

    @staticmethod
    def default(entry: Entry) -> tuple[str, str] | None:
        """(value, bullet line) of the entry's default. "默认值：" followed by a nested list
        (one item per version) gives the nested text as the value."""
        marker = "-" if entry.doc == "system-variables.md" else "+"
        lines = entry.body.splitlines()
        for i, line in enumerate(lines):
            match = re.match(rf"^\{marker}[ \t]+(默认值|默认)[：:][ \t]*(.*)$", line)
            if not match:
                continue
            value = match.group(2).strip()
            if not value:  # the values are in the nested bullets below
                nested = []
                for sub in lines[i + 1 :]:
                    if sub.strip() and not sub.startswith((" ", "\t")):
                        break
                    nested.append(sub.strip().lstrip("+-* ").strip())
                value = " ".join(x for x in nested if x)
            return value, line.strip()
        return None


def change_component(change: ReleaseChange) -> Component | None:
    if change.table == "system_variable":
        return "system_variable"
    return TABLE_COMPONENTS.get(change.component or "")


def change_label(change: ReleaseChange) -> str | None:
    component = change_component(change)
    return COMPONENT_LABELS[component] if component else None


def _same_value(expected: str, documented: str) -> bool:
    """``OFF`` vs ``"OFF"``, ``8192`` vs ``8192。TiDB v7.6.0 之前…``: compare the first sentence
    without quotes, backticks, spaces and case."""

    def norm(text: str) -> str:
        return re.sub(r"[`\"'\s]", "", text.split("。")[0]).lower()

    if expected in ("空", "空字符串", "空值"):  # the docs write an empty default as ``""``
        return norm(documented) in ("", "空", "空字符串")
    return bool(norm(expected)) and norm(expected) in norm(documented)


def _deprecation_versions(entry: Entry) -> tuple[list[str], str | None]:
    """Versions of "从 vX 和 vY 开始，该配置项被废弃" in the entry, with that line."""
    for line in entry.body.splitlines():
        match = _DEPRECATED.search(line)
        if match:
            return _VERSION.findall(match.group(1)), line.strip()
    return [], None


def version_question(
    change: ReleaseChange,
    new: ParamIndex,
    old: ParamIndex,
    changed_on_old_line: set[tuple[str, str]],
    base: str = BASE_VERSION,
    target: str = TARGET_VERSION,
) -> tuple[DraftQuestion | None, str]:
    """The question for one release-notes row, checked against the parameter docs of both
    versions; returns (question, "ok") or (None, why it was dropped)."""
    component = change_component(change)
    if component is None:
        return None, "outside_subset"
    template = CHANGE_TEMPLATES.get(change.change)
    if template is None:
        return None, "unknown_change"
    label = COMPONENT_LABELS[component]
    if change.change == "修改" and (change.old_default is None or change.new_default is None):
        return None, "not_a_default_change"  # no single answer
    entries = new.find(component, change.name)
    if change.change == "删除":
        if entries:
            return None, "still_documented"
        entries = old.find(component, change.name)
    if not entries:
        return None, "not_in_parameter_docs"
    if len(entries) > 1:
        return None, "ambiguous_name"
    entry = entries[0]
    key = entry.name
    more: list[EvidenceRef] = []
    versions = [change.version]
    if change.change == "修改":
        assert change.old_default is not None and change.new_default is not None
        if {(component, key), (component, change.name)} & changed_on_old_line:
            return None, "changed_on_the_v8_1_line"
        now = new.default(entry)
        before = [d for e in old.find(component, change.name) if (d := old.default(e))]
        if now is None or not _same_value(change.new_default, now[0]):
            return None, "new_default_mismatch"
        if len(before) != 1 or not _same_value(change.old_default, before[0][0]):
            return None, "old_default_mismatch"
        more.append(EvidenceRef(doc=entry.doc, quote=now[1], anchor=entry.path[-1]))
        question = template.format(base=base, target=target, label=label, name=key)
        pair = f"从 {change.old_default} 改为 {change.new_default}"
        answer, aliases = f"{pair}（{change.version} 起）", [pair]
    else:
        if change.change == "新增":
            versions, line = entry.since, entry.path[-1]
            if not versions:
                return None, "no_version_mark"
            more.append(EvidenceRef(doc=entry.doc, quote=line, anchor=None))
        elif change.change == "废弃":
            versions, note = _deprecation_versions(entry)
            if not versions or note is None:
                return None, "no_version_mark"
            more.append(EvidenceRef(doc=entry.doc, quote=note, anchor=entry.path[-1]))
        if change.version not in versions:
            return None, "version_mismatch"
        v8 = [v for v in versions if v.startswith("v8.")]
        if len(versions) > 1:
            question = CHANGE_TEMPLATES_V8[change.change].format(label=label, name=key)
        else:
            question = template.format(label=label, name=key)
        answer = "、".join(v8)
        aliases = [*v8, *(v.removeprefix("v") for v in v8)] if len(v8) > 1 else [v8[0][1:]]
    section = key.rsplit(".", 1)[0] if "." in key and component != "system_variable" else None
    return (
        DraftQuestion(
            kind="version",
            question=question,
            answer=answer,
            answer_aliases=aliases,
            doc=change.doc,
            quote=change.quote,
            source=change.quote,
            more_evidence=more,
            stratum=f"{change.version}|{label}",
            group=f"{label}:{section}" if section else None,
            rule=f"release:{change.change}",
            fact={**change.model_dump(), "key": key, "doc_versions": versions},
        ),
        "ok",
    )


def changed_names(changes: list[ReleaseChange]) -> set[tuple[str, str]]:
    """(component, name) of every row, e.g. the changes made on the v8.1 line after v8.1.0."""
    return {(c, ch.name) for ch in changes if (c := change_component(ch))}


def unique_changes(changes: list[ReleaseChange]) -> list[ReleaseChange]:
    """Drop names listed more than once for the same change type (several versions)."""
    counts: dict[tuple[str, str, str], int] = {}
    for c in changes:
        key = (c.table, c.name, c.change)
        counts[key] = counts.get(key, 0) + 1
    return [c for c in changes if counts[(c.table, c.name, c.change)] == 1]


def seeded_sample(
    items: list[DraftQuestion], n: int, seed: int, exclude: set[str] | None = None
) -> list[DraftQuestion]:
    """``n`` items from a seeded shuffle of the items sorted by question text, at most one per
    source (a parameter, a release-notes row) and none from ``exclude``."""
    ordered = sorted(items, key=lambda q: q.question)
    random.Random(seed).shuffle(ordered)
    seen = set(exclude or ())
    picked: list[DraftQuestion] = []
    for q in ordered:
        if q.source not in seen and len(picked) < n:
            picked.append(q)
            seen.add(q.source)
    return picked


def stratified_sample(
    items: list[DraftQuestion],
    n: int,
    seed: int,
    max_per_group: int = 2,
    exclude: set[str] | None = None,
    group_counts: dict[str, int] | None = None,
) -> list[DraftQuestion]:
    """Round-robin over the strata ("version|component"), a seeded shuffle within each.

    The strata are visited version by version (each version's components in a seeded
    order), so a small quota spreads over the versions before a version gets a second
    question. At most one question per source and ``max_per_group`` per config section;
    pass the same ``group_counts`` to several calls to share that limit.
    """
    rng = random.Random(seed)
    strata: dict[str, list[DraftQuestion]] = {}
    for q in sorted(items, key=lambda q: q.question):
        strata.setdefault(q.stratum, []).append(q)
    for key in sorted(strata):
        rng.shuffle(strata[key])
    by_version: dict[str, list[str]] = {}
    for key in sorted(strata):
        by_version.setdefault(key.split("|", 1)[0], []).append(key)
    for keys in by_version.values():
        rng.shuffle(keys)
    order = [
        keys[i]
        for i in range(max((len(k) for k in by_version.values()), default=0))
        for keys in by_version.values()
        if i < len(keys)
    ]
    sources = set(exclude or ())
    groups = group_counts if group_counts is not None else {}
    picked: list[DraftQuestion] = []
    while len(picked) < n and any(strata.values()):
        for key in order:
            while strata[key] and len(picked) < n:
                q = strata[key].pop(0)
                if q.source in sources or (q.group and groups.get(q.group, 0) >= max_per_group):
                    continue
                picked.append(q)
                sources.add(q.source)
                if q.group:
                    groups[q.group] = groups.get(q.group, 0) + 1
                break
    return picked


def locate_chunk(chunks: list[dict[str, Any]], q: DraftQuestion) -> str | None:
    """Id of the chunk that holds the evidence.

    A release-notes row is unique. A parameter bullet such as "- 默认值：`OFF`" is not, so
    the quote must follow the entry's own heading line in the chunk body; if the entry was
    split across chunks, the chunk whose sections include the entry and whose text has the
    quote is taken.
    """
    candidates = [c for c in chunks if c["doc"] == q.doc and q.quote in str(c["text"])]
    if q.anchor is None:
        return str(candidates[0]["id"]) if candidates else None
    heading = re.compile(rf"^#+[ \t]+{re.escape(q.anchor)}[ \t]*$", re.MULTILINE)
    for chunk in candidates:
        text = str(chunk["text"])
        match = heading.search(text)
        if match:
            nxt = text.find("\n#", match.end())
            if q.quote in text[match.end() : nxt if nxt >= 0 else len(text)]:
                return str(chunk["id"])
    for chunk in candidates:
        if any(path and path[-1] == q.anchor for path in chunk.get("sections", [])):
            return str(chunk["id"])
    return None
