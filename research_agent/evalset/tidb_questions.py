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
from typing import Literal

from pydantic import BaseModel, Field

from research_agent.evalset.tidb_facts import COMPONENT_LABELS, ParamFact, ReleaseChange

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
CHANGE_TEMPLATES = {
    "修改": "从 v8.1 升级到最新的 v8.5 版本后，{label} `{name}` 的默认值发生了什么变化？",
    "新增": "{label} `{name}` 是在 TiDB 哪个版本新增的？",
    "废弃": "{label} `{name}` 是在 TiDB 哪个版本被废弃的？",
    "删除": "{label} `{name}` 是在 TiDB 哪个版本被删除的？",
}


class DraftQuestion(BaseModel):
    """A rule-made question before it becomes an evalset candidate."""

    kind: QuestionKind
    question: str
    answer: str
    answer_aliases: list[str] = Field(default_factory=list)
    doc: str
    quote: str  # evidence, verbatim from the normalized document
    rule: str  # which template / field produced it
    fact: dict[str, object]  # the extracted fact, for the review page


def clean_value(value: str) -> str | None:
    """The current value as a short answer, or None when it cannot be one."""
    main = value.split("。")[0].strip().rstrip("，,；;").replace("`", "").strip()
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
    return answer, [match.group("main").strip(), match.group("alt").strip()]


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


def change_label(change: ReleaseChange) -> str | None:
    if change.table == "system_variable":
        return "TiDB 系统变量"
    return CORE_COMPONENTS.get(change.component or "")


def version_question(change: ReleaseChange) -> DraftQuestion | None:
    label = change_label(change)
    template = CHANGE_TEMPLATES.get(change.change)
    if label is None or template is None:
        return None
    if change.change == "修改":
        if change.old_default is None or change.new_default is None:
            return None  # a change of something other than the default: no single answer
        answer = f"{change.version} 起默认值从 {change.old_default} 改为 {change.new_default}"
        aliases = [f"从 {change.old_default} 改为 {change.new_default}"]
    else:
        answer, aliases = change.version, [change.version.removeprefix("v")]
    return DraftQuestion(
        kind="version",
        question=template.format(label=label, name=change.name),
        answer=answer,
        answer_aliases=aliases,
        doc=change.doc,
        quote=change.quote,
        rule=f"release:{change.change}",
        fact=change.model_dump(),
    )


def unique_changes(changes: list[ReleaseChange]) -> list[ReleaseChange]:
    """Drop names listed more than once for the same change type (several versions)."""
    counts: dict[tuple[str, str, str], int] = {}
    for c in changes:
        key = (c.table, c.name, c.change)
        counts[key] = counts.get(key, 0) + 1
    return [c for c in changes if counts[(c.table, c.name, c.change)] == 1]


def seeded_sample(items: list[DraftQuestion], n: int, seed: int) -> list[DraftQuestion]:
    """``n`` items chosen by a seeded shuffle of the items sorted by question text."""
    ordered = sorted(items, key=lambda q: q.question)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered[:n]
