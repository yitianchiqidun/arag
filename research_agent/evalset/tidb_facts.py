"""Structured facts extracted by rules from the TiDB docs, the source of the stage 4 answers.

Two sources, both read from the normalized documents (the text the index holds, so every
evidence quote can be found in a chunk):

- **Parameter references**: ``system-variables.md`` (``### `name```` headings, ``- 字段：值``
  bullets) and the TiDB / TiKV / PD configuration files (``## section`` / ``### `item```,
  ``+ 字段：值`` bullets). One :class:`ParamFact` per entry and field (默认值, 作用域, …).
- **Release notes** v8.2.0 - v8.5.x: the rows of the "兼容性变更 > 系统变量 / 配置参数"
  tables (name, change type, description). One :class:`ReleaseChange` per row; for "修改"
  rows the old and new default values are parsed from "默认值从 `A` 修改为 `B`".
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from research_agent.ingest.loaders import Document
from research_agent.ingest.markdown import fence_update

Component = Literal["system_variable", "tidb_config", "tikv_config", "pd_config"]
PARAM_DOCS: dict[str, Component] = {
    "system-variables.md": "system_variable",
    "tidb-configuration-file.md": "tidb_config",
    "tikv-configuration-file.md": "tikv_config",
    "pd-configuration-file.md": "pd_config",
}
COMPONENT_LABELS: dict[Component, str] = {
    "system_variable": "TiDB 系统变量",
    "tidb_config": "TiDB 配置项",
    "tikv_config": "TiKV 配置项",
    "pd_config": "PD 配置项",
}
# Bullet keys that hold a fact; the first other bullet of an entry is its description.
FIELDS = ("默认值", "作用域", "类型", "范围", "取值范围", "可选值", "单位", "最大值", "最小值")
META_FIELDS = ("是否持久化到集群", "是否受 Hint SET_VAR 控制")

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_BULLET = re.compile(r"^[-+*][ \t]+(.*)$")
_FIELD = re.compile(r"^(?P<key>[^：:\n]{1,24})[：:][ \t]*(?P<value>.*)$")
_NAME = re.compile(r"`([^`]+)`")
_SINCE = re.compile(r"从 (v[\d.]+) 版本开始引入")
_RELEASE_DOC = re.compile(r"releases/release-(\d+\.\d+\.\d+)\.md$")
# "默认值从 `A` 修改为 `B`"; a value in backticks may contain spaces ("MAX(1, MIN(4, ...))").
_VALUE = r"(?:`(?P<{0}q>[^`]+)`|(?P<{0}>[^`，。；\s]+))"
_DEFAULT_CHANGE = re.compile(
    r"默认值(?:由原来的|从)\s*"
    + _VALUE.format("old")
    + r"\s*(?:修改|变更|改)为\s*"
    + _VALUE.format("new")
)


class Entry(BaseModel):
    """One parameter's section of a reference page."""

    doc: str
    path: list[str]  # heading path, version marks included
    name: str  # system variable name, or the full configuration key ("log.file.max-size")
    since: str | None = None  # "v6.4.0" from "从 v6.4.0 版本开始引入"
    bullets: list[str]  # top-level bullets, without the marker


class ParamFact(BaseModel):
    component: Component
    name: str
    field: str  # one of FIELDS
    value: str  # as written after the colon
    quote: str  # the bullet line, verbatim from the normalized document
    description: str  # the entry's first non-field bullet (what the parameter does)
    since: str | None
    doc: str
    path: list[str]


class ReleaseChange(BaseModel):
    version: str  # "v8.5.0"
    doc: str
    table: Literal["system_variable", "config"]
    component: str | None  # first column of the config table ("TiKV", "PD", "TiDB", ...)
    name: str
    change: str  # 新增 / 修改 / 废弃 / 删除
    description: str
    quote: str  # the table row, verbatim
    old_default: str | None = None
    new_default: str | None = None


def _clean(text: str) -> str:
    return text.strip().strip("`").strip()


def _section_name(heading: str) -> str:
    """``log.file 从 v5.4.0 版本开始引入`` -> ``log.file``; ``label-property（已废弃）`` keeps
    the marker so such sections can be skipped."""
    return _SINCE.sub("", heading).strip()


def parse_entries(doc: Document) -> list[Entry]:
    """Parameter entries: headings that name a parameter in backticks, with their bullets.

    Configuration keys are prefixed with the enclosing section headings that are not
    themselves parameters (``## log.file`` + ``### `max-size``` -> ``log.file.max-size``).
    """
    entries: list[Entry] = []
    stack: list[tuple[int, str]] = []
    current: Entry | None = None
    fence: str | None = None
    is_sysvar = PARAM_DOCS.get(doc.path) == "system_variable"
    for line in doc.text.splitlines():
        heading = _HEADING.match(line) if fence is None else None
        fence = fence_update(line, fence)
        if heading:
            level, text = len(heading.group(1)), heading.group(2)
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, text)]
            name = _NAME.match(text)
            current = None
            if name and level >= 2:
                if is_sysvar:
                    key = name.group(1)
                else:
                    sections = [
                        _section_name(t)
                        for lv, t in stack[:-1]
                        if lv >= 2 and not _NAME.match(t) and t != "全局配置"
                    ]
                    key = ".".join([*sections, name.group(1)])
                since = _SINCE.search(text)
                current = Entry(
                    doc=doc.path,
                    path=[t for _, t in stack],
                    name=key,
                    since=since.group(1) if since else None,
                    bullets=[],
                )
                entries.append(current)
            continue
        bullet = _BULLET.match(line) if fence is None else None
        if current is not None and bullet:
            current.bullets.append(bullet.group(1).strip())
    return entries


def param_facts(doc: Document) -> list[ParamFact]:
    component = PARAM_DOCS[doc.path]
    facts: list[ParamFact] = []
    for entry in parse_entries(doc):
        fields: list[tuple[str, str, str]] = []
        description = ""
        for bullet in entry.bullets:
            match = _FIELD.match(bullet)
            key = match.group("key").strip() if match else ""
            if match and key in FIELDS:
                marker = "-" if component == "system_variable" else "+"
                fields.append((key, match.group("value").strip(), f"{marker} {bullet}"))
            elif not description and not (match and key in META_FIELDS):
                description = bullet
        facts.extend(
            ParamFact(
                component=component,
                name=entry.name,
                field=key,
                value=value,
                quote=quote,
                description=description,
                since=entry.since,
                doc=entry.doc,
                path=entry.path,
            )
            for key, value, quote in fields
        )
    return facts


def _cells(row: str) -> list[str]:
    """Cells of a Markdown table row; ``\\|`` inside a cell is a literal pipe."""
    inner = row.strip().removeprefix("|").removesuffix("|")
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", inner)]


def release_changes(doc: Document) -> list[ReleaseChange]:
    """Rows of the system variable and configuration tables under "兼容性变更"."""
    match = _RELEASE_DOC.search(doc.path)
    if match is None:
        return []
    version = f"v{match.group(1)}"
    changes: list[ReleaseChange] = []
    section: str | None = None
    in_compat = False
    lines = doc.text.splitlines()
    for i, line in enumerate(lines):
        heading = _HEADING.match(line)
        if heading:
            level, text = len(heading.group(1)), heading.group(2)
            if level == 2:
                in_compat = text.startswith("兼容性变更")
                section = None
            elif level == 3 and in_compat:
                section = text
            continue
        if not in_compat or section is None or not line.lstrip().startswith("|"):
            continue
        if i + 1 < len(lines) and re.match(r"^\|?[ \t:]*-{3,}", lines[i + 1]):
            continue  # header row
        if re.match(r"^\|?[ \t:]*-{3,}", line):
            continue
        cells = _cells(line)
        if section == "系统变量" and len(cells) >= 3:
            table: Literal["system_variable", "config"] = "system_variable"
            component, name_cell, change, description = None, cells[0], cells[1], cells[2]
        elif section in ("配置参数", "配置文件参数") and len(cells) >= 4:
            table = "config"
            component, name_cell, change, description = cells[0], cells[1], cells[2], cells[3]
        else:
            continue
        # A cell may list several names ("`a`；`b`"): one change per name, same row.
        names = _NAME.findall(name_cell) or [_clean(name_cell)]
        default = _DEFAULT_CHANGE.search(description)
        old = (default.group("oldq") or default.group("old")) if default else None
        new = (default.group("newq") or default.group("new")) if default else None
        changes.extend(
            ReleaseChange(
                version=version,
                doc=doc.path,
                table=table,
                component=component,
                name=name,
                change=change,
                description=description,
                quote=line.strip(),
                old_default=old.strip() if old else None,
                new_default=new.strip() if new else None,
            )
            for name in names
        )
    return changes
