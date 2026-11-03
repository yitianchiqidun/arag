from __future__ import annotations

import hashlib

from research_agent.evalset.tidb_facts import param_facts, parse_entries, release_changes
from research_agent.evalset.tidb_questions import (
    DraftQuestion,
    ParamIndex,
    changed_names,
    clean_value,
    locate_chunk,
    param_question,
    split_aliases,
    stratified_sample,
    unique_changes,
    unique_param_facts,
    version_question,
)
from research_agent.ingest.loaders import Document


def doc(path: str, text: str) -> Document:
    return Document(
        path=path, title="t", text=text, sha256=hashlib.sha256(text.encode()).hexdigest()
    )


SYSVARS = doc(
    "system-variables.md",
    """# 系统变量

## 变量参考

### `mem_quota` 从 v6.1.0 版本开始引入

- 作用域：SESSION | GLOBAL
- 是否持久化到集群：是
- 类型：整数型
- 默认值：`1073741824` (1 GiB)
- 范围：`[-1, 9223372036854775807]`
- 该变量用来设置单个会话的内存使用限制。

```sql
- 默认值：不是条目里的内容
```

### `debug_ratio`

- 作用域：SESSION
- 默认值：`0`
- 该变量用于 TiDB 内部测试，不推荐修改。

### `old_default`

- 作用域：GLOBAL
- 默认值：在 v7.0.0 之前版本中为 `OFF`，之后为 `ON`。
- 控制某个功能。
""",
)

TIKV = doc(
    "tikv-configuration-file.md",
    """# TiKV 配置文件描述

## 全局配置

### `abort-on-panic`

+ 设置 TiKV panic 时是否调用 `abort()` 退出进程。
+ 默认值：false

## log.file 从 v5.4.0 版本开始引入

### `max-size` 从 v5.4.0 版本开始引入

+ 单个 log 文件最大大小。
+ 默认值：300
+ 单位：MiB

## rocksdb.defaultcf | rocksdb.writecf

### `block-size`

+ RocksDB block 的默认大小。
+ 默认值：32KiB

## raftdb

### `block-size`

+ raftdb 的 block 大小。
+ 默认值：64KiB
""",
)

RELEASE = doc(
    "releases/release-8.5.0.md",
    """# TiDB 8.5.0 Release Notes

## 功能详情

| `fake` | 修改 | 不在兼容性变更里 |

## 兼容性变更

### 系统变量

| 变量名 | 修改类型 | 描述 |
|--------|----------|------|
| `fast_create` | 修改 | 经测试后，默认值从 `OFF` 修改为 `ON`。 |
| `scan_conc` | 修改 | 最小值从 `1` 改为 `0`。 |
| `new_var` | 新增 | 控制新功能。 |
| `gone_var` | 删除 | 该变量被删除。 |
| `twice` | 修改 | 默认值从 `OFF` 修改为 `ON`。 |

### 配置参数

| 配置文件或组件 | 配置项 | 修改类型 | 描述 |
|--------|----------|------|------|
| TiKV | `raft-client-queue-size` | 修改 | 默认值从 `8192` 修改为 `16384`。 |
| TiKV | `grpc-conn` | 修改 | 默认值从 `1` 修改为 `MAX(1, MIN(4, CPU 核数 / 8))`。 |
| TiKV | `old-a`；`old-b` | 废弃 | 由新配置组替代。 |
| TiKV | `rocksdb.defaultcf \\| rocksdb.writecf` | 修改 | 默认值从 `a` 修改为 `b`。 |
| TiFlash | `flash.x` | 新增 | 不在子集里的组件。 |

## 移除功能

| `gone` | 删除 | 不在兼容性变更里 |
""",
)


def test_parse_entries_and_param_facts() -> None:
    entries = parse_entries(TIKV)
    assert [e.name for e in entries] == [
        "abort-on-panic",
        "log.file.max-size",
        "rocksdb.defaultcf | rocksdb.writecf.block-size",
        "raftdb.block-size",
    ]
    assert entries[1].since == ["v5.4.0"]
    facts = param_facts(SYSVARS)
    mem = [f for f in facts if f.name == "mem_quota"]
    assert [f.field for f in mem] == ["作用域", "类型", "默认值", "范围"]  # 持久化 is metadata
    default = mem[2]
    assert (
        default.value == "`1073741824` (1 GiB)"
        and default.quote == "- 默认值：`1073741824` (1 GiB)"
    )
    assert default.description == "该变量用来设置单个会话的内存使用限制。"
    assert default.since == ["v6.1.0"]
    tikv = param_facts(TIKV)
    assert tikv[1].quote == "+ 默认值：300" and tikv[1].description.startswith("单个 log 文件")


def test_param_questions() -> None:
    facts = unique_param_facts(param_facts(SYSVARS) + param_facts(TIKV))
    qs = {(q.fact["name"], q.rule): q for f in facts if (q := param_question(f))}
    default = qs[("mem_quota", "param:默认值")]
    assert default.question == "TiDB 系统变量 `mem_quota` 的默认值是多少？"
    assert (default.answer, default.answer_aliases) == (
        "1073741824 (1 GiB)",
        ["1073741824", "1 GiB"],
    )
    assert qs[("mem_quota", "param:作用域")].answer == "SESSION 和 GLOBAL"
    assert ("mem_quota", "param:类型") not in qs  # types are not asked
    assert not any(name == "debug_ratio" for name, _ in qs)  # internal test variable
    assert ("old_default", "param:默认值") not in qs  # the default depends on the version
    assert ("old_default", "param:作用域") in qs  # its scope does not
    assert not any("|" in str(name) for name, _ in qs)  # shared by several config sections
    assert (
        qs[("log.file.max-size", "param:默认值")].question
        == "TiKV 配置项 `log.file.max-size` 的默认值是多少？"
    )
    assert ("log.file.max-size", "param:单位") not in qs  # config items: defaults only
    assert ("raftdb.block-size", "param:默认值") in qs


def test_clean_value_and_aliases() -> None:
    assert clean_value("`8192`。TiDB v7.6.0 之前，默认值为 `1`") == "8192"
    assert clean_value("在 v7.0.0 之前为 `OFF`") is None
    assert clean_value('""') is None and clean_value("与 `x` 相同") is None
    assert split_aliases("300") == ("300", [])
    assert clean_value("`168h`，即 7 天") == "168h (7 天)"
    assert split_aliases("168h (7 天)") == ("168h (7 天)", ["168h", "7 天"])


NEW_SYSVARS = doc(
    "system-variables.md",
    """# 系统变量

## 变量参考

### `fast_create`

- 默认值：`ON`。在 v8.5.0 之前，默认值为 `OFF`。

### new_var 从 v7.5.5 和 v8.5.0 版本开始引入

- 默认值：`1`

### `twice`

- 默认值：`ON`
""",
)
OLD_SYSVARS = doc(
    "system-variables.md",
    """# 系统变量

## 变量参考

### `fast_create`

- 默认值：`OFF`

### `gone_var`

- 默认值：`1`

### `twice`

- 默认值：`ON`
""",
)
NEW_TIKV = doc(
    "tikv-configuration-file.md",
    """# TiKV 配置文件描述

## server

### `raft-client-queue-size`

+ 默认值：16384

### `grpc-conn`

+ 默认值：

    + 从 v8.5.0 起，默认值调整为 `MAX(1, MIN(4, CPU 核数 / 8))`。
    + 之前的版本中，默认值为 1。

## raftstore

### `old-a`

> 从 v7.5.7 和 v8.5.0 开始，该配置项被废弃。

+ 默认值：10s

### `old-b`

+ 默认值：1
""",
)
OLD_TIKV = doc(
    "tikv-configuration-file.md",
    """# TiKV 配置文件描述

## server

### `raft-client-queue-size`

+ 默认值：8192

### `grpc-conn`

+ 默认值：1
""",
)


def test_release_changes() -> None:
    changes = release_changes(RELEASE)
    assert [c.name for c in changes] == [
        "fast_create",
        "scan_conc",
        "new_var",
        "gone_var",
        "twice",
        "raft-client-queue-size",
        "grpc-conn",
        "old-a",
        "old-b",
        "rocksdb.defaultcf | rocksdb.writecf",
        "flash.x",
    ]
    assert (changes[0].old_default, changes[0].new_default) == ("OFF", "ON")
    assert changes[5].component == "TiKV" and changes[5].version == "v8.5.0"
    assert changes[6].new_default == "MAX(1, MIN(4, CPU 核数 / 8))"  # backticks keep spaces
    assert changes[7].quote == changes[8].quote  # one row, two names


def test_version_questions_are_checked_against_both_versions() -> None:
    new, old = ParamIndex([NEW_SYSVARS, NEW_TIKV]), ParamIndex([OLD_SYSVARS, OLD_TIKV])
    changes = unique_changes(release_changes(RELEASE))
    out = {c.name: version_question(c, new, old, set()) for c in changes}
    reasons = {name: why for name, (_, why) in out.items()}
    assert reasons == {
        "fast_create": "ok",
        "scan_conc": "not_a_default_change",
        "new_var": "ok",
        "gone_var": "ok",
        "twice": "old_default_mismatch",  # v8.1 docs already say ON
        "raft-client-queue-size": "ok",
        "grpc-conn": "ok",  # the default is a nested list in the v8.5 docs
        "old-a": "ok",
        "old-b": "no_version_mark",  # no deprecation note to check the version against
        "rocksdb.defaultcf | rocksdb.writecf": "not_in_parameter_docs",
        "flash.x": "outside_subset",
    }
    fast = out["fast_create"][0]
    assert fast is not None
    assert (
        fast.question
        == "与 v8.1.0 相比，v8.5.8 中 TiDB 系统变量 `fast_create` 的默认值有什么变化？"
    )
    assert fast.answer == "从 OFF 改为 ON（v8.5.0 起）" and fast.answer_aliases == [
        "从 OFF 改为 ON"
    ]
    assert [e.doc for e in fast.more_evidence] == ["system-variables.md"]
    queue = out["raft-client-queue-size"][0]
    assert queue is not None and "`server.raft-client-queue-size`" in queue.question
    assert queue.group == "TiKV 配置项:server" and queue.stratum == "v8.5.0|TiKV 配置项"
    # Introduced on two release lines at once: asked within v8, both checked against the mark.
    new_var = out["new_var"][0]
    assert new_var is not None and new_var.question.startswith("在 v8 系列中")
    assert new_var.answer == "v8.5.0"
    old_a = out["old-a"][0]
    assert (
        old_a is not None and old_a.question.startswith("在 v8 系列中") and old_a.answer == "v8.5.0"
    )
    assert out["gone_var"][0] is not None and out["gone_var"][0].answer == "v8.5.0"
    # A change also made on the v8.1 line after v8.1.0 is dropped.
    skipped = version_question(changes[0], new, old, {("system_variable", "fast_create")})
    assert skipped == (None, "changed_on_the_v8_1_line")
    assert changed_names(changes[:1]) == {("system_variable", "fast_create")}


def _draft(source: str, stratum: str, group: str | None = None) -> DraftQuestion:
    return DraftQuestion(
        kind="version",
        question=f"q {source}",
        answer="a",
        doc="d",
        quote=source,
        source=source,
        stratum=stratum,
        group=group,
        rule="release:新增",
        fact={},
    )


def test_stratified_sample_spreads_versions_and_limits_groups() -> None:
    items = [_draft(f"a{i}", "v8.4.0|TiKV", "TiKV:server") for i in range(6)]
    items += [_draft(f"b{i}", "v8.5.0|TiDB") for i in range(3)]
    items += [_draft("c0", "v8.5.4|PD")]
    picked = stratified_sample(items, 5, seed=1)
    strata = [q.stratum.split("|")[0] for q in picked]
    assert set(strata[:3]) == {"v8.4.0", "v8.5.0", "v8.5.4"}  # one per version first
    assert sum(q.group == "TiKV:server" for q in picked) == 2
    shared: dict[str, int] = {"TiKV:server": 2}
    again = stratified_sample(items, 5, seed=1, group_counts=shared)
    assert not any(q.group for q in again)  # the limit was already used by another call


def test_locate_chunk_uses_the_entry_heading() -> None:
    scope = next(f for f in param_facts(SYSVARS) if f.name == "mem_quota" and f.field == "作用域")
    draft = param_question(scope)
    assert draft is not None and draft.anchor == "`mem_quota` 从 v6.1.0 版本开始引入"
    line = "- 作用域：SESSION | GLOBAL"
    heading = "### `mem_quota` 从 v6.1.0 版本开始引入"
    header = "[system-variables.md] 系统变量 > `mem_quota` 从 v6.1.0 版本开始引入"
    other = {"id": "0", "doc": "system-variables.md", "text": f"### `x`\n\n{line}"}
    # The header line also names the entry; the quote must follow the heading in the body.
    own = {"id": "1", "doc": "system-variables.md", "text": f"{header}\n\n{heading}\n\n{line}"}
    assert locate_chunk([other, own], draft) == "1"
    # An entry continued from the previous chunk: found through the chunk's sections.
    rest = {"id": "2", "doc": "system-variables.md", "text": line, "sections": [[heading[4:]]]}
    assert locate_chunk([other, rest], draft) == "2"
    assert locate_chunk([other], draft) is None


def test_pd_default_synonym() -> None:
    pd = doc(
        "pd-configuration-file.md",
        "# PD\n\n## schedule\n\n### `max-merge-region-size`\n\n+ 控制合并上限。\n"
        "+ 默认：54。在 v8.4.0 之前，默认值为 20。\n",
    )
    (fact,) = param_facts(pd)
    assert (fact.name, fact.field) == ("schedule.max-merge-region-size", "默认值")
    draft = param_question(fact)
    assert draft is not None and draft.answer == "54"
    assert ParamIndex.default(parse_entries(pd)[0]) == (
        "54。在 v8.4.0 之前，默认值为 20。",
        "+ 默认：54。在 v8.4.0 之前，默认值为 20。",
    )
