from __future__ import annotations

import hashlib

from research_agent.evalset.tidb_facts import param_facts, parse_entries, release_changes
from research_agent.evalset.tidb_questions import (
    clean_value,
    param_question,
    seeded_sample,
    split_aliases,
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

### 配置参数

| 配置文件或组件 | 配置项 | 修改类型 | 描述 |
|--------|----------|------|------|
| TiKV | `raft-client-queue-size` | 修改 | 默认值从 `8192` 修改为 `16384`。 |
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
    assert entries[1].since == "v5.4.0"
    facts = param_facts(SYSVARS)
    mem = [f for f in facts if f.name == "mem_quota"]
    assert [f.field for f in mem] == ["作用域", "类型", "默认值", "范围"]  # 持久化 is metadata
    default = mem[2]
    assert (
        default.value == "`1073741824` (1 GiB)"
        and default.quote == "- 默认值：`1073741824` (1 GiB)"
    )
    assert default.description == "该变量用来设置单个会话的内存使用限制。"
    assert default.since == "v6.1.0"
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
    assert not any("|" in name for name, _ in qs)  # shared by several config sections
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


def test_release_changes_and_questions() -> None:
    changes = release_changes(RELEASE)
    assert [c.name for c in changes] == [
        "fast_create",
        "scan_conc",
        "new_var",
        "raft-client-queue-size",
        "rocksdb.defaultcf | rocksdb.writecf",
        "flash.x",
    ]
    assert (changes[0].old_default, changes[0].new_default) == ("OFF", "ON")
    assert changes[3].component == "TiKV" and changes[3].version == "v8.5.0"
    qs = [q for c in unique_changes(changes) if (q := version_question(c))]
    assert [q.rule for q in qs] == ["release:修改", "release:新增", "release:修改", "release:修改"]
    fast = qs[0]
    assert fast.question == (
        "从 v8.1 升级到最新的 v8.5 版本后，TiDB 系统变量 `fast_create` 的默认值发生了什么变化？"
    )
    assert fast.answer == "v8.5.0 起默认值从 OFF 改为 ON"
    assert qs[1].answer == "v8.5.0" and qs[1].answer_aliases == ["8.5.0"]
    assert (
        qs[2].question.startswith("从 v8.1")
        and "TiKV 配置项 `raft-client-queue-size`" in qs[2].question
    )
    assert not any("flash.x" in q.question for q in qs)  # TiFlash is outside the subset
    assert seeded_sample(qs, 2, seed=1) == seeded_sample(list(reversed(qs)), 2, seed=1)
