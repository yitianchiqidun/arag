from __future__ import annotations

from research_agent.ingest.sentences import (
    SPLITTER_IDS,
    index_splitter,
    snippet_splitter,
    split_multilingual,
    weight,
)

MIXED = (
    "从 v8.5.0 开始，该变量默认值从 `1.5` 改为 `2.0`。修改后需要重启才能生效！"
    "注意：仅对新连接生效；旧连接不受影响。See the docs. Version v8.5.0 is LTS.\n短句。"
)


def test_split_multilingual_keeps_versions_and_splits_chinese_punctuation() -> None:
    assert split_multilingual(MIXED) == [
        "从 v8.5.0 开始，该变量默认值从 `1.5` 改为 `2.0`",
        "修改后需要重启才能生效",
        "注意：仅对新连接生效",
        "旧连接不受影响",
        "See the docs",
        "Version v8.5.0 is LTS",
    ]
    assert "短句" in split_multilingual(MIXED, min_weight=0)  # snippets keep short sentences
    upstream = index_splitter("upstream")(MIXED)
    assert upstream[0] == "0 开始，该变量默认值从 `1"  # upstream cuts versions and decimals


def test_weight_counts_cjk_twice() -> None:
    assert weight("abc") == 3 and weight("内存") == 4 and weight("默认值 0") == 8
    assert split_multilingual("默认值：0") == []  # weight 10 is not above the minimum
    assert split_multilingual("单条查询内存上限") == ["单条查询内存上限"]


def test_splitter_choices() -> None:
    assert index_splitter("multilingual") is split_multilingual
    assert snippet_splitter("upstream") is None
    split = snippet_splitter("multilingual")
    assert split is not None and split("甲。乙") == ["甲", "乙"]
    assert set(SPLITTER_IDS) == {"upstream", "multilingual"}
