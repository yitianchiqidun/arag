from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research_agent.ingest.loaders import LoaderConfig, load_corpus, select_files
from research_agent.ingest.markdown import normalize_markdown, split_front_matter
from research_agent.ingest.sources import DirectorySource, GitSource, clone_or_fetch


def test_normalize_markdown() -> None:
    lines = [
        "<!-- 编辑说明 -->",
        "## 部署 { #deploy }",
        "",
        "",
        "",
        "见 ![架构图](media/arch.png) 和 <span>标注</span>。",
        "```html",
        "<p>代码块里的标签保留</p>",
        "```",
    ]
    assert normalize_markdown("\n".join(lines)).splitlines() == [
        "## 部署",
        "",
        "见 架构图 和 标注。",
        "```html",
        "<p>代码块里的标签保留</p>",
        "```",
    ]


def test_normalize_markdown_site_syntax() -> None:
    lines = [
        '### `mem_quota` <span class="version-mark">从 v6.1.0 版本开始引入</span>',
        "",
        '{{< copyable "sql" >}}',
        "",
        "<EmailSubscriptionWrapper />",
        "参见 [`KILL [TIDB]`](/sql-statements/kill.md) 与 [提示](/hints.md#a_(b))。",
        "",
        "| 配置项 | 说明 |",
        "| --- | --- |",
        "| <ul><li>`a`</li><li>`b`</li></ul> | <code>x</code> |",
        "",
        "<table>",
        "<tr><th>分类</th><th>功能</th></tr>",
        '<tr><td rowspan="2">性能</td><td>包括：<ul><li>甲</li><li>乙 &amp; 丙</li></ul></td></tr>',
        "</table>",
        "```toml",
        "# 见[官方格式](https://example.com)，<host>:<port>",
        "```",
    ]
    assert normalize_markdown("\n".join(lines), link_targets=False).splitlines() == [
        "### `mem_quota` 从 v6.1.0 版本开始引入",
        "",
        "参见 `KILL [TIDB]` 与 提示。",
        "",
        "| 配置项 | 说明 |",
        "| --- | --- |",
        "| `a`；`b` | `x` |",
        "",
        "分类 | 功能",
        "",
        "性能 | 包括：甲；乙 & 丙",
        "",
        "```toml",
        "# 见[官方格式](https://example.com)，<host>:<port>",
        "```",
    ]
    assert "[提示](/hints.md#a_(b))" in normalize_markdown(lines[5])  # targets kept by default


def test_inline_table_tags_do_not_start_a_table() -> None:
    lines = [
        "当 `SELECT ... FOR UPDATE OF <table>` 引用表别名时报错。",
        "",
        "## 兼容性变更",
        "<table>",
        "没有结束标签的表格，后面的正文必须保留",
        "## 系统变量",
    ]
    assert normalize_markdown("\n".join(lines)).splitlines() == [
        "当 `SELECT ... FOR UPDATE OF <table>` 引用表别名时报错。",
        "",
        "## 兼容性变更",
        "<table>",  # not a table without "</table>": left as it is
        "没有结束标签的表格，后面的正文必须保留",
        "## 系统变量",
    ]


def test_split_front_matter() -> None:
    meta, body = split_front_matter("---\ntitle: 安装\nsummary: 简介\n---\n# 安装\n")
    assert meta == {"title": "安装", "summary": "简介"} and body == "# 安装\n"
    assert split_front_matter("# 无元数据\n---\n") == ({}, "# 无元数据\n---\n")


def test_select_files() -> None:
    cfg = LoaderConfig(include=["*.md"], exclude=["media/*", "TOC*.md"])
    files = ["a.md", "sub/b.md", "media/x.md", "TOC.md", "logo.png"]
    assert select_files(files, cfg) == (["a.md", "sub/b.md"], {"media/*": 1, "TOC*.md": 1})


def write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_loader_link_targets(tmp_path: Path) -> None:
    write(tmp_path, {"a.md": "# A\n见 [B](b.md)。\n"})
    keep = load_corpus(DirectorySource(tmp_path), LoaderConfig(include=["*"]))[0][0]
    drop = load_corpus(DirectorySource(tmp_path), LoaderConfig(include=["*"], link_targets=False))
    assert "[B](b.md)" in keep.text and drop[0][0].text == "# A\n见 B。\n"


def test_load_corpus_from_directory(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "guide.md": "---\ntitle: 使用指南\n---\n正文\n",
            "notes/plain.md": "# 笔记\n内容\n",
            "empty.md": "<!-- nothing -->\n",
            "page.html": "<h1>Page</h1>",
            "logo.png": "png",
        },
    )
    docs, report = load_corpus(DirectorySource(tmp_path), LoaderConfig(include=["*"]))

    assert [d.path for d in docs] == ["guide.md", "notes/plain.md"]
    guide, plain = docs
    assert (guide.title, guide.text, len(guide.sha256)) == ("使用指南", "正文\n", 64)
    assert plain.title == "笔记"
    assert report.skipped_empty == ["empty.md"]
    assert report.unsupported == ["logo.png", "page.html"]
    with pytest.raises(FileNotFoundError):
        DirectorySource(tmp_path).read_text("../outside.md")


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_git_source_reads_each_ref(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    write(repo, {"a.md": "# A\nold\n", "b.md": "# B\n"})
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "v1")
    git(repo, "tag", "v1")
    write(repo, {"a.md": "# A\nnew\n", "c.md": "# C\n"})
    (repo / "b.md").unlink()
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "v2")

    v1, v2 = GitSource(repo, "v1"), GitSource(repo, "HEAD")
    assert v1.list_files() == ["a.md", "b.md"]
    assert v2.list_files() == ["a.md", "c.md"]
    assert v1.read_text("a.md") == "# A\nold\n" and v2.read_text("a.md") == "# A\nnew\n"
    assert v1.describe()["ref"] == "v1" and len(v1.describe()["commit"]) == 40
    with pytest.raises(FileNotFoundError):
        v2.read_text("b.md")


def test_shallow_fetch_of_branches(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "release-1")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    for i in range(2):  # two commits on release-1: only the latest is fetched
        write(repo, {"a.md": f"# A\nv1.{i}\n"})
        git(repo, "add", ".")
        git(repo, "commit", "-qm", f"v1.{i}")
    git(repo, "checkout", "-q", "-b", "release-2")
    write(repo, {"a.md": "# A\nv2\n"})
    git(repo, "commit", "-qam", "v2")

    dest = tmp_path / "clone"
    clone_or_fetch(f"file://{repo}", dest, ["release-2", "release-1"])
    clone_or_fetch(f"file://{repo}", dest, ["release-2", "release-1"])  # updates in place
    assert GitSource(dest, "origin/release-1").read_text("a.md") == "# A\nv1.1\n"
    assert GitSource(dest, "origin/release-2").read_text("a.md") == "# A\nv2\n"
    log = subprocess.run(
        ["git", "-C", str(dest), "rev-list", "--count", "origin/release-1"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert log.stdout.strip() == "1"
