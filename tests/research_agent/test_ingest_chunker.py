from __future__ import annotations

import hashlib

from research_agent.ingest.chunker import (
    ChunkerConfig,
    chunk_document,
    parse_sections,
    split_long,
    tiktoken_counter,
)
from research_agent.ingest.loaders import Document


def words(text: str) -> int:
    return len(text.split())


def doc(text: str, path: str = "docs/guide.md") -> Document:
    return Document(
        path=path,
        title="Guide",
        text=text,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def test_parse_sections_tracks_heading_paths_outside_code() -> None:
    text = "Intro\n# Guide\nA\n## Install\n```bash\n# not a heading\n```\n### GPU\nB\n## Use\nC\n"
    sections = parse_sections(text)
    assert [(s.level, s.path) for s in sections] == [
        (0, ()),
        (1, ("Guide",)),
        (2, ("Guide", "Install")),
        (3, ("Guide", "Install", "GPU")),
        (2, ("Guide", "Use")),
    ]
    assert "# not a heading" in sections[2].text


def test_small_sections_are_merged_up_to_the_limit() -> None:
    text = "# Guide\none two\n## A\nthree four five\n## B\n" + "w " * 10 + "\n## C\nsix\n"
    cfg = ChunkerConfig(max_tokens=50)
    chunks = chunk_document(doc(text), cfg, words)
    assert len(chunks) == 1 and chunks[0].headings == ["Guide"]
    assert chunks[0].text.startswith("[docs/guide.md] Guide\n\n# Guide\none two\n\n## A")

    small = chunk_document(doc(text), ChunkerConfig(max_tokens=50, merge_sections=False), words)
    assert [c.headings for c in small] == [
        ["Guide"],
        ["Guide", "A"],
        ["Guide", "B"],
        ["Guide", "C"],
    ]
    assert [c.position for c in small] == [0, 1, 2, 3]

    split = chunk_document(doc(text), ChunkerConfig(max_tokens=50), lambda t: 3 * words(t))
    assert [c.headings for c in split] == [["Guide"], ["Guide", "B"]]  # B no longer fits


def test_long_sections_are_split_and_blocks_stay_valid() -> None:
    table = "| k | v |\n| --- | --- |\n" + "\n".join(f"| r{i} | x y |" for i in range(30))
    code = "```python\n" + "\n".join(f"x{i} = {i}  # comment" for i in range(30)) + "\n```"
    text = f"# Ref\n\nShort intro.\n\n{table}\n\n{code}\n"
    limit = 60
    chunks = chunk_document(doc(text), ChunkerConfig(max_tokens=limit, source_header=False), words)

    assert all(c.tokens <= limit for c in chunks)
    table_chunks = [c.text for c in chunks if "| r" in c.text]
    assert len(table_chunks) > 1
    assert all("| k | v |\n| --- | --- |\n" in t for t in table_chunks)  # header repeated
    code_chunks = [c.text for c in chunks if "x1" in c.text or "x2" in c.text]
    assert all(t.count("```") % 2 == 0 for t in code_chunks)  # every piece is re-fenced
    body = "\n".join(c.text for c in chunks)
    assert all(f"| r{i} |" in body for i in range(30)) and all(f"x{i} =" in body for i in range(30))


def test_split_long_falls_back_to_sentences_and_words() -> None:
    sentence = "word " * 8
    paragraph = ". ".join([sentence.strip()] * 6) + "."
    pieces = split_long(paragraph, 20, words)
    assert len(pieces) > 1 and all(words(p) <= 20 for p in pieces)
    assert " ".join(pieces).split() == paragraph.split()
    assert split_long("x" * 50, 5, lambda t: len(t)) == ["x" * 50]  # unbreakable


def test_header_uses_title_without_headings_and_tiktoken_counts() -> None:
    chunks = chunk_document(doc("plain text\n"), ChunkerConfig(), tiktoken_counter())
    assert chunks[0].text == "[docs/guide.md] Guide\n\nplain text"
    assert chunks[0].tokens == tiktoken_counter()(chunks[0].text)
