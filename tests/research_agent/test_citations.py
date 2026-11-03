from __future__ import annotations

import json
from pathlib import Path

from research_agent.eval.citations import (
    ChunkRef,
    Citation,
    citation_stats,
    cited_chunk_ids,
    load_chunk_refs,
    record_citations,
    resolve_citations,
    shown_chunk_ids,
)


def test_cited_chunk_ids() -> None:
    answer = (
        "Founded in **2005** (Chunk 58). See chunks 54, 55 and 60; also [Chunk 12], "
        "Chunk ID: 7, chunk #9 and Chunk 58 again. The chunker uses a chunk size of 2."
    )
    assert cited_chunk_ids(answer) == ["58", "54", "55", "60", "12", "7", "9"]
    assert cited_chunk_ids("No sources.") == []


READ = "=" * 80 + "\n[Chunk 3]\n" + "-" * 80 + "\ntext\n" + "=" * 80
READ_AGAIN = "[Chunk 4]\n(This chunk has been read before)"
NOT_FOUND = "\n[Chunk 99] - Not found"
KEYWORD = "Chunk ID: 5, Matched keywords in chunk: ... x ...\n\nChunk ID: 3, Matched: ..."
SEMANTIC = "Chunk ID: 6 (Similarity: 0.512)\nMatched: ... y ..."
FULL_CHUNK_SEARCH = "=" * 80 + "\n[Chunk 8]\n" + "-" * 80 + "\nbody\n" + "=" * 80


def test_shown_chunk_ids() -> None:
    full, snippet = shown_chunk_ids(
        [
            ("read_chunk", READ),
            ("read_chunk", READ_AGAIN),
            ("read_chunk", NOT_FOUND),
            ("keyword_search", KEYWORD),
            ("semantic_search", SEMANTIC),
            ("keyword_search", FULL_CHUNK_SEARCH),  # the w/o read_chunk ablation
            ("dense_retrieval", json.dumps([{"chunk_id": "10", "score": 0.5}])),
        ]
    )
    assert full == {"3", "4", "8", "10"} and snippet == {"5", "6"}


def test_resolve_citations_and_labels() -> None:
    refs = {
        "3": ChunkRef(doc="docs/a.md", headings=["A", "Install"]),
        "5": ChunkRef(doc="docs/b.md"),
        "6": ChunkRef(),
    }
    citations = resolve_citations("Chunks 3, 5, 6 and 77", refs, full={"3"}, snippet={"5"})
    assert [(c.chunk_id, c.status) for c in citations] == [
        ("3", "read"),
        ("5", "snippet"),
        ("6", "unseen"),
        ("77", "unknown"),
    ]
    assert citations[0].label() == "docs/a.md › A › Install"
    assert citations[2].label() == "Chunk 6"
    merged = Citation(
        chunk_id="9",
        status="read",
        doc="r.md",
        headings=["R", "功能", "安全"],
        sections=[["R", "功能", "安全"], ["R", "兼容性变更"], ["R", "兼容性变更", "系统变量"]],
    )
    assert merged.label() == "r.md › R › 功能 › 安全；兼容性变更 › 系统变量"
    assert Citation(chunk_id="1", status="read", doc="d.md").label() == "d.md"


def test_load_chunk_refs_reads_both_formats(tmp_path: Path) -> None:
    dicts = tmp_path / "dicts.json"
    dicts.write_text(
        json.dumps(
            [{"id": "0", "text": "t", "doc": "docs/a.md", "headings": ["A"], "sections": [["A"]]}]
        )
    )
    strings = tmp_path / "strings.json"
    strings.write_text(json.dumps(["0:alpha", "1:beta: gamma"]))
    assert load_chunk_refs(dicts) == {
        "0": ChunkRef(doc="docs/a.md", headings=["A"], sections=[["A"]])
    }
    assert load_chunk_refs(strings) == {"0": ChunkRef(), "1": ChunkRef()}


def test_record_citations_and_stats() -> None:
    refs = {str(i): ChunkRef(doc=f"docs/{i}.md") for i in range(10)}
    records = [
        {
            "pred_answer": "Paris (Chunk 3; see also chunk 6)",
            "trajectory": [
                {"tool_name": "keyword_search", "tool_result": KEYWORD},
                {"tool_name": "read_chunk", "tool_result": READ},
            ],
            "evidence_chunks": ["3", "9"],
        },
        {"pred_answer": "Lyon (Chunk 5)", "trajectory": [], "evidence_chunks": ["2"]},
        {"pred_answer": "No idea", "trajectory": []},
    ]
    assert [c.status for c in record_citations(records[0], refs)] == ["read", "unseen"]
    stats = citation_stats(records, refs)
    assert stats.answers_citing == 2 / 3 and stats.cited == 3
    assert stats.cited_read == 1 / 3 and stats.cited_unseen == 2 and stats.cited_snippet == 0
    assert (stats.evidence_n, stats.evidence_cited) == (2, 0.5)
