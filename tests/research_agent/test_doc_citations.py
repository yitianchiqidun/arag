from __future__ import annotations

from typing import Any

from research_agent.assistant.retrieval import UNCITED_NOTE, doc_citations
from research_agent.eval.citations import ChunkRef

REFS = {
    "7": ChunkRef(doc="a.md", headings=["A", "x"], sections=[["A", "x"]]),
    "12": ChunkRef(doc="b.md", headings=["B"], sections=[["B"]]),
}


def record(answer: str) -> dict[str, Any]:
    return {
        "pred_answer": answer,
        "trajectory": [
            {"tool_name": "keyword_search", "tool_result": "Chunk ID: 3, Matched ..."},
            {"tool_name": "read_chunk", "tool_result": "[Chunk 12]\nbody\n\n[Chunk 7]\nbody"},
        ],
    }


def test_cited_answers_keep_their_citations() -> None:
    (c,) = doc_citations(record("默认值是 1 GB（Chunk 7）"), REFS, read_fallback=True)
    assert (c.chunk_id, c.label, c.status, c.cited) == ("7", "a.md › A › x", "read", True)


def test_uncited_arag_answer_falls_back_to_the_chunks_it_read() -> None:
    cs = doc_citations(record("默认值是 1 GB。"), REFS, read_fallback=True)
    assert [(c.chunk_id, c.cited, c.status) for c in cs] == [
        ("7", False, "read"),
        ("12", False, "read"),
    ]
    assert cs[0].label == "a.md › A › x" + UNCITED_NOTE


def test_no_fallback_for_the_fast_channel_or_a_cited_snippet() -> None:
    assert doc_citations(record("默认值是 1 GB。"), REFS, read_fallback=False) == []
    seen = record("见 Chunk 3。")
    seen["trajectory"][0]["tool_result"] = "Chunk ID: 3, Matched keywords in chunk: ..."
    refs = REFS | {"3": ChunkRef(doc="c.md")}
    assert [c.chunk_id for c in doc_citations(seen, refs, read_fallback=True)] == ["3"]
