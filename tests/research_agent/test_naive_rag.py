from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from arag.tools.read_chunk import ReadChunkTool
from research_agent.baselines.naive_rag import CITE_INSTRUCTION, SYSTEM_PROMPT, ChunkIndex, NaiveRAG
from research_agent.llm.client import MeteredLLMClient

from .fakes import PLUS_PRICING, completion

VOCAB = ["paris", "berlin", "physics", "louvre"]


class BagOfWordsModel:
    """Deterministic stand-in for SentenceTransformer.encode."""

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.encoded += texts
        vectors = np.array([[t.lower().count(w) for w in VOCAB] for t in texts], dtype=float)
        vectors += 1e-3  # avoid zero vectors
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def build(corpus: dict[str, Path], model: BagOfWordsModel, name: str = "bow") -> ChunkIndex:
    reader = ReadChunkTool(str(corpus["chunks"]))
    return ChunkIndex.load_or_build(
        reader, corpus["root"] / "index", model, name, 2, threading.Lock()
    )


def test_chunk_index_is_cached_per_model_and_corpus(tiny_corpus: dict[str, Path]) -> None:
    model = BagOfWordsModel()
    index = build(tiny_corpus, model)
    assert index.ids == ["0", "1", "2"] and len(model.encoded) == 3

    build(tiny_corpus, model)
    assert len(model.encoded) == 3  # loaded from cache

    build(tiny_corpus, model, name="other-model")
    assert len(model.encoded) == 6  # different model -> rebuilt

    tiny_corpus["chunks"].write_text(json.dumps(["0:berlin wall", "1:paris"]))
    assert build(tiny_corpus, model, name="other-model").ids == ["0", "1"]  # corpus changed


def test_chunk_index_search_ranks_by_cosine(tiny_corpus: dict[str, Path]) -> None:
    model = BagOfWordsModel()
    index = build(tiny_corpus, model)
    hits = index.search(model.encode(["louvre in paris"])[0], top_k=2)
    assert [cid for cid, _ in hits] == ["1", "0"]
    assert hits[0][1] > hits[1][1]


def test_naive_rag_single_call_with_top_k_chunks(
    env_api: None, tiny_corpus: dict[str, Path]
) -> None:
    model = BagOfWordsModel()
    reader = ReadChunkTool(str(tiny_corpus["chunks"]))
    client = MeteredLLMClient(model="m", pricing=PLUS_PRICING)
    payloads: list[dict[str, Any]] = []

    def post(payload: dict[str, Any], attempts: list[int]) -> dict[str, Any]:
        attempts[0] = 1
        payloads.append(payload)
        return completion("Berlin", prompt_tokens=300, completion_tokens=3).json()

    client._post_with_retry = post  # type: ignore[method-assign]
    rag = NaiveRAG(client, build(tiny_corpus, model), reader, model, threading.Lock(), top_k=2)

    result = rag.run("Where does Bob live? berlin")

    assert result["answer"] == "Berlin" and rag.last_result is result
    assert result["loops"] == 1 and result["chunks_read_ids"][0] == "2"
    assert result["total_cost"] == pytest.approx(PLUS_PRICING.cost(300, 3))
    assert result["total_retrieved_tokens"] == sum(
        len(reader.tokenizer.encode(reader.chunks_dict[c])) for c in result["chunks_read_ids"]
    )
    (payload,) = payloads
    assert "tools" not in payload
    system, user = payload["messages"]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert user["content"].startswith("[Context]\n[Chunk 2]\nbob lives in berlin.")
    assert user["content"].endswith("Please provide a direct answer based on the context above.")
    assert result["trajectory"][0]["tool_name"] == "dense_retrieval"


def test_naive_rag_can_ask_for_citations(env_api: None, tiny_corpus: dict[str, Path]) -> None:
    model = BagOfWordsModel()
    reader = ReadChunkTool(str(tiny_corpus["chunks"]))
    client = MeteredLLMClient(model="m", pricing=PLUS_PRICING)
    index = build(tiny_corpus, model)
    plain = NaiveRAG(client, index, reader, model, threading.Lock())
    citing = NaiveRAG(client, index, reader, model, threading.Lock(), cite_chunks=True)
    _, user = plain.build_messages("q", ["2"])
    _, cited = citing.build_messages("q", ["2"])
    assert cited["content"] == user["content"] + " " + CITE_INSTRUCTION
