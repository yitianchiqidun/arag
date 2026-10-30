"""Naive RAG baseline as described in the A-RAG paper (Section 4, Appendix B, Figure 6).

One dense retrieval of the top-k whole chunks (Qwen3-Embedding-0.6B, k=5), concatenated into
a single prompt, one LLM call without tools. ``NaiveRAG.run`` returns the same dict shape as
``BaseAgent.run`` so the batch runner, judge and summary treat it like any other agent.
"""

from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from arag.tools.read_chunk import ReadChunkTool
from research_agent.eval.subsets import sha256_file
from research_agent.llm.client import MeteredLLMClient

# Figure 6 of the paper shows one template without role markers; the first sentence is used
# as the system message and the rest as the user message.
SYSTEM_PROMPT = "You are a helpful assistant. Answer the question based on the provided context."
USER_TEMPLATE = (
    "[Context]\n{context}\n\n[Question]\n{question}\n\n"
    "Please provide a direct answer based on the context above."
)


class ChunkIndexMeta(BaseModel):
    model: str
    chunks_file: str
    chunks_sha256: str
    n: int
    dim: int
    created: str


class ChunkIndex:
    """Chunk-level embeddings (A-RAG's own index is sentence-level), cached next to it."""

    def __init__(self, ids: list[str], embeddings: np.ndarray) -> None:
        self.ids = ids
        self.embeddings = embeddings

    @staticmethod
    def paths(index_dir: str | Path) -> tuple[Path, Path]:
        base = Path(index_dir)
        return base / "chunk_index.npz", base / "chunk_index.meta.json"

    @classmethod
    def load_or_build(
        cls,
        reader: ReadChunkTool,
        index_dir: str | Path,
        model: Any,
        model_name: str,
        batch_size: int,
        lock: threading.Lock,
    ) -> ChunkIndex:
        """Load the cache if it matches this corpus and model, otherwise rebuild it."""
        data_path, meta_path = cls.paths(index_dir)
        chunks_sha = sha256_file(reader.chunks_file)
        if data_path.exists() and meta_path.exists():
            meta = ChunkIndexMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
            if meta.model == model_name and meta.chunks_sha256 == chunks_sha:
                with np.load(data_path) as data:
                    return cls([str(i) for i in data["ids"]], data["embeddings"])

        ids = [str(c["id"]) for c in reader.chunks]
        texts = [c["text"] for c in reader.chunks]
        # Same settings as scripts/build_index.py: normalized, no query/document instruction.
        with lock:
            embeddings = np.asarray(
                model.encode(
                    texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
                ),
                dtype=np.float32,
            )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(data_path, ids=np.array(ids), embeddings=embeddings)
        meta = ChunkIndexMeta(
            model=model_name,
            chunks_file=str(reader.chunks_file),
            chunks_sha256=chunks_sha,
            n=len(ids),
            dim=int(embeddings.shape[1]),
            created=date.today().isoformat(),
        )
        meta_path.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return cls(ids, embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        scores = self.embeddings @ query_embedding
        # Stable sort so ties keep corpus order, which makes runs reproducible.
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.ids[i], float(scores[i])) for i in order]


class NaiveRAG:
    """Duck-types ``BaseAgent`` for ``MeteredBatchRunner`` (``llm``, ``run``, ``last_result``)."""

    def __init__(
        self,
        llm_client: MeteredLLMClient,
        index: ChunkIndex,
        reader: ReadChunkTool,
        embedding_model: Any,
        embedding_lock: threading.Lock,
        top_k: int = 5,
    ) -> None:
        self.llm = llm_client
        self.index = index
        self.reader = reader
        self.embedding_model = embedding_model
        self.embedding_lock = embedding_lock
        self.top_k = top_k
        self.last_result: dict[str, Any] | None = None

    def build_messages(self, query: str, chunk_ids: list[str]) -> list[dict[str, Any]]:
        context = "\n\n".join(f"[Chunk {cid}]\n{self.reader.chunks_dict[cid]}" for cid in chunk_ids)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(context=context, question=query)},
        ]

    def run(self, query: str) -> dict[str, Any]:
        self.last_result = None
        with self.embedding_lock:
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        hits = self.index.search(np.asarray(query_embedding, dtype=np.float32), self.top_k)
        chunk_ids = [cid for cid, _ in hits]
        retrieved_tokens = sum(
            len(self.reader.tokenizer.encode(self.reader.chunks_dict[cid])) for cid in chunk_ids
        )

        response = self.llm.chat(messages=self.build_messages(query, chunk_ids), tools=None)
        step = {
            "loop": 1,
            "tool_name": "dense_retrieval",
            "arguments": {"query": query, "top_k": self.top_k},
            "tool_result": json.dumps([{"chunk_id": c, "score": round(s, 4)} for c, s in hits]),
            "retrieved_tokens": retrieved_tokens,
            "chunks_found": len(hits),
        }
        result = {
            "answer": response["message"].get("content") or "",
            "trajectory": [step],
            "total_cost": response["cost"],
            "loops": 1,
            "total_retrieved_tokens": retrieved_tokens,
            "retrieval_logs": [
                {
                    "tool_name": "dense_retrieval",
                    "tokens": retrieved_tokens,
                    "metadata": {"query": query, "chunk_ids": chunk_ids},
                }
            ],
            "chunks_read_count": len(chunk_ids),
            "chunks_read_ids": chunk_ids,
        }
        self.last_result = result
        return result
