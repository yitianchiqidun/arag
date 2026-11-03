"""Incremental index builds, written in A-RAG's own on-disk format.

A build directory holds:

- ``chunks.json``: the chunks as a list of dicts (``id``, ``text`` plus our metadata); the
  upstream tools accept this shape as is and ignore the extra keys.
- ``index/sentence_index.pkl``: the same dict ``scripts/build_index.py`` writes (sentence-level
  embeddings for ``semantic_search``), with sentences split by the upstream function or the
  multilingual one (sentences.py).
- ``index/chunk_index.npz`` + ``.meta.json``: the chunk-level cache of the Naive RAG baseline.
- ``manifest.json``: per document the hash of its normalized text, written last.

Rebuilding over an existing build reuses what has not changed. Documents whose hash is the
same keep their chunks without re-chunking; sentences and chunks whose text already has a
vector in the previous build keep that vector; only new texts are embedded. Chunk ids are
positions in (document path, position) order, so an incremental build produces the same
files as a full build of the same input.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, Field

from research_agent.baselines.naive_rag import ChunkIndex, ChunkIndexMeta
from research_agent.ingest.chunker import Chunk, ChunkerConfig, TokenCounter, chunk_document
from research_agent.ingest.loaders import Document, LoaderConfig
from research_agent.ingest.sentences import SPLITTER_IDS, SentenceSplit, index_splitter

CHUNKS_FILE = "chunks.json"
INDEX_DIR = "index"
SENTENCE_INDEX = "sentence_index.pkl"
MANIFEST_FILE = "manifest.json"
REPORT_FILE = "build_report.json"


EmbeddingDtype = Literal["float32", "model_default"]


class Embedder(Protocol):
    model_name: str
    dtype: str  # precision the model runs in; vectors of different precisions are not mixed

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Normalized float32 vectors, one row per text (``batch_size`` overrides the default)."""
        ...


class SentenceTransformerEmbedder:
    """Same call as ``scripts/build_index.py``: normalized, no instruction prefix.

    ``dtype="float32"`` overrides the model's default precision (bfloat16 for
    Qwen3-Embedding). In bfloat16 a text's vector depends on which other texts share its
    batch (padding), by up to ~5e-3 per component, enough to reorder search results; in
    float32 the difference is ~5e-7, so an incremental build matches a full one.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None,
        batch_size: int,
        dtype: EmbeddingDtype = "float32",
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.dtype: str = dtype
        kwargs = {"torch_dtype": torch.float32} if dtype == "float32" else {}
        self._model = SentenceTransformer(model_name, device=device, model_kwargs=kwargs)

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 2000,
        )
        return np.asarray(vectors, dtype=np.float32)


class DocEntry(BaseModel):
    sha256: str
    title: str
    chunks: int


class Manifest(BaseModel):
    schema_version: int = 1
    corpus: str
    source: dict[str, str]
    loader: LoaderConfig
    chunker: ChunkerConfig
    embedding_model: str
    embedding_dtype: str
    sentence_split: str = SPLITTER_IDS["upstream"]
    chunks_sha256: str
    built_at: str
    docs: dict[str, DocEntry]


class BuildReport(BaseModel):
    mode: Literal["full", "incremental"]
    notes: list[str] = Field(default_factory=list)  # why something could not be reused
    source: dict[str, str]
    previous_source: dict[str, str] | None = None
    docs_added: list[str] = Field(default_factory=list)
    docs_modified: list[str] = Field(default_factory=list)
    docs_deleted: list[str] = Field(default_factory=list)
    docs_unchanged: int = 0
    chunks: int = 0
    chunks_reused: int = 0  # taken from the previous build without re-chunking
    chunks_rechunked: int = 0
    chunks_new_text: int = 0  # chunk texts that did not exist in the previous build
    sentences: int = 0
    sentence_vectors_reused: int = 0
    sentence_vectors_embedded: int = 0  # distinct new sentence texts sent to the model
    chunk_vectors_reused: int = 0
    chunk_vectors_embedded: int = 0
    sentence_embed_seconds: float = 0.0
    chunk_embed_seconds: float = 0.0  # whole chunks are long: most of a full build's time
    sentence_batch_size: int | None = None
    chunk_batch_size: int | None = None
    seconds: float = 0.0
    # Process peaks so far (the whole build when run from the CLI); None when unknown.
    peak_rss_mib: float | None = None
    peak_gpu_allocated_mib: float | None = None
    peak_gpu_reserved_mib: float | None = None


def _peak_memory() -> dict[str, float | None]:
    """Peak resident memory of this process and peak CUDA memory of torch, in MiB."""
    peaks: dict[str, float | None] = {"peak_rss_mib": None}
    try:
        import resource

        peaks["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except ImportError:  # not on Windows
        pass
    torch = sys.modules.get("torch")  # only if the embedder loaded it
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        peaks["peak_gpu_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        peaks["peak_gpu_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
    return peaks


class DocChanges(BaseModel):
    added: list[str]
    modified: list[str]
    deleted: list[str]
    unchanged: list[str]


def doc_changes(old: dict[str, str], new: dict[str, str]) -> DocChanges:
    """Compare ``path -> sha256`` maps of two corpus versions."""
    return DocChanges(
        added=sorted(p for p in new if p not in old),
        modified=sorted(p for p in new if p in old and old[p] != new[p]),
        deleted=sorted(p for p in old if p not in new),
        unchanged=sorted(p for p in new if p in old and old[p] == new[p]),
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _record(chunk_id: str, chunk: Chunk) -> dict[str, Any]:
    return {"id": chunk_id, **chunk.model_dump()}


class _Previous:
    """What a previous build in the output directory offers for reuse."""

    def __init__(self, out: Path, notes: list[str], sentence_split: str) -> None:
        self.manifest: Manifest | None = None
        self.chunks_by_doc: dict[str, list[Chunk]] = {}
        self.sentence_vectors: dict[str, np.ndarray] = {}
        self.chunk_vectors: dict[str, np.ndarray] = {}
        manifest_path = out / MANIFEST_FILE
        if not manifest_path.exists():
            notes.append("no previous build")
            return
        manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        chunks_path = out / CHUNKS_FILE
        raw = chunks_path.read_bytes() if chunks_path.exists() else b""
        if hashlib.sha256(raw).hexdigest() != manifest.chunks_sha256:
            notes.append("previous chunks.json does not match its manifest (interrupted build?)")
            return
        self.manifest = manifest
        records = json.loads(raw)
        text_by_id = {r["id"]: r["text"] for r in records}
        for r in records:
            chunk = Chunk.model_validate({k: v for k, v in r.items() if k != "id"})
            self.chunks_by_doc.setdefault(chunk.doc, []).append(chunk)

        sentence_file = out / INDEX_DIR / SENTENCE_INDEX
        if manifest.sentence_split != sentence_split:
            notes.append("sentence split changed: sentence vectors not reused")
        elif sentence_file.exists():
            with open(sentence_file, "rb") as f:
                data = pickle.load(f)
            if data.get("model_name") == manifest.embedding_model:
                self.sentence_vectors = dict(zip(data["sentences"], data["embeddings"]))
        chunk_file, _ = ChunkIndex.paths(out / INDEX_DIR)
        if chunk_file.exists():
            with np.load(chunk_file) as data:
                ids = [str(i) for i in data["ids"]]
                self.chunk_vectors = {
                    text_by_id[i]: v for i, v in zip(ids, data["embeddings"]) if i in text_by_id
                }


def _vectors(
    texts: list[str],
    known: dict[str, np.ndarray],
    embedder: Embedder,
    batch_size: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """Matrix for ``texts`` reusing ``known`` vectors; returns (matrix, reused, embedded)."""
    missing = list(dict.fromkeys(t for t in texts if t not in known))
    table = dict(known)
    if missing:
        table.update(zip(missing, embedder.encode(missing, batch_size)))
    matrix = np.stack([table[t] for t in texts]).astype(np.float32)
    reused = sum(t in known for t in texts)
    return matrix, reused, len(missing)


def build_index(
    docs: list[Document],
    out_dir: str | Path,
    *,
    corpus: str,
    source: dict[str, str],
    loader: LoaderConfig,
    chunker: ChunkerConfig,
    embedder: Embedder,
    count_tokens: TokenCounter,
    sentence_split: SentenceSplit = "upstream",
    chunk_batch_size: int | None = None,
) -> BuildReport:
    """Build (or update in place) the index of ``docs`` in ``out_dir``."""
    start = time.perf_counter()
    if not docs:
        raise ValueError("no documents to index")
    paths = [d.path for d in docs]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate document paths")
    out = Path(out_dir)
    notes: list[str] = []
    prev = _Previous(out, notes, SPLITTER_IDS[sentence_split])
    pm = prev.manifest
    if pm is not None and (pm.embedding_model, pm.embedding_dtype) != (
        embedder.model_name,
        embedder.dtype,
    ):
        notes.append(
            f"embedding model changed from {pm.embedding_model} ({pm.embedding_dtype}): "
            "vectors not reused"
        )
        prev.sentence_vectors, prev.chunk_vectors = {}, {}
    reuse_chunks = pm is not None and pm.chunker == chunker
    if pm is not None and not reuse_chunks:
        notes.append("chunker config changed: every document is re-chunked")

    old = {p: e.sha256 for p, e in pm.docs.items()} if pm else {}
    changes = doc_changes(old, {d.path: d.sha256 for d in docs})
    report = BuildReport(
        mode="incremental" if pm is not None else "full",
        notes=notes,
        source=source,
        previous_source=pm.source if pm else None,
        docs_added=changes.added,
        docs_modified=changes.modified,
        docs_deleted=changes.deleted,
        docs_unchanged=len(changes.unchanged),
    )

    unchanged = set(changes.unchanged)
    records: list[dict[str, Any]] = []
    entries: dict[str, DocEntry] = {}
    for doc in sorted(docs, key=lambda d: d.path):
        if reuse_chunks and doc.path in unchanged and doc.path in prev.chunks_by_doc:
            chunks = prev.chunks_by_doc[doc.path]
            report.chunks_reused += len(chunks)
        else:
            chunks = chunk_document(doc, chunker, count_tokens)
            report.chunks_rechunked += len(chunks)
        for chunk in chunks:
            records.append(_record(str(len(records)), chunk))
        entries[doc.path] = DocEntry(sha256=doc.sha256, title=doc.title, chunks=len(chunks))

    split = index_splitter(sentence_split)
    sentences: list[str] = []
    sentence_to_chunk: list[str] = []
    for r in records:
        for sentence in split(r["text"]):
            sentences.append(sentence)
            sentence_to_chunk.append(r["id"])
    chunk_texts = [r["text"] for r in records]
    old_texts = {c.text for cs in prev.chunks_by_doc.values() for c in cs}
    report.chunks = len(records)
    report.chunks_new_text = sum(t not in old_texts for t in chunk_texts)
    report.sentences = len(sentences)
    t0 = time.perf_counter()
    sentence_matrix, report.sentence_vectors_reused, report.sentence_vectors_embedded = _vectors(
        sentences, prev.sentence_vectors, embedder
    )
    t1 = time.perf_counter()
    # Whole chunks are long: a smaller batch keeps the CUDA memory peak down (D26).
    chunk_matrix, report.chunk_vectors_reused, report.chunk_vectors_embedded = _vectors(
        chunk_texts, prev.chunk_vectors, embedder, chunk_batch_size
    )
    report.sentence_embed_seconds = round(t1 - t0, 1)
    report.chunk_embed_seconds = round(time.perf_counter() - t1, 1)
    report.sentence_batch_size = getattr(embedder, "batch_size", None)
    report.chunk_batch_size = chunk_batch_size or report.sentence_batch_size

    # Write every file, then the manifest: a crash in between leaves a manifest whose
    # chunks hash no longer matches, and the next build falls back to a full build.
    chunks_bytes = json.dumps(records, ensure_ascii=False, indent=1).encode("utf-8")
    chunks_path = out / CHUNKS_FILE
    _atomic_write(chunks_path, chunks_bytes)
    index_data = {
        "sentences": sentences,
        "embeddings": sentence_matrix,
        "sentence_to_chunk": sentence_to_chunk,
        "chunks": {r["id"]: r for r in records},
        "model_name": embedder.model_name,
    }
    _atomic_write(out / INDEX_DIR / SENTENCE_INDEX, pickle.dumps(index_data))
    chunk_file, chunk_meta_file = ChunkIndex.paths(out / INDEX_DIR)
    buf = io.BytesIO()
    np.savez(buf, ids=np.array([r["id"] for r in records]), embeddings=chunk_matrix)
    _atomic_write(chunk_file, buf.getvalue())
    chunks_sha = hashlib.sha256(chunks_bytes).hexdigest()
    chunk_meta = ChunkIndexMeta(
        model=embedder.model_name,
        chunks_file=str(chunks_path),
        chunks_sha256=chunks_sha,
        n=len(records),
        dim=int(chunk_matrix.shape[1]),
        created=date.today().isoformat(),
    )
    _atomic_write(chunk_meta_file, (chunk_meta.model_dump_json(indent=2) + "\n").encode())
    manifest = Manifest(
        corpus=corpus,
        source=source,
        loader=loader,
        chunker=chunker,
        embedding_model=embedder.model_name,
        embedding_dtype=embedder.dtype,
        sentence_split=SPLITTER_IDS[sentence_split],
        chunks_sha256=chunks_sha,
        built_at=datetime.now().isoformat(timespec="seconds"),
        docs=entries,
    )
    _atomic_write(out / MANIFEST_FILE, (manifest.model_dump_json(indent=1) + "\n").encode())
    report.seconds = round(time.perf_counter() - start, 1)
    for name, value in _peak_memory().items():
        setattr(report, name, value)
    _atomic_write(out / REPORT_FILE, (report.model_dump_json(indent=1) + "\n").encode())
    return report


class BuildComparison(BaseModel):
    chunks_identical: bool | None  # None when one side has no chunks.json (upstream output)
    sentences_identical: bool  # sentences, sentence_to_chunk, chunk lookup and model name
    sentence_max_abs_diff: float
    chunk_ids_identical: bool | None  # None when one side has no Naive RAG chunk index
    chunk_max_abs_diff: float | None

    @property
    def equivalent(self) -> bool:
        """Texts identical and vectors equal up to batching noise of the embedding model."""
        tol = 1e-4
        return (
            self.chunks_identical is not False
            and self.sentences_identical
            and self.chunk_ids_identical is not False
            and self.sentence_max_abs_diff <= tol
            and (self.chunk_max_abs_diff is None or self.chunk_max_abs_diff <= tol)
        )


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).max()) if a.size else 0.0


def _load_sentence_index(build: Path) -> dict[str, Any]:
    with open(build / INDEX_DIR / SENTENCE_INDEX, "rb") as f:
        data: dict[str, Any] = pickle.load(f)
    return data


def compare_builds(a: str | Path, b: str | Path) -> BuildComparison:
    """Is build ``a`` the same as build ``b``? Also accepts an upstream build_index.py output
    directory (an ``index/`` with only ``sentence_index.pkl``)."""
    a, b = Path(a), Path(b)
    sa, sb = _load_sentence_index(a), _load_sentence_index(b)
    chunks_identical: bool | None = None
    if (a / CHUNKS_FILE).exists() and (b / CHUNKS_FILE).exists():
        chunks_identical = (a / CHUNKS_FILE).read_bytes() == (b / CHUNKS_FILE).read_bytes()
    ids_same: bool | None = None
    chunk_diff: float | None = None
    ca, _ = ChunkIndex.paths(a / INDEX_DIR)
    cb, _ = ChunkIndex.paths(b / INDEX_DIR)
    if ca.exists() and cb.exists():
        with np.load(ca) as da, np.load(cb) as db:
            ids_same = da["ids"].tolist() == db["ids"].tolist()
            chunk_diff = _max_abs_diff(da["embeddings"], db["embeddings"])
    keys = ("sentences", "sentence_to_chunk", "chunks", "model_name")
    return BuildComparison(
        chunks_identical=chunks_identical,
        sentences_identical=all(sa[k] == sb[k] for k in keys),
        sentence_max_abs_diff=_max_abs_diff(
            np.asarray(sa["embeddings"]), np.asarray(sb["embeddings"])
        ),
        chunk_ids_identical=ids_same,
        chunk_max_abs_diff=chunk_diff,
    )
