from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

from arag.core.context import AgentContext
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.read_chunk import ReadChunkTool
from research_agent.baselines.naive_rag import ChunkIndexMeta
from research_agent.ingest.chunker import ChunkerConfig
from research_agent.ingest.index import (
    MANIFEST_FILE,
    build_index,
    compare_builds,
    doc_changes,
)
from research_agent.ingest.loaders import Document, LoaderConfig
from research_agent.ingest.sentences import SPLITTER_IDS, SentenceSplit, split_multilingual
from research_agent.upstream import load_script


class FakeEmbedder:
    """Deterministic unit vectors from the text hash; remembers what it was asked to embed."""

    def __init__(self, model_name: str = "fake-embed", dtype: str = "float32") -> None:
        self.model_name = model_name
        self.dtype = dtype
        self.calls: list[list[str]] = []
        self.batch_sizes: list[int | None] = []

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        self.calls.append(list(texts))
        self.batch_sizes.append(batch_size)
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "little")
            v = np.random.default_rng(seed).normal(size=8)
            rows.append(v / np.linalg.norm(v))
        return np.asarray(rows, dtype=np.float32)

    @property
    def embedded(self) -> list[str]:
        return [t for call in self.calls for t in call]


def words(text: str) -> int:
    return len(text.split())


def doc(path: str, text: str) -> Document:
    return Document(
        path=path,
        title=path,
        text=text,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


V1 = [
    doc("docs/a.md", "# A\nAlpha serves models. It uses paged attention.\n"),
    doc("docs/b.md", "# B\nBeta explains tensor parallelism in detail.\n"),
    doc("docs/c.md", "# C\nGamma covers quantization formats today.\n"),
]
V2 = [
    V1[0],
    doc("docs/b.md", "# B\nBeta explains tensor parallelism in detail.\nNew pipeline note here.\n"),
    doc("docs/d.md", "# D\nDelta documents the metrics endpoint.\n"),
]
LOADER = LoaderConfig(include=["docs/*.md"])
CHUNKER = ChunkerConfig(max_tokens=50)


def build(
    docs: list[Document],
    out: Path,
    embedder: FakeEmbedder,
    chunker: ChunkerConfig = CHUNKER,
    sentence_split: SentenceSplit = "upstream",
    chunk_batch_size: int | None = None,
):
    return build_index(
        docs,
        out,
        corpus="test",
        source={"kind": "test"},
        loader=LOADER,
        chunker=chunker,
        embedder=embedder,
        count_tokens=words,
        sentence_split=sentence_split,
        chunk_batch_size=chunk_batch_size,
    )


def test_full_build_writes_the_upstream_format(tmp_path: Path) -> None:
    out = tmp_path / "build"
    report = build(V1, out, FakeEmbedder())
    assert report.mode == "full" and report.docs_added == ["docs/a.md", "docs/b.md", "docs/c.md"]
    assert report.peak_rss_mib is not None and report.peak_rss_mib > 0

    records = json.loads((out / "chunks.json").read_text())
    assert [r["id"] for r in records] == ["0", "1", "2"]
    assert records[0]["doc"] == "docs/a.md" and records[0]["headings"] == ["A"]
    with open(out / "index/sentence_index.pkl", "rb") as f:
        index = pickle.load(f)
    split = load_script("build_index").split_sentences
    expected = [s for r in records for s in split(r["text"])]
    assert index["sentences"] == expected and index["model_name"] == "fake-embed"
    assert index["embeddings"].shape == (len(expected), 8)
    assert index["chunks"]["1"]["text"] == records[1]["text"]
    meta = ChunkIndexMeta.model_validate_json((out / "index/chunk_index.meta.json").read_text())
    assert (
        meta.n == 3
        and meta.chunks_sha256 == hashlib.sha256((out / "chunks.json").read_bytes()).hexdigest()
    )

    # The upstream tools read the build directly.
    reader = ReadChunkTool(str(out / "chunks.json"))
    text, _ = reader.execute(AgentContext(), chunk_ids=["2"])
    assert "[docs/c.md] C" in text and "quantization" in text
    result, log = KeywordSearchTool(str(out / "chunks.json")).execute(AgentContext(), ["tensor"])
    assert result.startswith("Chunk ID: 1,") and log["chunks_found"] == 1


def test_incremental_build_equals_a_full_build_and_embeds_only_new_text(tmp_path: Path) -> None:
    inc, full = tmp_path / "inc", tmp_path / "full"
    build(V1, inc, FakeEmbedder())
    embedder = FakeEmbedder()
    report = build(V2, inc, embedder)

    assert report.mode == "incremental"
    assert (report.docs_added, report.docs_modified, report.docs_deleted) == (
        ["docs/d.md"],
        ["docs/b.md"],
        ["docs/c.md"],
    )
    assert report.docs_unchanged == 1 and report.chunks_reused == 1 and report.chunks_rechunked == 2
    assert report.chunks_new_text == 2 and report.chunk_vectors_reused == 1
    assert set(embedder.embedded) >= {"New pipeline note here"}
    assert "Alpha serves models" not in embedder.embedded  # unchanged sentence reused
    assert report.sentence_vectors_embedded == len(set(embedder.embedded)) - 2  # minus 2 chunks

    build(V2, full, FakeEmbedder())
    comparison = compare_builds(inc, full)
    assert comparison.equivalent and comparison.chunks_identical
    assert comparison.sentence_max_abs_diff == 0.0


def test_unchanged_rebuild_embeds_nothing(tmp_path: Path) -> None:
    out = tmp_path / "build"
    build(V1, out, FakeEmbedder())
    embedder = FakeEmbedder()
    report = build(V1, out, embedder)
    assert embedder.calls == [] and report.docs_unchanged == 3 and report.chunks_reused == 3


def test_config_changes_limit_reuse(tmp_path: Path) -> None:
    out = tmp_path / "build"
    build(V1, out, FakeEmbedder())

    for embedder in (FakeEmbedder("fake-embed", "model_default"), FakeEmbedder("other-embed")):
        report = build(V1, out, embedder)
        assert report.sentence_vectors_reused == 0 and report.chunk_vectors_reused == 0
        assert any("embedding model changed" in n for n in report.notes)

    embedder = FakeEmbedder("other-embed")
    report = build(V1, out, embedder, ChunkerConfig(max_tokens=50, source_header=False))
    assert report.chunks_rechunked == 3 and report.chunks_reused == 0
    assert report.sentence_vectors_embedded == 0  # same sentences, only the header lines differ
    assert report.chunk_vectors_embedded == 3


def test_sentence_splitter_is_recorded_and_checked(tmp_path: Path) -> None:
    out = tmp_path / "build"
    build(V1, out, FakeEmbedder())
    report = build(V1, out, FakeEmbedder(), sentence_split="multilingual")
    assert any("sentence split changed" in n for n in report.notes)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["sentence_split"] == SPLITTER_IDS["multilingual"]
    with open(out / "index/sentence_index.pkl", "rb") as f:
        index = pickle.load(f)
    records = json.loads((out / "chunks.json").read_text())
    assert index["sentences"] == [s for r in records for s in split_multilingual(r["text"])]


def test_interrupted_build_falls_back_to_full(tmp_path: Path) -> None:
    out = tmp_path / "build"
    build(V1, out, FakeEmbedder())
    shutil.copy(out / "chunks.json", tmp_path / "old.json")
    (out / "chunks.json").write_text("[]")  # as if the process died after writing chunks
    report = build(V1, out, FakeEmbedder())
    assert report.mode == "full" and "does not match" in report.notes[0]
    assert (out / MANIFEST_FILE).exists()
    assert (out / "chunks.json").read_bytes() == (tmp_path / "old.json").read_bytes()


def test_doc_changes() -> None:
    changes = doc_changes({"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "9", "d": "4"})
    assert (changes.added, changes.modified, changes.deleted, changes.unchanged) == (
        ["d"],
        ["b"],
        ["c"],
        ["a"],
    )


def test_chunk_vectors_use_their_own_batch_size(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    report = build(V1, tmp_path / "build", embedder, chunk_batch_size=4)
    assert embedder.batch_sizes == [None, 4]  # sentences, then chunks
    assert report.chunk_batch_size == 4
