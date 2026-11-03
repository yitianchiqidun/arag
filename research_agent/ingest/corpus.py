"""Corpus configuration: where the files are and how to load, chunk and embed them."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from research_agent.config import EmbeddingConfig
from research_agent.ingest.chunker import ChunkerConfig, tiktoken_counter
from research_agent.ingest.index import (
    BuildReport,
    EmbeddingDtype,
    SentenceTransformerEmbedder,
    build_index,
)
from research_agent.ingest.loaders import Document, LoaderConfig, LoadReport, load_corpus
from research_agent.ingest.sources import CorpusSource, DirectorySource, GitSource


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    # Either a git repository (read at a ref given on the command line) or a directory.
    repo_url: str | None = None
    repo_dir: str | None = None
    root_dir: str | None = None
    loader: LoaderConfig
    chunker: ChunkerConfig = ChunkerConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    # Precision of the index vectors (queries keep the upstream default); see
    # SentenceTransformerEmbedder and design_decisions.md D24.
    embedding_dtype: EmbeddingDtype = "float32"

    @model_validator(mode="after")
    def _one_source(self) -> CorpusConfig:
        if (self.repo_dir is None) == (self.root_dir is None):
            raise ValueError("set exactly one of repo_dir (git) and root_dir (directory)")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> CorpusConfig:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    def source(self, ref: str | None = None) -> CorpusSource:
        if self.repo_dir is not None:
            if ref is None:
                raise ValueError(f"corpus {self.name} is a git repository: give a ref")
            return GitSource(self.repo_dir, ref)
        assert self.root_dir is not None
        if ref is not None:
            raise ValueError(f"corpus {self.name} is a directory: refs do not apply")
        return DirectorySource(self.root_dir)

    def load(self, ref: str | None = None) -> tuple[CorpusSource, list[Document], LoadReport]:
        source = self.source(ref)
        docs, report = load_corpus(source, self.loader)
        return source, docs, report

    def build(self, out_dir: str | Path, ref: str | None = None) -> BuildReport:
        source, docs, _ = self.load(ref)
        embedder = SentenceTransformerEmbedder(
            self.embedding.model,
            self.embedding.device,
            self.embedding.batch_size,
            self.embedding_dtype,
        )
        return build_index(
            docs,
            out_dir,
            corpus=self.name,
            source=source.describe(),
            loader=self.loader,
            chunker=self.chunker,
            embedder=embedder,
            count_tokens=tiktoken_counter(self.chunker.tokenizer),
        )
