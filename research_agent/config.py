"""Typed run configuration.

The YAML layout is a superset of A-RAG's ``configs/example.yaml`` (sections ``llm``,
``embedding``, ``agent``, ``data``), so the same file also works with the upstream
``scripts/batch_runner.py``. Extra sections (``judge``, ``pricing_file``) are ignored upstream.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from arag.core.config import Config

ToolName = Literal["keyword_search", "semantic_search", "read_chunk"]
ALL_TOOLS: tuple[ToolName, ...] = ("keyword_search", "semantic_search", "read_chunk")


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    base_url: str | None = None  # falls back to ARAG_BASE_URL
    temperature: float = 0.0
    max_tokens: int = 16384
    enable_thinking: bool | None = False  # Qwen hybrid models; None = do not send the flag
    timeout_s: float = 300.0
    max_retries: int = Field(default=4, ge=0)

    def extra_body(self) -> dict[str, Any]:
        return {} if self.enable_thinking is None else {"enable_thinking": self.enable_thinking}

    def resolved_base_url(self) -> str:
        url = self.base_url or os.getenv("ARAG_BASE_URL")
        if not url:
            raise ValueError("base_url not configured: set llm.base_url or ARAG_BASE_URL")
        return url


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str | None = "cuda:0"
    batch_size: int = 16


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_loops: int = Field(default=15, ge=1)
    max_token_budget: int = Field(default=128000, ge=1)
    # Ablation switches. Defaults reproduce A-RAG (Full).
    tools: list[ToolName] = Field(default_factory=lambda: list(ALL_TOOLS))
    # "full_chunk" = the paper's "w/o Chunk Read": searches return whole chunks instead of
    # snippets, and read_chunk is removed.
    search_result: Literal["snippet", "full_chunk"] = "snippet"

    @model_validator(mode="after")
    def _check_tools(self) -> AgentConfig:
        if len(set(self.tools)) != len(self.tools):
            raise ValueError(f"duplicate tools: {self.tools}")
        if not {"keyword_search", "semantic_search"} & set(self.tools):
            raise ValueError("at least one search tool (keyword_search/semantic_search) is needed")
        has_read = "read_chunk" in self.tools
        if has_read and self.search_result == "full_chunk":
            raise ValueError("search_result=full_chunk replaces read_chunk; remove it from tools")
        if not has_read and self.search_result == "snippet":
            raise ValueError(
                "without read_chunk the agent only sees snippets; set search_result: full_chunk"
            )
        return self


class NaiveRAGConfig(BaseModel):
    """Paper baseline: one dense retrieval over whole chunks, then one LLM call."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=5, ge=1)
    # Ask for the supporting chunk ids in the answer, with A-RAG's own wording ("Cite the
    # specific chunks that support your answer"). Off by default: the paper's prompt has no
    # such line, and stage 1 used it as is.
    cite_chunks: bool = False


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks_file: str
    index_dir: str
    # How keyword_search cuts snippets into sentences; "multilingual" for Chinese corpora
    # (research_agent/ingest/sentences.py). The sentence index is split when it is built.
    sentence_split: Literal["upstream", "multilingual"] = "upstream"


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["arag", "naive_rag"] = "arag"
    llm: LLMConfig
    judge: LLMConfig
    embedding: EmbeddingConfig = EmbeddingConfig()
    agent: AgentConfig = AgentConfig()
    naive_rag: NaiveRAGConfig = NaiveRAGConfig()
    data: DataConfig
    pricing_file: str = "configs/pricing_bailian.yaml"

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    def to_arag_config(self) -> Config:
        """Dict-backed config in the shape A-RAG's BatchRunner expects."""
        return Config(self.model_dump(mode="json"))
