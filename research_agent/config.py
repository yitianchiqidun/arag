"""Typed run configuration.

The YAML layout is a superset of A-RAG's ``configs/example.yaml`` (sections ``llm``,
``embedding``, ``agent``, ``data``), so the same file also works with the upstream
``scripts/batch_runner.py``. Extra sections (``judge``, ``pricing_file``) are ignored upstream.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from arag.core.config import Config


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


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks_file: str
    index_dir: str


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LLMConfig
    judge: LLMConfig
    embedding: EmbeddingConfig = EmbeddingConfig()
    agent: AgentConfig = AgentConfig()
    data: DataConfig
    pricing_file: str = "configs/pricing_bailian.yaml"

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    def to_arag_config(self) -> Config:
        """Dict-backed config in the shape A-RAG's BatchRunner expects."""
        return Config(self.model_dump(mode="json"))
