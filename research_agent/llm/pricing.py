"""Token pricing for LLM providers that bill by prompt-length tiers and cached input."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class PriceTier(BaseModel):
    """Prices (per 1M tokens) applying to requests whose prompt has at most ``max_input_tokens``."""

    max_input_tokens: int | None = None
    input: float = Field(ge=0)
    output: float = Field(ge=0)


class ModelPricing(BaseModel):
    """Tiered pricing of one model snapshot."""

    tiers: list[PriceTier] = Field(min_length=1)
    cached_input_ratio: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def _check_tiers(self) -> ModelPricing:
        bounds = [t.max_input_tokens for t in self.tiers]
        if bounds[-1] is not None:
            raise ValueError("last tier must be unbounded (max_input_tokens: null)")
        finite = [b for b in bounds[:-1] if b is not None]
        if len(finite) != len(bounds) - 1 or finite != sorted(finite):
            raise ValueError("tier bounds must be ascending and only the last may be null")
        return self

    def tier_for(self, prompt_tokens: int) -> PriceTier:
        for tier in self.tiers:
            if tier.max_input_tokens is None or prompt_tokens <= tier.max_input_tokens:
                return tier
        raise AssertionError("unreachable: last tier is unbounded")

    def cost(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
        """Cost of one request. ``cached_tokens`` is the part of ``prompt_tokens`` that hit cache."""
        tier = self.tier_for(prompt_tokens)
        cached = min(cached_tokens, prompt_tokens)
        uncached = prompt_tokens - cached
        return (
            uncached * tier.input
            + cached * tier.input * self.cached_input_ratio
            + completion_tokens * tier.output
        ) / 1_000_000


class PricingTable(BaseModel):
    currency: str
    models: dict[str, ModelPricing]
    aliases: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_aliases(self) -> PricingTable:
        missing = {a: t for a, t in self.aliases.items() if t not in self.models}
        if missing:
            raise ValueError(f"aliases point to unknown models: {missing}")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> PricingTable:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    def resolve(self, model: str) -> str:
        """Map an alias to the snapshot name used for pricing."""
        return self.aliases.get(model, model)

    def get(self, model: str) -> ModelPricing:
        name = self.resolve(model)
        if name not in self.models:
            raise KeyError(f"no pricing for model {model!r}; add it to the pricing table")
        return self.models[name]
