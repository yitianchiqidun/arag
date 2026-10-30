from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_agent.llm.pricing import ModelPricing, PriceTier, PricingTable
from research_agent.upstream import REPO_ROOT

from .fakes import PLUS_PRICING


def test_tier_boundary_is_inclusive() -> None:
    assert PLUS_PRICING.tier_for(131072).input == 0.8
    assert PLUS_PRICING.tier_for(131073).input == 4.8


def test_cost_bills_cached_tokens_at_ratio() -> None:
    cost = PLUS_PRICING.cost(prompt_tokens=10_000, completion_tokens=500, cached_tokens=4_000)
    expected = (6_000 * 0.8 + 4_000 * 0.8 * 0.2 + 500 * 2.0) / 1_000_000
    assert cost == pytest.approx(expected)


def test_cached_tokens_are_capped_by_prompt_tokens() -> None:
    assert PLUS_PRICING.cost(100, 0, cached_tokens=500) == pytest.approx(100 * 0.8 * 0.2 / 1e6)


def test_last_tier_must_be_unbounded() -> None:
    with pytest.raises(ValidationError):
        ModelPricing(tiers=[PriceTier(max_input_tokens=10, input=1, output=1)])


def test_tiers_must_ascend() -> None:
    with pytest.raises(ValidationError):
        ModelPricing(
            tiers=[
                PriceTier(max_input_tokens=20, input=1, output=1),
                PriceTier(max_input_tokens=10, input=1, output=1),
                PriceTier(max_input_tokens=None, input=1, output=1),
            ]
        )


def test_alias_resolution_and_unknown_model() -> None:
    table = PricingTable(currency="CNY", models={"m-2025": PLUS_PRICING}, aliases={"m": "m-2025"})
    assert table.get("m") is table.get("m-2025")
    with pytest.raises(KeyError):
        table.get("other")


def test_alias_to_missing_model_rejected() -> None:
    with pytest.raises(ValidationError):
        PricingTable(currency="CNY", models={}, aliases={"m": "missing"})


def test_shipped_pricing_table_loads() -> None:
    table = PricingTable.from_yaml(REPO_ROOT / "configs" / "pricing_bailian.yaml")
    assert table.currency == "CNY"
    for alias in table.aliases:
        table.get(alias)
