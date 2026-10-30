"""Experiment suites: one YAML describes a whole comparison table.

Every variant is ``base_config`` plus a small ``overrides`` dict, so rows can only differ in
what the overrides change; ``compare.check_comparable`` verifies that again from the run
metadata. Running a suite is resumable: finished runs are skipped, unfinished ones continue
from their checkpoint (upstream BatchRunner skips answered questions).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from research_agent.config import RunConfig
from research_agent.eval.compare import ComparedRun, compare_runs, write_comparison
from research_agent.eval.summarize import load_predictions, write_summary
from research_agent.llm.pricing import PricingTable


class SuiteVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str = ""
    overrides: dict[str, Any] = Field(default_factory=dict)
    # Reuse an existing run instead of running ``<output_root>/<name>``; it is only read.
    run_dir: str | None = None


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    base_config: str
    questions: str
    output_root: str
    reference: str
    variants: list[SuiteVariant] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_names(self) -> Suite:
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate variant names: {names}")
        if self.reference not in names:
            raise ValueError(f"reference {self.reference!r} is not a variant")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Suite:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    def variant(self, name: str) -> SuiteVariant:
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(f"no variant {name!r} in suite {self.name!r}")

    def resolved_config(self, variant: SuiteVariant) -> RunConfig:
        with open(self.base_config, encoding="utf-8") as f:
            base = yaml.safe_load(f)
        return RunConfig.model_validate(deep_merge(base, variant.overrides))

    def run_dir(self, variant: SuiteVariant) -> Path:
        return Path(variant.run_dir) if variant.run_dir else Path(self.output_root) / variant.name


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Dicts merge recursively; any other value (including lists) replaces the base value."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _n_questions(path: str | Path) -> int:
    with open(path, encoding="utf-8") as f:
        return len(json.load(f))


def is_judged(run_dir: Path, n_questions: int) -> bool:
    if not (run_dir / "judge_meta.json").exists() or not (run_dir / "predictions.jsonl").exists():
        return False
    records = load_predictions(run_dir)
    return len(records) == n_questions and all("llm_accuracy" in r for r in records)


def run_variant(suite: Suite, variant: SuiteVariant, workers: int) -> None:
    """Run, judge and summarize one variant, skipping whatever is already done."""
    from research_agent.eval.judge import judge_predictions
    from research_agent.eval.runner import MeteredBatchRunner

    cfg = suite.resolved_config(variant)
    out = suite.run_dir(variant)
    n = _n_questions(suite.questions)
    if variant.run_dir is not None:
        if not is_judged(out, n):
            raise RuntimeError(f"reused run {out} is incomplete; finish it or drop run_dir")
        return
    if is_judged(out, n):
        print(f"[{variant.name}] already judged, skipping")
        write_summary(out)
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True), "utf-8"
    )
    pricing = PricingTable.from_yaml(cfg.pricing_file)
    print(f"[{variant.name}] running {variant.description or variant.overrides}")
    MeteredBatchRunner(cfg, pricing, suite.questions, str(out), num_workers=workers).run()
    judge_predictions(out / "predictions.jsonl", cfg, pricing, workers)
    write_summary(out)


def compare_suite(suite: Suite) -> tuple[list[ComparedRun], str]:
    """Comparison table over the variants that are fully judged (others are listed as missing)."""
    n = _n_questions(suite.questions)
    runs = {v.name: suite.run_dir(v) for v in suite.variants if is_judged(suite.run_dir(v), n)}
    missing = [v.name for v in suite.variants if v.name not in runs]
    if suite.reference not in runs:
        raise RuntimeError(f"reference {suite.reference!r} has no judged run yet")
    rows = compare_runs({name: str(path) for name, path in runs.items()}, suite.reference)
    markdown = write_comparison(rows, suite.reference, Path(suite.output_root) / "comparison")
    if missing:
        markdown += f"\n\nNot run yet: {', '.join(missing)}"
    return rows, markdown


def run_suite(suite: Suite, workers: int = 5, only: list[str] | None = None) -> str:
    for name in only or [v.name for v in suite.variants]:
        run_variant(suite, suite.variant(name), workers)
    return compare_suite(suite)[1]
