"""Score predictions with A-RAG's own Evaluator, using a fixed, metered judge model."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from research_agent.config import RunConfig
from research_agent.llm.client import UsageSummary, client_from_config
from research_agent.llm.pricing import PricingTable
from research_agent.upstream import load_script


class JudgeMeta(BaseModel):
    judge_model: str
    judge_snapshot: str
    currency: str
    usage: UsageSummary
    # Upstream scores a prediction correct only if the reply is exactly "correct" after
    # strip().lower(); the distribution shows whether any reply was mis-scored by format.
    response_counts: dict[str, int]
    llm_accuracy: float
    contain_accuracy: float


def judge_predictions(
    predictions_path: str | Path, cfg: RunConfig, pricing: PricingTable, workers: int = 5
) -> JudgeMeta:
    """Adds ``llm_accuracy``/``contain_accuracy``/``status`` to each record in place
    (upstream behaviour) and writes ``judge_meta.json`` next to the predictions."""
    evaluator_cls = load_script("eval").Evaluator
    client = client_from_config(cfg.judge, pricing)
    evaluator = evaluator_cls(client, str(predictions_path))
    llm_acc, contain_acc = evaluator.evaluate(max_workers=workers)

    responses = Counter(
        (r.response_text or "").strip().lower() for r in client.records if r.error is None
    )
    meta = JudgeMeta(
        judge_model=cfg.judge.model,
        judge_snapshot=pricing.resolve(cfg.judge.model),
        currency=pricing.currency,
        usage=client.usage_summary(),
        response_counts=dict(responses.most_common(20)),
        llm_accuracy=llm_acc,
        contain_accuracy=contain_acc,
    )
    out = Path(predictions_path).parent / "judge_meta.json"
    out.write_text(json.dumps(meta.model_dump(), ensure_ascii=False, indent=2) + "\n", "utf-8")
    return meta
