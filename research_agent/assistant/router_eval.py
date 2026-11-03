"""Router test set and evaluation (stage 8, D35): only the intent classifier, no agent.

The set (``evalsets/router/questions.json``) has four groups:

- ``symptom``: business symptoms (slow, errors, timeouts) that never name TiDB -> diagnosis;
- ``unrelated``: questions unrelated to the database, some of them device or network
  problems that look like symptoms -> out_of_scope;
- ``param`` / ``version``: questions sampled from the stage 4 evalset -> param_lookup /
  research (their classification should not change when the router prompt changes).

    python -m research_agent.assistant.router_eval build
    python -m research_agent.assistant.router_eval run --out results/stage8/router_before
    python -m research_agent.assistant.router_eval run --out results/stage8/router_after \\
        --before results/stage8/router_before
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from research_agent.assistant import prompts
from research_agent.assistant.routing import ChatClient, classify

HANDWRITTEN = Path("evalsets/router/handwritten.yaml")
EVALSET = Path("evalsets/tidb/questions.json")
QUESTIONS = Path("evalsets/router/questions.json")
EXPECTED = {
    "symptom": "diagnosis",
    "unrelated": "out_of_scope",
    "param": "param_lookup",
    "version": "research",
}
SEED = 2026


class RouterQuestion(BaseModel):
    id: str
    group: str
    question: str
    expected: str
    source: str


class RouterResult(BaseModel):
    id: str
    group: str
    question: str
    expected: str
    intent: str
    reason: str
    cost: float


def build(
    handwritten: Path = HANDWRITTEN, evalset: Path = EVALSET, per_group: int = 10
) -> list[RouterQuestion]:
    """The hand-written questions plus ``per_group`` parameter questions (half direct, half
    descriptive) and ``per_group`` version questions of the evalset, sampled with a fixed seed."""
    hand = yaml.safe_load(handwritten.read_text(encoding="utf-8"))
    out = [
        RouterQuestion(
            id=f"{group}_{i:02d}",
            group=group,
            question=q,
            expected=EXPECTED[group],
            source=str(handwritten),
        )
        for key, group in (("diagnosis", "symptom"), ("out_of_scope", "unrelated"))
        for i, q in enumerate(hand[key], 1)
    ]
    items = json.loads(evalset.read_text(encoding="utf-8"))
    rng = random.Random(SEED)

    def pick(kind: str, n: int) -> list[dict[str, Any]]:
        pool = sorted((q for q in items if q["question_type"] == kind), key=lambda q: q["id"])
        return rng.sample(pool, n)

    params = pick("direct", per_group // 2) + pick("descriptive", per_group - per_group // 2)
    for group, chosen in (("param", params), ("version", pick("version", per_group))):
        out += [
            RouterQuestion(
                id=q["id"],
                group=group,
                question=q["question"],
                expected=EXPECTED[group],
                source=str(evalset),
            )
            for q in chosen
        ]
    return out


def evaluate(client: ChatClient, questions: list[RouterQuestion]) -> list[RouterResult]:
    results = []
    for q in questions:
        route = classify(client, q.question)
        results.append(
            RouterResult(
                id=q.id,
                group=q.group,
                question=q.question,
                expected=q.expected,
                intent=route.intent,
                reason=route.reason,
                cost=route.cost,
            )
        )
    return results


def summary_markdown(results: list[RouterResult], before: list[RouterResult] | None = None) -> str:
    previous = {r.id: r.intent for r in before or []}
    lines = [
        "| 组 | 期望 | 题数 | 准确率 | 分类分布 |" + (" 与修改前相同 |" if before else ""),
        "| --- | --- | ---: | ---: | --- |" + (" ---: |" if before else ""),
    ]
    for group, expected in EXPECTED.items():
        rs = [r for r in results if r.group == group]
        if not rs:
            continue
        right = sum(r.intent == expected for r in rs)
        dist = ", ".join(f"{k} {v}" for k, v in Counter(r.intent for r in rs).most_common())
        row = f"| {group} | {expected} | {len(rs)} | {right / len(rs):.0%}（{right}） | {dist} |"
        if before:
            same = sum(previous.get(r.id) == r.intent for r in rs)
            row += f" {same} / {len(rs)} |"
        lines.append(row)
    lines.append(f"\n花费 {sum(r.cost for r in results):.4f} 元。")
    return "\n".join(lines)


def run(out: Path, client: ChatClient, before_dir: Path | None = None) -> list[RouterResult]:
    questions = [RouterQuestion.model_validate(q) for q in json.loads(QUESTIONS.read_text("utf-8"))]
    results = evaluate(client, questions)
    before = None
    if before_dir is not None:
        rows = (before_dir / "results.jsonl").read_text("utf-8").splitlines()
        before = [RouterResult.model_validate_json(line) for line in rows]
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in results), "utf-8"
    )
    meta = {
        "prompt_sha256": hashlib.sha256(prompts.ROUTE.encode()).hexdigest()[:12],
        "prompt": prompts.ROUTE,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", "utf-8")
    (out / "summary.md").write_text(summary_markdown(results, before) + "\n", "utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m research_agent.assistant.router_eval")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--before", default=None, help="an earlier run, to count unchanged intents")
    r.add_argument("--config", default="configs/assistant.yaml")
    args = parser.parse_args()
    if args.command == "build":
        questions = build()
        QUESTIONS.write_text(
            json.dumps([q.model_dump() for q in questions], ensure_ascii=False, indent=1) + "\n",
            "utf-8",
        )
        print(Counter(q.group for q in questions))
        return
    from research_agent.assistant.app import AssistantConfig
    from research_agent.config import RunConfig
    from research_agent.llm.client import client_from_config
    from research_agent.llm.pricing import PricingTable
    from research_agent.upstream import load_env

    load_env()
    cfg = AssistantConfig.from_yaml(args.config)
    pricing = PricingTable.from_yaml(RunConfig.from_yaml(cfg.run_config).pricing_file)
    out = Path(args.out)
    run(out, client_from_config(cfg.router, pricing), Path(args.before) if args.before else None)
    print((out / "summary.md").read_text("utf-8"))


if __name__ == "__main__":
    main()
