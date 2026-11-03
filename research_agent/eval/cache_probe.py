"""Does the forced final answer reuse the prompt cache the agent loop warmed? (stage 8, D33)

Upstream ``BaseAgent._force_final_answer`` sends the conversation without tools. Bailian's
implicit cache matches prompt prefixes, and the tool definitions are part of the prefix, so
the forced call starts cold: in the stage 2 MuSiQue run the 25 forced answers had 0 cached
tokens, while the loop calls right before them were 90% cached.

The probe replays recorded conversations that ended in a forced answer. For each one it sends
the last loop request again (which warms the cache), then the forced request in one variant.
Every conversation is used by one variant only, so the variants cannot warm each other's
prefixes.

    python -m research_agent.eval.cache_probe results/stage2/musique100_full_traced \\
        --per-variant 5 --out results/stage8/cache_probe
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from research_agent.config import RunConfig
from research_agent.llm.client import MeteredLLMClient, client_from_config
from research_agent.llm.pricing import PricingTable
from research_agent.tracing.schema import TRAJECTORIES_FILE

# Upstream's forced-answer prompt (src/arag/agent/base.py, _force_final_answer).
FORCE_PROMPT = (
    "You have reached the limit. "
    "You MUST now provide a final answer based on the information you have gathered so far. "
    "Do NOT call any more tools. Synthesize the available information and respond directly."
)
# variant -> (send the tools, tool_choice)
VARIANTS: dict[str, tuple[bool, str | None]] = {
    "no_tools": (False, None),  # upstream
    "tool_choice_none": (True, "none"),
    "tools_auto": (True, None),
}


class ProbeResult(BaseModel):
    question_id: str
    variant: str
    warm_prompt_tokens: int
    warm_cached_tokens: int
    forced_prompt_tokens: int
    forced_cached_tokens: int
    forced_cost: float
    warm_cost: float
    forced_called_tools: bool  # the model ignored the instruction and asked for a tool
    answer: str


def loop_prefix(trajectory: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The messages of the last loop request: everything before the forced-answer prompt."""
    messages = trajectory.get("messages") or []
    for i, m in enumerate(messages):
        if i > 1 and m.get("role") == "user" and m.get("content") == FORCE_PROMPT:
            return messages[:i]
    return None


def cases(trajectories: list[dict[str, Any]], per_variant: int) -> list[tuple[str, dict[str, Any]]]:
    """(variant, trajectory) pairs: forced-answer conversations in question-id order, dealt
    to the variants in turn."""
    forced = sorted(
        (t for t in trajectories if loop_prefix(t) is not None and t.get("tools")),
        key=lambda t: str(t["question_id"]),
    )
    names = list(VARIANTS)
    return [(names[i % len(names)], t) for i, t in enumerate(forced[: per_variant * len(names)])]


def probe_one(client: MeteredLLMClient, variant: str, trajectory: dict[str, Any]) -> ProbeResult:
    prefix = loop_prefix(trajectory)
    assert prefix is not None
    tools = trajectory["tools"]
    client.chat(prefix, tools=tools, temperature=0.0, max_tokens=1)  # the loop request again
    send_tools, tool_choice = VARIANTS[variant]
    forced = client.chat(
        [*prefix, {"role": "user", "content": FORCE_PROMPT}],
        tools=tools if send_tools else None,
        temperature=0.0,
        tool_choice=tool_choice,
    )
    warm, final = client.records[-2], client.records[-1]
    return ProbeResult(
        question_id=str(trajectory["question_id"]),
        variant=variant,
        warm_prompt_tokens=warm.prompt_tokens,
        warm_cached_tokens=warm.cached_tokens,
        forced_prompt_tokens=final.prompt_tokens,
        forced_cached_tokens=final.cached_tokens,
        forced_cost=final.cost,
        warm_cost=warm.cost,
        forced_called_tools=bool(forced["message"].get("tool_calls")),
        answer=str(forced["message"].get("content") or "")[:300],
    )


def summary_markdown(results: list[ProbeResult]) -> str:
    lines = [
        "| Variant | n | Forced prompt tokens (avg) | Cached share | Forced cost (avg) | Called a tool |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        rs = [r for r in results if r.variant == variant]
        if not rs:
            continue
        prompt = sum(r.forced_prompt_tokens for r in rs)
        cached = sum(r.forced_cached_tokens for r in rs)
        lines.append(
            f"| {variant} | {len(rs)} | {prompt / len(rs):,.0f} | {cached / prompt:.0%} | "
            f"{sum(r.forced_cost for r in rs) / len(rs):.4f} | "
            f"{sum(r.forced_called_tools for r in rs)} |"
        )
    return "\n".join(lines)


def run_probe(
    trajectories: list[dict[str, Any]],
    new_client: Callable[[], MeteredLLMClient],
    per_variant: int,
    out: Path,
) -> list[ProbeResult]:
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for variant, trajectory in cases(trajectories, per_variant):
        client = new_client()
        results.append(probe_one(client, variant, trajectory))
    (out / "results.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8"
    )
    (out / "summary.md").write_text(summary_markdown(results) + "\n", encoding="utf-8")
    return results


def main() -> None:
    from research_agent.upstream import load_env

    parser = argparse.ArgumentParser(prog="python -m research_agent.eval.cache_probe")
    parser.add_argument("run_dir", help="a run with recorded trajectories (same model is used)")
    parser.add_argument("--per-variant", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    load_env()
    run_dir = Path(args.run_dir)
    cfg = RunConfig.model_validate(json.loads((run_dir / "run_meta.json").read_text())["config"])
    pricing = PricingTable.from_yaml(cfg.pricing_file)
    path = run_dir / TRAJECTORIES_FILE
    trajectories = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    results = run_probe(
        trajectories, lambda: client_from_config(cfg.llm, pricing), args.per_variant, Path(args.out)
    )
    print(summary_markdown(results))
    print(f"cost {sum(r.forced_cost + r.warm_cost for r in results):.4f}")


if __name__ == "__main__":
    main()
