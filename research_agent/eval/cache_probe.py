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
import time
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
# variant -> (send the tools, tool_choice, where the instruction goes: "user" message or
# appended to the last "tool" result). Qwen3-style chat templates render the assistant turns
# before the last user message differently from those after it, so a new user message may
# change the whole conversation after the question, not just its end.
VARIANTS: dict[str, tuple[bool, str | None, str]] = {
    "no_tools": (False, None, "user"),  # upstream
    "tool_choice_none": (True, "none", "user"),
    "tools_auto": (True, None, "user"),
    "tools_auto_in_tool_result": (True, None, "tool"),
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


def cases(
    trajectories: list[dict[str, Any]],
    per_variant: int,
    variants: list[str] | None = None,
    skip: int = 0,
) -> list[tuple[str, dict[str, Any]]]:
    """(variant, trajectory) pairs: forced-answer conversations in question-id order, after the
    first ``skip`` (used by an earlier probe, so their prefixes may be cached), dealt to the
    variants in turn."""
    forced = sorted(
        (t for t in trajectories if loop_prefix(t) is not None and t.get("tools")),
        key=lambda t: str(t["question_id"]),
    )[skip:]
    names = variants or list(VARIANTS)
    return [(names[i % len(names)], t) for i, t in enumerate(forced[: per_variant * len(names)])]


def force_messages(prefix: list[dict[str, Any]], place: str) -> list[dict[str, Any]]:
    """The forced request: the instruction as a new user message, or appended to the last
    tool result (which must end the prefix)."""
    if place == "user":
        return [*prefix, {"role": "user", "content": FORCE_PROMPT}]
    last = prefix[-1]
    if last.get("role") != "tool":
        raise ValueError("the loop prefix does not end with a tool result")
    return [*prefix[:-1], {**last, "content": f"{last.get('content') or ''}\n\n{FORCE_PROMPT}"}]


def probe_one(
    client: MeteredLLMClient,
    variant: str,
    trajectory: dict[str, Any],
    warm_max_tokens: int | None = 1,
    pause_s: float = 0.0,
) -> ProbeResult:
    """``warm_max_tokens`` None lets the warm-up generate as the loop did (its tool calls);
    ``pause_s`` stands for the tool execution between the loop's last call and the forced one."""
    prefix = loop_prefix(trajectory)
    assert prefix is not None
    tools = trajectory["tools"]
    # The loop request again (max_tokens 0 or None: the config's limit).
    client.chat(prefix, tools=tools, temperature=0.0, max_tokens=warm_max_tokens)
    time.sleep(pause_s)
    send_tools, tool_choice, place = VARIANTS[variant]
    forced = client.chat(
        force_messages(prefix, place),
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
    variants: list[str] | None = None,
    skip: int = 0,
    warm_max_tokens: int | None = 1,
    pause_s: float = 0.0,
) -> list[ProbeResult]:
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for variant, trajectory in cases(trajectories, per_variant, variants, skip):
        client = new_client()
        results.append(probe_one(client, variant, trajectory, warm_max_tokens, pause_s))
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
    parser.add_argument("--variants", default=None, help=f"comma-separated, of {list(VARIANTS)}")
    parser.add_argument("--skip", type=int, default=0, help="conversations to skip (used before)")
    parser.add_argument(
        "--warm-max-tokens", type=int, default=1, help="0: generate as the loop did"
    )
    parser.add_argument("--pause-s", type=float, default=0.0)
    args = parser.parse_args()
    load_env()
    run_dir = Path(args.run_dir)
    cfg = RunConfig.model_validate(json.loads((run_dir / "run_meta.json").read_text())["config"])
    pricing = PricingTable.from_yaml(cfg.pricing_file)
    path = run_dir / TRAJECTORIES_FILE
    trajectories = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    results = run_probe(
        trajectories,
        lambda: client_from_config(cfg.llm, pricing),
        args.per_variant,
        Path(args.out),
        args.variants.split(",") if args.variants else None,
        args.skip,
        args.warm_max_tokens or None,
        args.pause_s,
    )
    print(summary_markdown(results))
    print(f"cost {sum(r.forced_cost + r.warm_cost for r in results):.4f}")


if __name__ == "__main__":
    main()
