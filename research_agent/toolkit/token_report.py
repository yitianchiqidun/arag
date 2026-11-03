"""Tool-definition tokens per intent, scoped vs everything exposed (stage 11, offline, no API).

    python -m research_agent.toolkit.token_report --out docs/results/stage11_tool_tokens.md

What each intent's LLM calls receive:

- param_lookup / research: the A-RAG agent behind the document channel, with its three
  retrieval tools (taken from a recorded stage 4 trajectory, exactly as sent);
- diagnosis: the diagnosis tools of the registry (``ToolHarness.schemas("diagnosis")``);
- out_of_scope: no tools (only the router's call).

"Everything exposed" gives every one of these calls the union of all tool definitions. Tokens
are counted with the Qwen3 tokenizer of the local embedding model (same family as the agent).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research_agent.toolkit.specs import exposed

TRAJECTORIES = "results/stage4/tidb_baseline_126/arag_full/trajectories.jsonl"
# Average LLM calls per question with tools, from measured runs (for the per-question estimate).
CALLS = {"param_lookup": 3.48, "research": 3.48, "diagnosis": None, "out_of_scope": 1.0}


def arag_tools(path: str = TRAJECTORIES) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return list(json.loads(f.readline())["tools"])


def count(tokenizer: Any, tools: list[dict[str, Any]]) -> int:
    return len(tokenizer.encode(json.dumps(tools, ensure_ascii=False))) if tools else 0


def report(tokenizer: Any, diagnosis_calls: float, arag: list[dict[str, Any]]) -> str:
    diagnosis = [s.schema() for s in exposed("diagnosis")]
    scoped = {"param_lookup": arag, "research": arag, "diagnosis": diagnosis, "out_of_scope": []}
    every = arag + diagnosis
    total = count(tokenizer, every)
    calls = {**CALLS, "diagnosis": diagnosis_calls}
    lines = [
        "| 意图 | 暴露的工具 | 每次调用的工具定义 token（按意图） | 全部暴露 | 每次少 | 每题调用次数 | 每题少 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for intent, tools in scoped.items():
        n = count(tokenizer, tools)
        per_q = calls[intent] or 0
        names = "、".join(t["function"]["name"] for t in tools) or "无"
        lines.append(
            f"| {intent} | {len(tools)} 个：{names} | {n:,} | {total:,} | {total - n:,} | {per_q:g} | {(total - n) * per_q:,.0f} |"
        )
    lines.append(
        f"\n全部暴露时每次调用带 {len(every)} 个工具定义，共 {total:,} token（Qwen3 分词器）。"
        "每题调用次数：参数和研究类取阶段 4 A-RAG 的平均轮数，越界只有分类器的一次调用，"
        "诊断类取阶段 11 场景运行的平均 LLM 调用数。工具定义在请求前部，通常能命中缓存，"
        "所以省下的主要是上下文长度和首字延迟，花费的节省小于 token 数的比例。"
    )
    return "\n".join(lines)


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(prog="python -m research_agent.toolkit.token_report")
    parser.add_argument("--diagnosis-calls", type=float, default=8.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", local_files_only=True)
    text = report(tokenizer, args.diagnosis_calls, arag_tools())
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
