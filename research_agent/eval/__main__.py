"""CLI: ``python -m research_agent.eval {subset,run,judge,summarize,pipeline}``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_agent.config import RunConfig
from research_agent.llm.pricing import PricingTable
from research_agent.upstream import load_env


def _load(config_path: str) -> tuple[RunConfig, PricingTable]:
    cfg = RunConfig.from_yaml(config_path)
    return cfg, PricingTable.from_yaml(cfg.pricing_file)


def cmd_subset(args: argparse.Namespace) -> None:
    from research_agent.eval.subsets import make_subset

    meta = make_subset(args.source, args.out, args.n, args.seed)
    print(f"wrote {args.out}: {meta.n}/{meta.source_size} questions, seed={meta.seed}")


def cmd_run(args: argparse.Namespace) -> None:
    from research_agent.eval.runner import MeteredBatchRunner

    cfg, pricing = _load(args.config)
    MeteredBatchRunner(
        cfg, pricing, args.questions, args.output, limit=args.limit, num_workers=args.workers
    ).run()


def cmd_judge(args: argparse.Namespace) -> None:
    from research_agent.eval.judge import judge_predictions

    cfg, pricing = _load(args.config)
    meta = judge_predictions(Path(args.output) / "predictions.jsonl", cfg, pricing, args.workers)
    print(json.dumps(meta.model_dump(), ensure_ascii=False, indent=2))


def cmd_summarize(args: argparse.Namespace) -> None:
    from research_agent.eval.summarize import summarize_run, to_markdown

    summary = summarize_run(args.output)
    out = Path(args.output)
    (out / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n", "utf-8")
    (out / "summary.md").write_text(to_markdown(summary) + "\n", "utf-8")
    print(to_markdown(summary))


def cmd_pipeline(args: argparse.Namespace) -> None:
    cmd_run(args)
    cmd_judge(args)
    cmd_summarize(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m research_agent.eval")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("subset", help="sample a fixed question subset")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_subset)

    def add_common(p: argparse.ArgumentParser, needs_config: bool = True) -> None:
        if needs_config:
            p.add_argument("--config", "-c", required=True)
        p.add_argument("--output", "-o", required=True, help="run directory")
        p.add_argument("--workers", "-w", type=int, default=5)

    for name, func, help_text in [
        ("run", cmd_run, "run the agent over a question file"),
        ("pipeline", cmd_pipeline, "run + judge + summarize"),
    ]:
        p = sub.add_parser(name, help=help_text)
        add_common(p)
        p.add_argument("--questions", "-q", required=True)
        p.add_argument("--limit", "-l", type=int, default=None)
        p.set_defaults(func=func)

    p = sub.add_parser("judge", help="score predictions with the configured judge")
    add_common(p)
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("summarize", help="aggregate a judged run directory")
    add_common(p, needs_config=False)
    p.set_defaults(func=cmd_summarize)

    args = parser.parse_args(argv)
    load_env()
    args.func(args)


if __name__ == "__main__":
    main()
