"""CLI: ``python -m research_agent.evalset {stats,closed-book,prereview,finalize}``.

Candidates come from a builder (``dataset.write_candidates``); then:

- ``stats``: candidate statuses per group, reject reasons, cost, review progress
- ``closed-book``: the closed-book filter for every candidate without a current result
  (API calls)
- ``prereview``: LLM pre-review of the groups that use it; ``--estimate`` only counts tokens
- ``finalize``: questions.json from the human decisions, the pre-review and the review rules
"""

from __future__ import annotations

import argparse
from collections import Counter

from research_agent.evalset.dataset import EvalsetConfig, stats
from research_agent.upstream import load_env


def cmd_stats(args: argparse.Namespace) -> None:
    print(stats(EvalsetConfig.from_yaml(args.config)).model_dump_json(indent=1))


def cmd_closed_book(args: argparse.Namespace) -> None:
    from research_agent.evalset.closed_book import run_closed_book

    load_env()
    cfg = EvalsetConfig.from_yaml(args.config)
    new = run_closed_book(cfg, args.workers)
    print(f"{len(new)} closed-book checks, cost {sum(r.cost for r in new):.4f}")
    print(stats(cfg).model_dump_json(indent=1))


def cmd_prereview(args: argparse.Namespace) -> None:
    from research_agent.evalset.review import estimate_prereview, run_prereview
    from research_agent.ingest.chunker import tiktoken_counter

    cfg = EvalsetConfig.from_yaml(args.config)
    if args.estimate:
        print(estimate_prereview(cfg, tiktoken_counter()).model_dump_json(indent=1))
        return
    load_env()
    new = run_prereview(cfg, args.workers)
    decisions = Counter(str(p.decision) for p in new)
    print(f"{len(new)} pre-reviews {dict(decisions)}, cost {sum(p.cost for p in new):.4f}")


def cmd_finalize(args: argparse.Namespace) -> None:
    from research_agent.evalset.review import finalize

    summary = finalize(EvalsetConfig.from_yaml(args.config))
    print(summary.model_dump_json(indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m research_agent.evalset")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("stats", cmd_stats), ("finalize", cmd_finalize)):
        p = sub.add_parser(name)
        p.add_argument("config")
        p.set_defaults(func=func)
    p = sub.add_parser("closed-book")
    p.add_argument("config")
    p.add_argument("--workers", type=int, default=5)
    p.set_defaults(func=cmd_closed_book)
    p = sub.add_parser("prereview")
    p.add_argument("config")
    p.add_argument("--estimate", action="store_true", help="count tokens and cost, no API calls")
    p.add_argument("--workers", type=int, default=5)
    p.set_defaults(func=cmd_prereview)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
