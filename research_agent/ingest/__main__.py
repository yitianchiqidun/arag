"""CLI: ``python -m research_agent.ingest {fetch,inspect,diff,build,compare}``.

- ``fetch``: blobless clone (or update) of a git corpus
- ``inspect``: documents and chunks of one version, without embedding anything
- ``diff``: added / modified / deleted documents and chunks between two versions
- ``build``: build or incrementally update an index directory
- ``compare``: check two index directories hold the same index
"""

from __future__ import annotations

import argparse
import json
import statistics

from research_agent.ingest.chunker import chunk_document, tiktoken_counter
from research_agent.ingest.corpus import CorpusConfig
from research_agent.ingest.index import compare_builds, doc_changes


def cmd_fetch(args: argparse.Namespace) -> None:
    from research_agent.ingest.sources import clone_or_fetch

    cfg = CorpusConfig.from_yaml(args.config)
    if cfg.repo_url is None or cfg.repo_dir is None:
        raise SystemExit(f"corpus {cfg.name} has no repo_url/repo_dir to fetch")
    clone_or_fetch(cfg.repo_url, cfg.repo_dir)


def cmd_inspect(args: argparse.Namespace) -> None:
    cfg = CorpusConfig.from_yaml(args.config)
    source, docs, report = cfg.load(args.ref)
    count = tiktoken_counter(cfg.chunker.tokenizer)
    chunks = [c for d in docs for c in chunk_document(d, cfg.chunker, count)]
    sizes = sorted(c.tokens for c in chunks)
    doc_tokens = sorted(count(d.text) for d in docs)
    print(json.dumps(source.describe()))
    print(report.model_dump_json(indent=1))
    print(
        f"documents: {len(docs)}, tokens {sum(doc_tokens):,} (median {statistics.median(doc_tokens):.0f}, max {doc_tokens[-1]:,})"
    )
    print(
        f"chunks: {len(chunks)}, tokens median {statistics.median(sizes):.0f}, "
        f"p10 {sizes[len(sizes) // 10]}, p90 {sizes[9 * len(sizes) // 10]}, max {sizes[-1]}, "
        f"over limit {sum(s > cfg.chunker.max_tokens + 60 for s in sizes)}"
    )


def cmd_diff(args: argparse.Namespace) -> None:
    cfg = CorpusConfig.from_yaml(args.config)
    _, old_docs, _ = cfg.load(args.old)
    _, new_docs, _ = cfg.load(args.new)
    changes = doc_changes(
        {d.path: d.sha256 for d in old_docs}, {d.path: d.sha256 for d in new_docs}
    )
    count = tiktoken_counter(cfg.chunker.tokenizer)
    old_chunks = {c.text for d in old_docs for c in chunk_document(d, cfg.chunker, count)}
    new_chunks = [c.text for d in new_docs for c in chunk_document(d, cfg.chunker, count)]
    for name in ("added", "modified", "deleted"):
        paths = getattr(changes, name)
        print(f"{name}: {len(paths)}")
        for p in paths if args.verbose or name != "modified" else []:
            print(f"  {p}")
    print(f"unchanged: {len(changes.unchanged)}")
    reused = sum(t in old_chunks for t in new_chunks)
    print(
        f"chunks: {len(old_chunks)} -> {len(new_chunks)}; unchanged text {reused} "
        f"({reused / len(new_chunks):.0%}), new text {len(new_chunks) - reused}, "
        f"gone {len(old_chunks - set(new_chunks))}"
    )


def cmd_build(args: argparse.Namespace) -> None:
    cfg = CorpusConfig.from_yaml(args.config)
    report = cfg.build(args.out, args.ref)
    print(report.model_dump_json(indent=1))


def cmd_compare(args: argparse.Namespace) -> None:
    result = compare_builds(args.a, args.b)
    print(result.model_dump_json(indent=1))
    print("equivalent" if result.equivalent else "DIFFERENT")
    if not result.equivalent:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m research_agent.ingest")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="clone or update a git corpus")
    p.add_argument("config")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("inspect", help="document and chunk statistics, no embedding")
    p.add_argument("config")
    p.add_argument("--ref")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("diff", help="document and chunk changes between two refs")
    p.add_argument("config")
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("build", help="build or incrementally update an index directory")
    p.add_argument("config")
    p.add_argument("--ref")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("compare", help="check two index directories are equivalent")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
