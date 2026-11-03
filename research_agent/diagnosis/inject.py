"""Inject or clean up one fault scenario by hand, e.g. for a demo in the chat page:

    python -m research_agent.diagnosis.inject memory_quota
    python -m research_agent.diagnosis.inject memory_quota --cleanup

Prints the symptom to ask the assistant. ``lock_wait`` needs its process alive (a held lock
and a retrying client), so it is only available through the scenario runner.
"""

from __future__ import annotations

import argparse

from research_agent.diagnosis.scenarios import SCENARIOS, LockWait


def main(argv: list[str] | None = None) -> None:
    names = [c.name for c in SCENARIOS if c is not LockWait]
    parser = argparse.ArgumentParser(prog="python -m research_agent.diagnosis.inject")
    parser.add_argument("scenario", choices=names)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args(argv)
    from research_agent.tidb.db import TiDBConfig, admin_database
    from research_agent.upstream import load_env

    load_env()
    cfg = TiDBConfig()
    scenario = next(c for c in SCENARIOS if c.name == args.scenario)(lambda: admin_database(cfg))
    if args.cleanup:
        scenario.cleanup()
        print(
            f"cleaned up {scenario.name}"
            + (f": {scenario.cleanup_errors}" if scenario.cleanup_errors else "")
        )
        return
    scenario.inject()
    scenario.workload()
    print(f"injected {scenario.name}（{scenario.title}）。可以这样问助手：\n{scenario.symptom}")


if __name__ == "__main__":
    main()
