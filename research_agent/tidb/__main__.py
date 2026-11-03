"""CLI: ``python -m research_agent.tidb {start,stop,status,setup,verify}``.

- ``start`` / ``stop`` / ``status``: the local ``tiup playground`` cluster
- ``setup``: sample database (loaded once) and the read-only account
- ``verify``: every write probe must be refused for the read-only account
"""

from __future__ import annotations

import argparse
import json

from research_agent.tidb import cluster
from research_agent.tidb.db import TiDBConfig


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m research_agent.tidb")
    parser.add_argument("command", choices=["start", "stop", "status", "setup", "verify"])
    args = parser.parse_args()
    ccfg = cluster.ClusterConfig()
    if args.command == "start":
        pid = cluster.start(ccfg)
        print(json.dumps({"pid": pid, **cluster.status(ccfg)}, ensure_ascii=False))
    elif args.command == "stop":
        print("stopped" if cluster.stop(ccfg) else "not running")
    elif args.command == "status":
        print(json.dumps(cluster.status(ccfg), ensure_ascii=False))
    else:
        from research_agent.upstream import load_env

        load_env()
        cfg = TiDBConfig()
        if args.command == "setup":
            from research_agent.tidb.setup import setup

            print(json.dumps(setup(cfg), ensure_ascii=False))
        else:
            from research_agent.tidb.setup import verify_readonly

            result = verify_readonly(cfg)
            for sql, error in result.items():
                print(
                    ("refused  " if error else "ACCEPTED ")
                    + sql
                    + (f"\n         {error}" if error else "")
                )
            if any(e is None for e in result.values()):
                raise SystemExit("the read-only account could write")


if __name__ == "__main__":
    main()
