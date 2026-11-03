"""CLI: ``python -m research_agent.assistant [--session ID] [--approve | --reject] QUESTION...``.

Asks the questions in order in one session and prints every event; a write proposal is
approved or rejected as told (otherwise left pending).
"""

from __future__ import annotations

import argparse
import json
import uuid

from research_agent.assistant.app import AssistantApp, AssistantConfig
from research_agent.upstream import load_env


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m research_agent.assistant")
    parser.add_argument("questions", nargs="+")
    parser.add_argument("--config", default="configs/assistant.yaml")
    parser.add_argument("--session", default=None)
    decision = parser.add_mutually_exclusive_group()
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")
    args = parser.parse_args()
    load_env()
    app = AssistantApp(AssistantConfig.from_yaml(args.config))
    session = args.session or uuid.uuid4().hex[:12]
    print(f"session {session}")
    for question in args.questions:
        print(f"\n>>> {question}")
        for event in app.ask(session, question):
            print(json.dumps(event, ensure_ascii=False, default=str)[:1500])
        if app.pending(session) and (args.approve or args.reject):
            print(f">>> {'approve' if args.approve else 'reject'}")
            for event in app.confirm(session, approved=args.approve):
                print(json.dumps(event, ensure_ascii=False, default=str)[:1500])


if __name__ == "__main__":
    main()
