"""Run the fault scenarios end to end through the assistant and score them (stage 8, D34).

Per scenario: inject the fault, run its workload, ask the assistant the user's symptom in a new
session, approve a proposed write only if it is the reference fix (a simulated user who knows
the answer; anything else is rejected), then clean up, also after an error. Scored per run:

- diagnosed: every keyword group of the scenario appears in the final answer;
- fix: the reference write was proposed (or, when no write fixes it, none was proposed);
- related docs: the answer cites a document of the scenario's list;
- tool calls, writes executed without approval (from the audit log; must be 0), latency, cost.

    python -m research_agent.diagnosis --out results/stage8/scenarios [--only lock_wait,...]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from research_agent.diagnosis.scenarios import SCENARIOS, SUPPLEMENTARY, Scenario

MAX_CONFIRMATIONS = 3


class Assistant(Protocol):
    """The AssistantApp surface the runner uses."""

    def ask(self, session_id: str, question: str) -> Iterable[dict[str, Any]]: ...

    def confirm(self, session_id: str, approved: bool) -> Iterable[dict[str, Any]]: ...

    def pending(self, session_id: str) -> dict[str, Any] | None: ...


class Proposal(BaseModel):
    kind: str
    statement: str
    approved: bool


class ScenarioResult(BaseModel):
    name: str
    title: str
    intent: str | None
    diagnosed: bool
    missing_keywords: list[list[str]]
    expected_write: str | None
    proposals: list[Proposal]
    fix_proposed: bool
    cited_docs: list[str]
    related_doc_cited: bool
    tool_calls: int
    tools: dict[str, int]
    unapproved_writes: int  # writes executed without the user's approval; must be 0
    latency_s: float
    cost: float
    answer: str
    error: str | None = None
    cleanup_errors: list[str] = []


def missing_keywords(answer: str, groups: list[tuple[str, ...]]) -> list[list[str]]:
    text = answer.lower()
    return [list(g) for g in groups if not any(k.lower() in text for k in g)]


def unapproved_writes(audit_lines: list[str]) -> int:
    rows = [json.loads(line) for line in audit_lines if line.strip()]
    return sum(1 for r in rows if r.get("executed") and not r.get("confirmed"))


def _audit_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def run_scenario(
    app: Assistant, scenario: Scenario, session: str, audit_file: Path, events_file: Path
) -> ScenarioResult:
    audit_before = len(_audit_lines(audit_file))
    events: list[dict[str, Any]] = []
    proposals: list[Proposal] = []
    error: str | None = None
    try:
        scenario.inject()
        scenario.workload()
        events += list(app.ask(session, scenario.symptom))
        for _ in range(MAX_CONFIRMATIONS):
            pending = app.pending(session)
            if not pending:
                break
            approved = scenario.accepts(pending)
            proposals.append(
                Proposal(
                    kind=str(pending.get("kind")),
                    statement=str(pending.get("statement")),
                    approved=approved,
                )
            )
            events += list(app.confirm(session, approved))
    except Exception as e:  # noqa: BLE001 - recorded as the scenario's error; cleanup still runs
        error = f"{type(e).__name__}: {e}"
    finally:
        scenario.cleanup()
    events_file.parent.mkdir(parents=True, exist_ok=True)
    events_file.write_text(
        "".join(json.dumps(e, ensure_ascii=False, default=str) + "\n" for e in events), "utf-8"
    )
    answers = [e for e in events if e.get("type") == "answer"]
    final = str(answers[-1]["text"]) if answers else ""
    errors = [str(e.get("message")) for e in events if e.get("type") == "error"]
    if errors and error is None:
        error = "; ".join(errors)
    tools = Counter(str(e["tool"]) for e in events if e.get("type") == "tool_call")
    cited = list(
        dict.fromkeys(str(c).split(" › ")[0] for a in answers for c in a.get("citations") or [])
    )
    missing = missing_keywords(final, scenario.keywords())
    accepted = [p for p in proposals if p.approved]
    return ScenarioResult(
        name=scenario.name,
        title=scenario.title,
        intent=next((e.get("intent") for e in events if e.get("type") == "route"), None),
        diagnosed=bool(final) and not missing,
        missing_keywords=missing,
        expected_write=scenario.expected_write,
        proposals=proposals,
        fix_proposed=bool(accepted) if scenario.expected_write else not proposals,
        cited_docs=cited,
        related_doc_cited=any(d in doc for doc in cited for d in scenario.related_docs),
        tool_calls=sum(tools.values()),
        tools=dict(tools),
        unapproved_writes=unapproved_writes(_audit_lines(audit_file)[audit_before:]),
        latency_s=round(
            sum(float(e.get("latency_s") or 0) for e in events if e.get("type") == "done"), 1
        ),
        cost=round(sum(float(a.get("cost") or 0) for a in answers), 4),
        answer=final,
        error=error,
        cleanup_errors=scenario.cleanup_errors,
    )


def _mark(value: bool) -> str:
    return "✓" if value else "✗"


def summary_markdown(results: list[ScenarioResult]) -> str:
    lines = [
        "| 场景 | 路由 | 诊断正确 | 处理建议 | 引用相关文档 | 工具调用 | 未经确认的写操作 | 耗时 (s) | 花费（元） |",
        "| --- | --- | :---: | --- | :---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        if r.expected_write:
            fix = f"{_mark(r.fix_proposed)} " + (
                "；".join(
                    f"{p.statement}（{'批准' if p.approved else '拒绝'}）" for p in r.proposals
                )
                or "未提出"
            )
        else:
            fix = f"{_mark(r.fix_proposed)} " + (
                "未提出修改（正确）"
                if not r.proposals
                else "；".join(f"{p.statement}（拒绝）" for p in r.proposals)
            )
        lines.append(
            f"| {r.title} | {r.intent} | {_mark(r.diagnosed)} | {fix} | {_mark(r.related_doc_cited)} | "
            f"{r.tool_calls} | {r.unapproved_writes} | {r.latency_s} | {r.cost:.4f} |"
        )
    n = len(results)
    lines.append(
        f"\n诊断正确 {sum(r.diagnosed for r in results)} / {n}，处理建议正确 "
        f"{sum(r.fix_proposed for r in results)} / {n}，未经确认的写操作 "
        f"{sum(r.unapproved_writes for r in results)}，花费 {sum(r.cost for r in results):.4f} 元。"
    )
    return "\n".join(lines)


def run_all(
    app: Assistant,
    scenarios: list[Scenario],
    out: Path,
    audit_file: Path,
    session_prefix: str = "stage8",
) -> list[ScenarioResult]:
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for scenario in scenarios:
        print(f"[{scenario.name}] {scenario.title}", flush=True)
        result = run_scenario(
            app,
            scenario,
            f"{session_prefix}-{scenario.name}",
            audit_file,
            out / f"{scenario.name}.events.jsonl",
        )
        print(
            f"  diagnosed={result.diagnosed} fix={result.fix_proposed} error={result.error}",
            flush=True,
        )
        results.append(result)
    (out / "results.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8"
    )
    (out / "summary.md").write_text(summary_markdown(results) + "\n", encoding="utf-8")
    return results


def main(
    argv: list[str] | None = None, make_app: Callable[[Path], Assistant] | None = None
) -> None:
    parser = argparse.ArgumentParser(prog="python -m research_agent.diagnosis")
    parser.add_argument("--out", default="results/stage8/scenarios")
    parser.add_argument("--config", default="configs/assistant.yaml")
    parser.add_argument(
        "--only", default=None, help="comma-separated scenario names (also the supplementary ones)"
    )
    args = parser.parse_args(argv)
    from research_agent.upstream import load_env

    load_env()
    from research_agent.assistant.app import AssistantConfig
    from research_agent.tidb.db import admin_database

    cfg = AssistantConfig.from_yaml(args.config)
    out = Path(args.out)
    audit_file = out / "writes.jsonl"
    if make_app is None:
        from langgraph.checkpoint.memory import InMemorySaver

        from research_agent.assistant.app import AssistantApp

        app_cfg = cfg.model_copy(
            update={"trace_dir": str(out / "traces"), "audit_file": str(audit_file)}
        )
        app: Assistant = AssistantApp(app_cfg, checkpointer=InMemorySaver())
    else:
        app = make_app(audit_file)
    names = set(args.only.split(",")) if args.only else None
    pool = SCENARIOS if names is None else SCENARIOS + SUPPLEMENTARY
    scenarios = [
        cls(lambda: admin_database(cfg.tidb)) for cls in pool if names is None or cls.name in names
    ]
    results = run_all(app, scenarios, out, audit_file)
    print(summary_markdown(results))
