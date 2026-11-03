"""Run the fault scenarios end to end through the assistant and score them (stage 8, D34).

Per scenario: inject the fault, run its workload, ask the assistant the user's symptom in a new
session, approve a proposed write only if it is the reference fix (a simulated user who knows
the answer; anything else is rejected), then clean up, also after an error. Scored per run:

- diagnosed: every keyword group of the scenario appears in the final answer;
- fix: the reference write was proposed (or, when no write fixes it, none was proposed);
- documents: the final answer cites a document (a chunk id it names, or a document file name)
  and whether one of them is in the scenario's list of related documents;
- tool calls, writes executed without approval (from the audit log; must be 0), latency, cost.

    python -m research_agent.diagnosis --out results/stage8/scenarios [--only lock_wait,...]
        [--before <earlier run dirs, comma-separated>] [--max-cost 1.5]
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
from research_agent.eval.citations import ChunkRef, cited_chunk_ids

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
    cited_docs: list[str]  # documents of the search_docs results attached to the answers
    related_doc_cited: bool  # one of answer_docs is a related document of the scenario
    tool_calls: int
    tools: dict[str, int]
    unapproved_writes: int  # writes executed without the user's approval; must be 0
    latency_s: float
    cost: float
    answer: str
    error: str | None = None
    cleanup_errors: list[str] = []
    answer_docs: list[str] = []  # documents the final answer itself cites


def answer_docs(answer: str, refs: dict[str, ChunkRef]) -> list[str]:
    """Documents the answer cites: those of the chunk ids it names, then file names it writes."""
    docs = [refs[c].doc for c in cited_chunk_ids(answer) if c in refs and refs[c].doc]
    known = sorted({r.doc for r in refs.values() if r.doc}, key=len, reverse=True)
    docs += [d for d in known if d in answer]
    return [str(d) for d in dict.fromkeys(docs)]


def score_documents(
    result: ScenarioResult, related: tuple[str, ...], refs: dict[str, ChunkRef]
) -> ScenarioResult:
    """Recompute the document columns from the final answer (also for runs made before them)."""
    docs = answer_docs(result.answer, refs)
    return result.model_copy(
        update={
            "answer_docs": docs,
            "related_doc_cited": any(r in d for d in docs for r in related),
        }
    )


def missing_keywords(answer: str, groups: list[tuple[str, ...]]) -> list[list[str]]:
    text = answer.lower()
    return [list(g) for g in groups if not any(k.lower() in text for k in g)]


def unapproved_writes(audit_lines: list[str]) -> int:
    rows = [json.loads(line) for line in audit_lines if line.strip()]
    return sum(1 for r in rows if r.get("executed") and not r.get("confirmed"))


def _audit_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def run_scenario(
    app: Assistant,
    scenario: Scenario,
    session: str,
    audit_file: Path,
    events_file: Path,
    refs: dict[str, ChunkRef] | None = None,
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
    result = ScenarioResult(
        name=scenario.name,
        title=scenario.title,
        intent=next((e.get("intent") for e in events if e.get("type") == "route"), None),
        diagnosed=bool(final) and not missing,
        missing_keywords=missing,
        expected_write=scenario.expected_write,
        proposals=proposals,
        fix_proposed=bool(accepted) if scenario.expected_write else not proposals,
        cited_docs=cited,
        related_doc_cited=False,
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
    return score_documents(result, scenario.related_docs, refs or {})


def _mark(value: bool) -> str:
    return "✓" if value else "✗"


def _fix(r: ScenarioResult) -> str:
    if r.expected_write:
        done = "；".join(
            f"{p.statement}（{'批准' if p.approved else '拒绝'}）" for p in r.proposals
        )
        return f"{_mark(r.fix_proposed)} " + (done or "未提出")
    if not r.proposals:
        return f"{_mark(r.fix_proposed)} 未提出修改（正确）"
    return f"{_mark(r.fix_proposed)} " + "；".join(f"{p.statement}（拒绝）" for p in r.proposals)


HEADER = (
    "| 路由 | 诊断正确 | 处理建议 | 回答引用文档 | 引用相关文档 | 工具调用 | 未经确认的写操作 "
    "| 耗时 (s) | 花费（元） |"
)


def _cells(r: ScenarioResult) -> str:
    return (
        f"| {r.intent} | {_mark(r.diagnosed)} | {_fix(r)} | {_mark(bool(r.answer_docs))} | "
        f"{_mark(r.related_doc_cited)} | {r.tool_calls} | {r.unapproved_writes} | {r.latency_s} | "
        f"{r.cost:.4f} |"
    )


def _totals(results: list[ScenarioResult]) -> str:
    n = len(results)
    return (
        f"诊断正确 {sum(r.diagnosed for r in results)} / {n}，处理建议正确 "
        f"{sum(r.fix_proposed for r in results)} / {n}，回答引用文档 "
        f"{sum(bool(r.answer_docs) for r in results)} / {n}，引用相关文档 "
        f"{sum(r.related_doc_cited for r in results)} / {n}，未经确认的写操作 "
        f"{sum(r.unapproved_writes for r in results)}，花费 {sum(r.cost for r in results):.4f} 元"
    )


def summary_markdown(results: list[ScenarioResult]) -> str:
    lines = [
        "| 场景 " + HEADER,
        "| --- | --- | :---: | --- | :---: | :---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [f"| {r.title} " + _cells(r) for r in results]
    lines.append(f"\n{_totals(results)}。")
    return "\n".join(lines)


def comparison_markdown(before: list[ScenarioResult], after: list[ScenarioResult]) -> str:
    """Two rows per scenario (before, after) and the totals of both runs."""
    earlier = {r.name: r for r in before}
    lines = [
        "| 场景 | 运行 " + HEADER,
        "| --- | --- | --- | :---: | --- | :---: | :---: | ---: | ---: | ---: | ---: |",
    ]
    for r in after:
        if r.name in earlier:
            lines.append(f"| {r.title} | 修改前 " + _cells(earlier[r.name]))
        lines.append(f"| {r.title} | 修改后 " + _cells(r))
    kept = [earlier[r.name] for r in after if r.name in earlier]
    lines.append(f"\n修改前：{_totals(kept)}。\n\n修改后：{_totals(after)}。")
    return "\n".join(lines)


def load_results(dirs: list[Path], refs: dict[str, ChunkRef]) -> list[ScenarioResult]:
    """Results of earlier runs, the later directory winning per scenario, rescored for the
    document columns."""
    related = {cls.name: cls.related_docs for cls in SCENARIOS + SUPPLEMENTARY}
    by_name: dict[str, ScenarioResult] = {}
    for d in dirs:
        for line in (d / "results.jsonl").read_text(encoding="utf-8").splitlines():
            r = ScenarioResult.model_validate_json(line)
            by_name[r.name] = score_documents(r, related.get(r.name, ()), refs)
    return list(by_name.values())


def run_all(
    app: Assistant,
    scenarios: list[Scenario],
    out: Path,
    audit_file: Path,
    session_prefix: str = "stage8",
    refs: dict[str, ChunkRef] | None = None,
    max_cost: float | None = None,
) -> list[ScenarioResult]:
    out.mkdir(parents=True, exist_ok=True)
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        spent = sum(r.cost for r in results)
        if max_cost is not None and spent >= max_cost:
            print(
                f"cost limit {max_cost} reached ({spent:.4f}); {scenario.name} not run", flush=True
            )
            break
        print(f"[{scenario.name}] {scenario.title}", flush=True)
        result = run_scenario(
            app,
            scenario,
            f"{session_prefix}-{scenario.name}",
            audit_file,
            out / f"{scenario.name}.events.jsonl",
            refs,
        )
        print(
            f"  diagnosed={result.diagnosed} fix={result.fix_proposed} cost={result.cost} "
            f"error={result.error}",
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
    parser.add_argument("--before", default=None, help="earlier run dirs to compare with")
    parser.add_argument("--max-cost", type=float, default=None, help="CNY; checked per scenario")
    args = parser.parse_args(argv)
    from research_agent.upstream import load_env

    load_env()
    from research_agent.assistant.app import AssistantConfig
    from research_agent.config import RunConfig
    from research_agent.eval.citations import load_chunk_refs
    from research_agent.tidb.db import admin_database

    cfg = AssistantConfig.from_yaml(args.config)
    refs = load_chunk_refs(RunConfig.from_yaml(cfg.run_config).data.chunks_file)
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
    results = run_all(app, scenarios, out, audit_file, refs=refs, max_cost=args.max_cost)
    print(summary_markdown(results))
    if args.before:
        before = load_results([Path(d) for d in args.before.split(",")], refs)
        table = comparison_markdown(before, results)
        (out / "comparison.md").write_text(table + "\n", encoding="utf-8")
        print(table)
