from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from research_agent.assistant import prompts
from research_agent.assistant.graph import Assistant, Services
from research_agent.assistant.retrieval import DocAnswer, DocCitation
from research_agent.tidb.db import QueryResult
from research_agent.tidb.readonly import ReadOnlyTools
from research_agent.tidb.write import WriteExecutor, WriteTools

from .test_tidb_tools import FakeDatabase

Script = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], dict[str, Any]]


class FakeChat:
    def __init__(self, script: Script) -> None:
        self.script = script
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages, tools=None, temperature=None, max_tokens=None):  # type: ignore[no-untyped-def]
        self.calls.append([dict(m) for m in messages])
        return {"message": self.script(messages, tools), "cost": 0.001}


class FakeDocs:
    def __init__(self, cited: bool = True) -> None:
        self.cited = cited
        self.questions: list[tuple[str, str]] = []

    def answer(self, question: str, channel: str = "arag") -> DocAnswer:
        self.questions.append((question, channel))
        citations = (
            [
                DocCitation(
                    chunk_id="408",
                    label="system-variables.md › tidb_mem_quota_query",
                    status="read",
                )
            ]
            if self.cited
            else []
        )
        return DocAnswer(
            channel="arag",
            question=question,
            answer="默认值是 1073741824（Chunk 408）",
            citations=citations,
            cost=0.01,
        )


def router(intent: str) -> FakeChat:
    def script(messages: list[dict[str, Any]], tools: Any) -> dict[str, Any]:
        text = messages[-1]["content"]
        if text.startswith("下面是用户和 TiDB 运维助手的对话"):
            return {"content": "tidb_mem_quota_query 在 v8.1 中的默认值是多少？"}
        return {"content": json.dumps({"intent": intent, "reason": "test"})}

    return FakeChat(script)


def tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def assistant(
    intent: str,
    agent: FakeChat | None = None,
    docs: FakeDocs | None = None,
    tmp: Path | None = None,
) -> tuple[Assistant, dict[str, Any]]:
    route = router(intent)
    variables = QueryResult(
        columns=["Variable_name", "Value"], rows=[["tidb_mem_quota_query", "1073741824"]]
    )
    schema = QueryResult(
        columns=["Table", "Create Table"], rows=[["orders", "CREATE TABLE orders (...)"]]
    )
    slow = QueryResult(
        columns=["Query_time", "Query"], rows=[[3.2, "SELECT * FROM orders WHERE status = 'paid'"]]
    )
    reader = ReadOnlyTools(
        FakeDatabase(
            {"VARIABLES": variables, "SHOW CREATE TABLE": schema, "cluster_slow_query": slow}
        )
    )
    admin = FakeDatabase()
    executor = WriteExecutor(admin, reader, (tmp or Path("/tmp")) / "audit.jsonl")
    services = Services(
        llm=lambda role: route if role == "router" else (agent or route),
        docs=docs or FakeDocs(),
        reader=reader,
        writes=WriteTools(reader),
        executor=executor,
    )
    return Assistant(services, InMemorySaver()), {"admin": admin, "router": route}


def events(stream: Any) -> list[dict[str, Any]]:
    return list(stream)


def test_param_lookup_answers_with_citations() -> None:
    bot, _ = assistant("param_lookup")
    out = events(bot.ask("s1", "tidb_mem_quota_query 的默认值是多少？"))
    assert [e["type"] for e in out] == ["route", "retrieval_start", "tool_result", "answer", "done"]
    answer = out[3]
    assert answer["text"].startswith("默认值是 1073741824")
    assert answer["citations"] == ["system-variables.md › tidb_mem_quota_query"]
    assert [m["role"] for m in bot.history("s1")] == ["user", "assistant"]


def test_follow_up_is_condensed() -> None:
    docs = FakeDocs()
    bot, parts = assistant("param_lookup", docs=docs)
    events(bot.ask("s1", "tidb_mem_quota_query 的默认值是多少？"))
    events(bot.ask("s1", "那这个参数在 8.1 里呢？"))
    assert docs.questions[-1][0] == "tidb_mem_quota_query 在 v8.1 中的默认值是多少？"
    condense_prompt = parts["router"].calls[-2][0]["content"]
    assert "user: tidb_mem_quota_query 的默认值是多少？" in condense_prompt
    assert len(bot.history("s1")) == 4


def test_answer_without_evidence_is_refused() -> None:
    bot, _ = assistant("research", docs=FakeDocs(cited=False))
    out = events(bot.ask("s1", "某个不存在的功能怎么用？"))
    answer = next(e for e in out if e["type"] == "answer")
    assert answer["text"].startswith(prompts.ANSWER_PREFIX_NO_EVIDENCE)


def test_out_of_scope() -> None:
    bot, _ = assistant("out_of_scope")
    out = events(bot.ask("s1", "今天天气怎么样？"))
    assert [e["type"] for e in out] == ["route", "answer", "done"]
    assert out[1]["text"] == prompts.OUT_OF_SCOPE


def diagnosis_agent() -> FakeChat:
    def script(messages: list[dict[str, Any]], tools: Any) -> dict[str, Any]:
        tool_results = [m for m in messages if m["role"] == "tool"]
        if not tool_results:
            return {"content": "", "tool_calls": [tool_call("slow_queries", {"limit": 5}, "c1")]}
        if len(tool_results) == 1:
            args = {"table": "orders", "columns": ["status"], "reason": "全表扫描"}
            return {"content": "", "tool_calls": [tool_call("propose_create_index", args, "c2")]}
        return {"content": f"根因：orders.status 没有索引。确认结果：{tool_results[-1]['content']}"}

    return FakeChat(script)


def test_diagnosis_write_runs_only_after_confirmation(tmp_path: Path) -> None:
    bot, parts = assistant("diagnosis", agent=diagnosis_agent(), tmp=tmp_path)
    out = events(bot.ask("s1", "订单查询突然变慢了"))
    types = [e["type"] for e in out]
    assert types == ["route", "tool_call", "tool_result", "tool_call", "confirm_required", "done"]
    write = out[4]["write"]
    assert write["statement"] == "CREATE INDEX idx_status ON orders (status)"
    assert "全表扫描" in write["impact"]
    assert parts["admin"].executed == []  # nothing ran yet
    assert bot.pending("s1") is not None
    blocked = events(bot.ask("s1", "另一个问题"))
    assert blocked[0]["type"] == "error"

    done = events(bot.confirm("s1", approved=True))
    assert parts["admin"].executed == [("CREATE INDEX idx_status ON orders (status)", None)]
    answer = next(e for e in done if e["type"] == "answer")
    assert answer["text"].startswith("根因：orders.status 没有索引。确认结果：已执行")
    assert bot.pending("s1") is None


def test_rejected_write_is_not_executed(tmp_path: Path) -> None:
    bot, parts = assistant("diagnosis", agent=diagnosis_agent(), tmp=tmp_path)
    events(bot.ask("s1", "订单查询突然变慢了"))
    done = events(bot.confirm("s1", approved=False))
    assert parts["admin"].executed == []
    answer = next(e for e in done if e["type"] == "answer")
    assert "用户拒绝了这项修改" in answer["text"]
    audit = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert [(a["confirmed"], a["executed"]) for a in audit] == [(False, False)]
    assert events(bot.confirm("s1", approved=True))[0]["type"] == "error"  # nothing pending


def test_app_persists_sessions_and_traces(tmp_path: Path) -> None:
    from research_agent.assistant.app import AssistantApp, AssistantConfig, sqlite_checkpointer
    from research_agent.config import LLMConfig

    cfg = AssistantConfig(
        router=LLMConfig(model="r"),
        agent=LLMConfig(model="a"),
        checkpoint_db=str(tmp_path / "s.sqlite"),
        trace_dir=str(tmp_path / "traces"),
    )
    first, _ = assistant("diagnosis", agent=diagnosis_agent(), tmp=tmp_path)
    app = AssistantApp(
        cfg, services=_services(first), checkpointer=sqlite_checkpointer(cfg.checkpoint_db)
    )
    list(app.ask("s9", "订单查询突然变慢了"))
    assert app.pending("s9") is not None
    # A new app on the same database finds the session and its pending write.
    again = AssistantApp(
        cfg, services=_services(first), checkpointer=sqlite_checkpointer(cfg.checkpoint_db)
    )
    assert again.pending("s9") is not None
    list(again.confirm("s9", approved=False))
    assert [m["role"] for m in again.history("s9")] == ["user", "assistant"]
    rows = [
        json.loads(line) for line in (tmp_path / "traces" / "s9.jsonl").read_text().splitlines()
    ]
    assert {r["kind"] for r in rows} == {"ask", "confirm"}
    assert rows[0]["type"] == "route" and rows[-1]["type"] == "done"


def _services(bot: Assistant) -> Services:
    return bot.services


def test_diagnosis_without_cluster_says_so() -> None:
    def agent_script(messages: list[dict[str, Any]], tools: Any) -> dict[str, Any]:
        if not [m for m in messages if m["role"] == "tool"]:
            return {"content": "", "tool_calls": [tool_call("slow_queries", {}, "c1")]}
        return {"content": "可能是缺少索引。"}

    bot, _ = assistant("diagnosis", agent=FakeChat(agent_script))
    bot.services.reader = None
    out = events(bot.ask("s1", "查询变慢了"))
    answer = next(e for e in out if e["type"] == "answer")
    assert answer["text"] == prompts.NO_CLUSTER


def test_diagnosis_stops_at_the_turn_deadline() -> None:
    bot, _ = assistant("diagnosis", agent=diagnosis_agent())
    bot.services.turn_timeout_s = 1e-9  # already over when the loop starts
    out = events(bot.ask("s1", "查询变慢了"))
    answer = next(e for e in out if e["type"] == "answer")
    assert answer["text"] == prompts.DIAG_TIMEOUT
    assert not [e for e in out if e["type"] == "tool_call"]
    assert bot.pending("s1") is None


def test_step_limit_asks_for_a_conclusion_without_tools() -> None:
    def agent_script(messages: list[dict[str, Any]], tools: Any) -> dict[str, Any]:
        if tools is None:  # the final call after the step limit
            return {"content": "根因：status 没有索引（依据：执行计划为 TableFullScan）。"}
        n = len([m for m in messages if m["role"] == "tool"])
        return {"content": "", "tool_calls": [tool_call("slow_queries", {}, f"c{n}")]}

    agent = FakeChat(agent_script)
    bot, _ = assistant("diagnosis", agent=agent)
    bot.services.max_steps = 3
    out = events(bot.ask("s1", "查询变慢了"))
    answer = next(e for e in out if e["type"] == "answer")
    assert answer["text"] == "根因：status 没有索引（依据：执行计划为 TableFullScan）。"
    assert len([e for e in out if e["type"] == "tool_call"]) == 3
    assert agent.calls[-1][-1] == {"role": "user", "content": prompts.DIAG_STEP_LIMIT}
