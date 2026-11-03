from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from streamlit.testing.v1 import AppTest

from research_agent.assistant.graph import Assistant
from research_agent.upstream import REPO_ROOT

from .test_assistant_graph import assistant, diagnosis_agent

APP = str(REPO_ROOT / "apps/chat_demo.py")
_STATE: dict[str, Any] = {}


class FakeApp:
    """The AssistantApp surface the page uses, over the fake graph services."""

    def __init__(self, bot: Assistant) -> None:
        self.bot = bot

    def ask(self, session: str, question: str) -> Any:
        return self.bot.ask(session, question)

    def confirm(self, session: str, approved: bool) -> Any:
        return self.bot.confirm(session, approved)

    def pending(self, session: str) -> Any:
        return self.bot.pending(session)


def fake_app() -> FakeApp:
    return _STATE["app"]


@pytest.fixture
def page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    bot, parts = assistant("diagnosis", agent=diagnosis_agent(), tmp=tmp_path)
    bot = Assistant(bot.services, InMemorySaver())
    _STATE["app"] = FakeApp(bot)
    monkeypatch.setenv("CHAT_DEMO_FACTORY", "tests.research_agent.test_chat_demo:fake_app")
    return parts


def test_page_confirms_a_write(page: dict[str, Any]) -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.chat_input[0].set_value("订单查询突然变慢了").run()
    assert any("CREATE INDEX idx_status ON orders (status)" in c.value for c in at.code)
    assert page["admin"].executed == []
    at.button(key="approve").click().run()
    assert not at.exception
    assert page["admin"].executed == [("CREATE INDEX idx_status ON orders (status)", None)]
    assert any("根因：orders.status 没有索引" in m.value for m in at.markdown)
    assert not any(b.key == "approve" for b in at.button)  # nothing pending any more
