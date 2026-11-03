"""Component configuration (``SHOW CONFIG``) for the agent (stage 10, D37).

``SHOW CONFIG`` needs the CONFIG privilege, which also allows ``SET CONFIG``; the read-only
account must not have it. The reader therefore uses its own account (``agent_cfg``, CONFIG
only, no table privileges) and only ever sends one fixed, parameterised statement.
"""

from __future__ import annotations

import re

from research_agent.tidb.db import Database, QueryResult

COMPONENTS = ("tidb", "tikv", "pd")
_NAME = re.compile(r"^[A-Za-z0-9_.\-]*$")


class ConfigReader:
    def __init__(self, db: Database, max_rows: int = 200) -> None:
        self.db = db
        self.max_rows = max_rows

    def show_config(self, component: str, name: str = "") -> QueryResult:
        """Configuration items of ``component`` whose name contains ``name``."""
        if component not in COMPONENTS:
            raise ValueError(f"组件只能是 {', '.join(COMPONENTS)}")
        if not _NAME.match(name):
            raise ValueError("配置项名只能包含字母、数字、点、下划线和连字符")
        return self.db.query(
            "SHOW CONFIG WHERE type = %s AND name LIKE %s",
            [component, f"%{name}%"],
            self.max_rows,
        )
