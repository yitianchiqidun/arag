"""Diagnosis reports (stage 10): one Markdown file per call, saved locally.

The model fills the sections (symptom, evidence, root cause, actions, citations); every field
is cut to a fixed length, so a report stays readable whatever the model sends.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

MAX_FIELD = 1500
MAX_ITEMS = 20


def _cut(text: str, limit: int = MAX_FIELD) -> str:
    text = str(text).strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _items(values: list[str] | None) -> list[str]:
    return [_cut(v, 600) for v in (values or [])[:MAX_ITEMS] if str(v).strip()]


class ReportWriter:
    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)

    def render(
        self,
        title: str,
        symptom: str,
        evidence: list[str],
        root_cause: str,
        actions: list[str],
        citations: list[str] | None = None,
    ) -> str:
        def bullets(values: list[str]) -> str:
            return "\n".join(f"- {v}" for v in values) or "- （无）"

        return (
            f"# {_cut(title, 120)}\n\n"
            f"生成时间：{datetime.now().isoformat(timespec='seconds')}\n\n"
            f"## 症状\n\n{_cut(symptom)}\n\n"
            f"## 证据\n\n{bullets(_items(evidence))}\n\n"
            f"## 根因\n\n{_cut(root_cause)}\n\n"
            f"## 处理\n\n{bullets(_items(actions))}\n\n"
            f"## 引用\n\n{bullets(_items(citations))}\n"
        )

    def export(self, **fields: object) -> str:
        """Writes the report; returns its path and size."""
        text = self.render(**fields)  # type: ignore[arg-type]
        slug = re.sub(r"[^0-9A-Za-z一-鿿]+", "-", str(fields.get("title", "report")))
        path = (
            self.out_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{slug.strip('-')[:40] or 'report'}.md"
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return f"已保存诊断报告：{path}（{len(text)} 字）"
