"""Searching the component logs of the local tiup playground (stage 10): read-only file access.

The playground keeps each component's log under ``~/.tiup/data/<tag>/<component>-<n>/``
(``tidb.log``, ``tikv.log``, ``pd.log``). Only the end of each file is read, and the newest
matching lines are returned, each cut to a fixed length.
"""

from __future__ import annotations

from pathlib import Path

LOG_FILES = {"tidb": "tidb.log", "tikv": "tikv.log", "pd": "pd.log"}
MAX_LINES = 200


class LogSearch:
    def __init__(
        self, data_dir: str | Path, tail_bytes: int = 20 << 20, max_line_chars: int = 400
    ) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.tail_bytes = tail_bytes
        self.max_line_chars = max_line_chars

    def files(self, component: str) -> list[Path]:
        if component not in LOG_FILES:
            raise ValueError(f"组件只能是 {', '.join(LOG_FILES)}")
        return sorted(self.data_dir.glob(f"{component}-*/{LOG_FILES[component]}"))

    def _tail(self, path: Path) -> list[str]:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - self.tail_bytes))
            data = f.read()
        lines = data.decode("utf-8", errors="replace").splitlines()
        return lines[1:] if size > self.tail_bytes else lines  # the first line may be cut

    def search(self, component: str, keyword: str = "", max_lines: int = 50) -> str:
        """The newest ``max_lines`` lines containing ``keyword`` (case-insensitive; empty: any)."""
        files = self.files(component)
        if not files:
            return f"没有找到 {component} 的日志文件（目录 {self.data_dir}）。"
        limit = max(1, min(int(max_lines), MAX_LINES))
        needle = keyword.strip().lower()
        out = []
        for path in files:
            lines = self._tail(path)
            hits = [line for line in lines if needle in line.lower()] if needle else lines
            shown = hits[-limit:]
            out.append(
                f"{path.parent.name}/{path.name}：匹配 {len(hits)} 行"
                + (f"，显示最新的 {len(shown)} 行" if len(hits) > len(shown) else "")
            )
            out += [
                line if len(line) <= self.max_line_chars else line[: self.max_line_chars] + "…"
                for line in shown
            ]
        return "\n".join(out)
