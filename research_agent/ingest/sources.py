"""Where corpus files come from: a directory on disk, or one commit of a git repository.

Reading from git (instead of a checkout) lets two releases be indexed side by side without
switching branches, and works on a blobless clone: git fetches each file the first time it
is read.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol


class CorpusSource(Protocol):
    def list_files(self) -> list[str]:
        """All file paths (posix, relative to the source root), sorted."""
        ...

    def read_text(self, path: str) -> str:
        """UTF-8 text of one file; ``FileNotFoundError`` if it does not exist."""
        ...

    def describe(self) -> dict[str, str]:
        """What was read, for the build manifest."""
        ...


class DirectorySource:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"corpus directory not found: {self.root}")

    def list_files(self) -> list[str]:
        return sorted(
            p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()
        )

    def read_text(self, path: str) -> str:
        full = (self.root / path).resolve()
        if not full.is_relative_to(self.root):
            raise FileNotFoundError(f"{path} is outside {self.root}")
        return full.read_text(encoding="utf-8")

    def describe(self) -> dict[str, str]:
        return {"kind": "directory", "root": str(self.root)}


class GitSource:
    """The files of one commit (a tag, branch or sha), read with git plumbing commands."""

    def __init__(self, repo: str | Path, ref: str) -> None:
        self.repo = Path(repo)
        self.ref = ref
        self.commit = self._git("rev-parse", "--verify", f"{ref}^{{commit}}").strip()
        self._cache: dict[str, str] = {}

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.repo), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout

    def list_files(self) -> list[str]:
        out = self._git("ls-tree", "-r", "--name-only", "-z", self.commit)
        return sorted(p for p in out.split("\0") if p)

    def read_text(self, path: str) -> str:
        if path not in self._cache:
            try:
                self._cache[path] = self._git("show", f"{self.commit}:{path}")
            except RuntimeError as e:
                raise FileNotFoundError(f"{path} not in {self.ref} ({self.commit[:12]})") from e
        return self._cache[path]

    def describe(self) -> dict[str, str]:
        return {"kind": "git", "ref": self.ref, "commit": self.commit}


def clone_or_fetch(url: str, dest: str | Path, branches: list[str] | None = None) -> None:
    """Blobless clone of ``url`` into ``dest``, or fetch new commits and tags if it exists.

    With ``branches``, only the latest commit of each is fetched (``--depth 1``), as
    ``origin/<branch>``: enough to read the files of those versions, without the history.
    """
    dest = Path(dest)
    exists = (dest / ".git").exists() or (dest / "HEAD").exists()
    if not branches:
        if exists:
            cmd = ["git", "-C", str(dest), "fetch", "--tags", "origin"]
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            cmd = ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(dest)]
        subprocess.run(cmd, check=True)
        return
    if not exists:
        dest.parent.mkdir(parents=True, exist_ok=True)
        clone = ["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1"]
        subprocess.run([*clone, "--branch", branches[0], url, str(dest)], check=True)
    for branch in branches:
        refspec = f"+refs/heads/{branch}:refs/remotes/origin/{branch}"
        fetch = ["git", "-C", str(dest), "fetch", "--depth", "1", "--filter=blob:none"]
        subprocess.run([*fetch, "origin", refspec], check=True)
