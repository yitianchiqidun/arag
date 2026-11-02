"""Bad case labels: a fixed taxonomy (configs/bad_case_tags.yaml) and a per-run label file.

Labels live in ``labels.json`` next to the run's predictions, keyed by question id. The
trajectories themselves are never rewritten, so a labelling mistake cannot damage data.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from research_agent.upstream import REPO_ROOT

DEFAULT_TAXONOMY = REPO_ROOT / "configs/bad_case_tags.yaml"
LABELS_FILE = "labels.json"


class Tag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str
    description: str = ""


class Taxonomy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tags: list[Tag] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Taxonomy:
        ids = [t.id for t in self.tags]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate tag ids: {ids}")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_TAXONOMY) -> Taxonomy:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    @property
    def ids(self) -> list[str]:
        return [t.id for t in self.tags]

    def label(self, tag_id: str) -> str:
        for t in self.tags:
            if t.id == tag_id:
                return t.label
        return tag_id  # an id dropped from the taxonomy still displays


class Label(BaseModel):
    tags: list[str] = Field(default_factory=list)
    note: str = ""
    updated_at: str


class LabelFile(BaseModel):
    schema_version: int = 1
    labels: dict[str, Label] = Field(default_factory=dict)


class LabelStore:
    """Reads and writes ``<run_dir>/labels.json``; every write re-reads the file first."""

    def __init__(self, run_dir: str | Path, taxonomy: Taxonomy) -> None:
        self.path = Path(run_dir) / LABELS_FILE
        self.taxonomy = taxonomy

    def load(self) -> LabelFile:
        if not self.path.exists():
            return LabelFile()
        return LabelFile.model_validate_json(self.path.read_text(encoding="utf-8"))

    def get(self, question_id: str) -> Label | None:
        return self.load().labels.get(question_id)

    def set(self, question_id: str, tags: list[str], note: str = "") -> Label | None:
        """Save the label of one question; no tags and no note removes it."""
        unknown = [t for t in tags if t not in self.taxonomy.ids]
        if unknown:
            raise ValueError(f"unknown tags {unknown}; known: {self.taxonomy.ids}")
        data = self.load()
        label: Label | None = None
        if tags or note.strip():
            ordered = [t for t in self.taxonomy.ids if t in tags]  # taxonomy order, no dups
            label = Label(
                tags=ordered,
                note=note.strip(),
                updated_at=datetime.now().isoformat(timespec="seconds"),
            )
            data.labels[question_id] = label
        else:
            data.labels.pop(question_id, None)
        self._write(data)
        return label

    def _write(self, data: LabelFile) -> None:
        # Write to a temporary file and rename, so a crash never leaves a truncated file.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".labels.", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data.model_dump_json(indent=2) + "\n")
        os.replace(tmp, self.path)
