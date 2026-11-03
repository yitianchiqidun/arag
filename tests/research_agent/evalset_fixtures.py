"""A made-up two-document corpus and evalset config shared by the evalset tests."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from research_agent.evalset.dataset import Candidate, EvalsetConfig, Evidence
from research_agent.llm.client import MeteredLLMClient

from .fakes import PLUS_PRICING, completion

CHUNKS: list[dict[str, Any]] = [
    {
        "id": "0",
        "doc": "params.md",
        "headings": ["系统变量", "mini_mem_quota"],
        "text": "[params.md] 系统变量 > mini_mem_quota\n\n`mini_mem_quota` 限制单条查询的内存，默认值为 1 GB。",
    },
    {
        "id": "1",
        "doc": "params.md",
        "headings": ["系统变量", "mini_timeout"],
        "text": "[params.md] 系统变量 > mini_timeout\n\n`mini_timeout` 控制语句的执行超时，默认值为 0。",
    },
    {
        "id": "2",
        "doc": "release-2.0.md",
        "headings": ["发布说明"],
        "text": "[release-2.0.md] 发布说明\n\n`mini_timeout` 的默认值从 30 改为 0。",
    },
]

GROUPS = {"direct": "直接问法", "descriptive": "描述问法", "version": "版本变更"}
CRITERIA = "1. 只有一个正确答案。\n2. 答案能从证据中直接得到。"


def candidate(cid: str, group: str, question: str, answer: str, chunk: str = "0") -> Candidate:
    text = next(c["text"] for c in CHUNKS if c["id"] == chunk)
    quote = text.split("\n\n", 1)[1]
    doc = next(c["doc"] for c in CHUNKS if c["id"] == chunk)
    return Candidate(
        candidate_id=cid,
        group=group,
        question=question,
        answer=answer,
        evidence=[Evidence(doc=doc, chunk_id=chunk, quote=quote)],
        origin={"rule": "test"},
    )


CANDIDATES = [
    candidate("d1", "direct", "mini_mem_quota 的默认值是多少？", "1 GB"),
    candidate("d2", "direct", "mini_timeout 的默认值是多少？", "0", chunk="1"),
    candidate("s1", "descriptive", "限制单条查询内存的变量默认值是多少？", "1 GB"),
    candidate("s2", "descriptive", "控制语句执行超时的变量默认值是多少？", "0", chunk="1"),
    candidate("v1", "version", "2.0 中 mini_timeout 的默认值改成了多少？", "0", chunk="2"),
]


def write_evalset(tmp_path: Path, spot_direct: int = 1, spot_descriptive: int = 1) -> Path:
    """Chunks, a run config and an evalset config under ``tmp_path``; returns the config path."""
    (tmp_path / "build").mkdir(exist_ok=True)
    (tmp_path / "build/chunks.json").write_text(json.dumps(CHUNKS, ensure_ascii=False))
    run = {
        "llm": {"model": "agent"},
        "judge": {"model": "judge"},
        "data": {"chunks_file": str(tmp_path / "build/chunks.json"), "index_dir": "i"},
    }
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(run))
    data = {
        "name": "mini",
        "source": "mini_docs",
        "subject": "MiniDB",
        "run_config": str(tmp_path / "run.yaml"),
        "out_dir": str(tmp_path / "set"),
        "groups": GROUPS,
        "review": {
            "criteria": CRITERIA,
            "llm": {"model": "reviewer"},
            "rules": [
                {"groups": ["direct", "version"], "spot_check_n": spot_direct},
                {"groups": ["descriptive"], "prereview": True, "spot_check_n": spot_descriptive},
            ],
        },
    }
    path = tmp_path / "evalset.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True))
    return path


def load(path: Path) -> EvalsetConfig:
    return EvalsetConfig.from_yaml(path)


Brain = Callable[[str, float], str]


def fake_factory(brain: Brain) -> Callable[[Any, Any], MeteredLLMClient]:
    """``client_from_config`` stand-in: the upstream judge compares answers, ``brain`` the rest."""

    def factory(cfg: Any, pricing: Any) -> MeteredLLMClient:
        client = MeteredLLMClient(model="m", pricing=PLUS_PRICING)

        def post(payload: dict[str, Any], attempts: list[int]) -> dict[str, Any]:
            attempts[0] = 1
            prompt = payload["messages"][-1]["content"]
            match = re.search(r"Generated answer: (.*)\nGold answer: (.*)\n", prompt)
            if match:  # the upstream judge
                same = match.group(1).strip().lower() == match.group(2).strip().lower()
                return completion("correct" if same else "incorrect").json()
            return completion(brain(prompt, payload.get("temperature", 0.0))).json()

        client._post_with_retry = post  # type: ignore[method-assign]
        return client

    return factory
