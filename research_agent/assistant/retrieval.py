"""Document retrieval for the assistant: A-RAG as one tool, and Naive RAG as the fast channel.

A-RAG runs unchanged (the same BaseAgent, tools and system prompt as in the evaluations); the
assistant only sees its final answer, the citations resolved to documents and headings, and
the recorded trajectory. Naive RAG is the stage 1 baseline with the citation instruction on
(D29), used by the fast channel in stage 7.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from research_agent.agent.prompt import build_system_prompt, load_upstream_prompt
from research_agent.agent.tools import build_tool_registry, plain_text_tokens, shared_semantic_tool
from research_agent.baselines.naive_rag import ChunkIndex, NaiveRAG
from research_agent.config import RunConfig
from research_agent.eval.citations import load_chunk_refs, record_citations
from research_agent.eval.runner import RecordingAgent
from research_agent.llm.client import client_from_config
from research_agent.llm.pricing import PricingTable
from research_agent.tracing.recorder import TimedToolRegistry, TraceRecorder

Channel = Literal["arag", "naive_rag"]


class DocCitation(BaseModel):
    chunk_id: str
    label: str  # "doc › heading › …"
    status: str  # read / snippet / unseen / unknown


class DocAnswer(BaseModel):
    channel: Channel
    question: str
    answer: str
    citations: list[DocCitation] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)  # tool calls, for the trace
    cost: float = 0.0
    latency_s: float = 0.0
    termination: str = "answered"


class DocRetriever:
    """Both channels over one index; safe to share across sessions (one call at a time)."""

    def __init__(self, run_config: RunConfig) -> None:
        self.cfg = run_config
        self.pricing = PricingTable.from_yaml(run_config.pricing_file)
        self.refs = load_chunk_refs(run_config.data.chunks_file)
        self._tools = build_tool_registry(run_config)
        self._prompt = build_system_prompt(load_upstream_prompt(), run_config.agent)
        self._naive: tuple[ChunkIndex, Any] | None = None
        self._lock = threading.Lock()

    def _citations(self, record: dict[str, Any]) -> list[DocCitation]:
        return [
            DocCitation(chunk_id=c.chunk_id, label=c.label(), status=c.status)
            for c in record_citations(record, self.refs)
        ]

    def arag(self, question: str) -> DocAnswer:
        recorder = TraceRecorder(agent_has_tools=True)
        llm = client_from_config(self.cfg.llm, self.pricing)
        llm.observer = recorder.on_llm_call
        agent = RecordingAgent(
            llm_client=llm,
            tools=TimedToolRegistry(self._tools, recorder),
            system_prompt=self._prompt,
            max_loops=self.cfg.agent.max_loops,
            max_token_budget=self.cfg.agent.max_token_budget,
            verbose=False,
        )
        start = time.perf_counter()
        with self._lock:
            result = agent.run(question)
        trajectory = result.get("trajectory") or []
        record = {"pred_answer": result.get("answer", ""), "trajectory": trajectory}
        termination = "answered"
        if result.get("max_loops_exceeded"):
            termination = "max_loops"
        elif result.get("token_budget_exceeded"):
            termination = "token_budget"
        return DocAnswer(
            channel="arag",
            question=question,
            answer=str(result.get("answer") or ""),
            citations=self._citations(record),
            steps=[{"tool": s.get("tool_name"), "args": s.get("arguments")} for s in trajectory],
            cost=llm.usage_summary().cost,
            latency_s=round(time.perf_counter() - start, 2),
            termination=termination,
        )

    def _naive_resources(self) -> tuple[ChunkIndex, Any]:
        if self._naive is None:
            from arag.tools.read_chunk import ReadChunkTool

            semantic = shared_semantic_tool(self.cfg.data, self.cfg.embedding)
            reader = ReadChunkTool(chunks_file=self.cfg.data.chunks_file)
            plain_text_tokens(reader)
            index = ChunkIndex.load_or_build(
                reader,
                self.cfg.data.index_dir,
                semantic.embedding_model,
                self.cfg.embedding.model,
                self.cfg.embedding.batch_size,
                semantic._embedding_lock,
            )
            self._naive = (index, reader)
        return self._naive

    def naive(self, question: str) -> DocAnswer:
        index, reader = self._naive_resources()
        semantic = shared_semantic_tool(self.cfg.data, self.cfg.embedding)
        llm = client_from_config(self.cfg.llm, self.pricing)
        rag = NaiveRAG(
            llm_client=llm,
            index=index,
            reader=reader,
            embedding_model=semantic.embedding_model,
            embedding_lock=semantic._embedding_lock,
            top_k=self.cfg.naive_rag.top_k,
            cite_chunks=True,  # the fast channel must cite (D29)
        )
        start = time.perf_counter()
        result = rag.run(question)
        record = {"pred_answer": result.get("answer", ""), "trajectory": result.get("trajectory")}
        return DocAnswer(
            channel="naive_rag",
            question=question,
            answer=str(result.get("answer") or ""),
            citations=self._citations(record),
            steps=[{"tool": "dense_retrieval", "args": {"top_k": self.cfg.naive_rag.top_k}}],
            cost=llm.usage_summary().cost,
            latency_s=round(time.perf_counter() - start, 2),
        )

    def answer(self, question: str, channel: Channel = "arag") -> DocAnswer:
        return self.arag(question) if channel == "arag" else self.naive(question)
