"""LLM client that meters every call (tokens, latency, cost) and retries transient failures."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from typing import Any

import requests
from pydantic import BaseModel

from arag.core.llm import LLMClient
from research_agent.config import LLMConfig
from research_agent.llm.pricing import ModelPricing, PricingTable

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class LLMCallRecord(BaseModel):
    """One logical chat call (possibly several HTTP attempts)."""

    latency_ms: float
    attempts: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    finish_reason: str | None = None
    response_text: str | None = None
    error: str | None = None


class UsageSummary(BaseModel):
    calls: int
    failed_calls: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost: float
    latency_ms: float


class MeteredLLMClient(LLMClient):
    """Drop-in replacement for :class:`arag.core.llm.LLMClient`.

    Differences from the upstream client:
    - cost comes from a tiered :class:`ModelPricing` (e.g. CNY for Bailian) instead of the
      built-in USD table, which has no Qwen entries;
    - ``extra_body`` is merged into the request (e.g. ``{"enable_thinking": False}``);
    - 429/5xx/timeouts are retried with exponential backoff;
    - every call is appended to ``self.records``.
    """

    def __init__(
        self,
        *,
        model: str,
        pricing: ModelPricing,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        extra_body: dict[str, Any] | None = None,
        timeout_s: float = 300.0,
        max_retries: int = 4,
        backoff_base_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,  # type: ignore[arg-type]  # upstream: implicit Optional, env fallback
            base_url=base_url,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.pricing = pricing
        self.extra_body = dict(extra_body or {})
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._sleep = sleep
        self.records: list[LLMCallRecord] = []
        self._lock = threading.Lock()

    def calculate_cost(self, usage: dict[str, Any]) -> float:
        prompt = usage.get("prompt_tokens", 0) or 0
        completion = usage.get("completion_tokens", 0) or 0
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0) or 0
        return self.pricing.cost(prompt, completion, cached)

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        # Mirrors LLMClient.chat; reasoning_effort is intentionally unsupported (OpenAI-only).
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        payload.update(self.extra_body)
        return payload

    def _post_with_retry(self, payload: dict[str, Any], attempts: list[int]) -> dict[str, Any]:
        """POST with retries; ``attempts[0]`` is updated so callers see it even on failure."""
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        while True:
            attempts[0] += 1
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                if response.status_code in RETRYABLE_STATUS and attempts[0] <= self.max_retries:
                    raise _Retryable(f"HTTP {response.status_code}: {response.text[:200]}")
                if not response.ok:
                    raise requests.HTTPError(
                        f"HTTP {response.status_code}: {response.text[:300]}", response=response
                    )
                body: dict[str, Any] = response.json()
                return body
            except (_Retryable, requests.Timeout, requests.ConnectionError):
                if attempts[0] > self.max_retries:
                    raise
                delay = self.backoff_base_s * 2 ** (attempts[0] - 1)
                self._sleep(delay * (0.5 + random.random()))

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload = self._build_payload(messages, tools, temperature, max_tokens)
        attempts = [0]
        start = time.perf_counter()
        try:
            result = self._post_with_retry(payload, attempts)
        except Exception as e:
            self._append(
                LLMCallRecord(
                    latency_ms=(time.perf_counter() - start) * 1000,
                    attempts=attempts[0],
                    error=f"{type(e).__name__}: {e}"[:500],
                )
            )
            raise
        latency_ms = (time.perf_counter() - start) * 1000

        usage = result.get("usage") or {}
        choice = result["choices"][0]
        message = choice["message"]
        cost = self.calculate_cost(usage)
        details = usage.get("prompt_tokens_details") or {}
        self._append(
            LLMCallRecord(
                latency_ms=latency_ms,
                attempts=attempts[0],
                prompt_tokens=usage.get("prompt_tokens", 0) or 0,
                completion_tokens=usage.get("completion_tokens", 0) or 0,
                cached_tokens=details.get("cached_tokens", 0) or 0,
                cost=cost,
                finish_reason=choice.get("finish_reason"),
                response_text=(message.get("content") or "")[:500],
            )
        )
        return {
            "message": message,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cost": cost,
            "raw_response": result,
        }

    def _append(self, record: LLMCallRecord) -> None:
        with self._lock:
            self.records.append(record)

    def usage_summary(self) -> UsageSummary:
        with self._lock:
            records = list(self.records)
        return UsageSummary(
            calls=len(records),
            failed_calls=sum(r.error is not None for r in records),
            prompt_tokens=sum(r.prompt_tokens for r in records),
            completion_tokens=sum(r.completion_tokens for r in records),
            cached_tokens=sum(r.cached_tokens for r in records),
            cost=sum(r.cost for r in records),
            latency_ms=sum(r.latency_ms for r in records),
        )


class _Retryable(Exception):
    """Raised internally for retryable HTTP statuses."""


def client_from_config(cfg: LLMConfig, pricing: PricingTable) -> MeteredLLMClient:
    return MeteredLLMClient(
        model=cfg.model,
        pricing=pricing.get(cfg.model),
        base_url=cfg.resolved_base_url(),
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        extra_body=cfg.extra_body(),
        timeout_s=cfg.timeout_s,
        max_retries=cfg.max_retries,
    )
