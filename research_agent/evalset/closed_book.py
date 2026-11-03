"""Closed-book filter: can the agent model answer a candidate without any documents?

The agent model answers once at temperature 0 (the evaluation setting); after a wrong answer
it gets ``samples`` more tries at a higher temperature. A correct answer at temperature 0
marks the candidate "closed_book", a correct sample "partial_memory"; both go to
``memorized.json`` instead of the main set, so the set measures retrieval rather than what
the model remembers. Answers are scored with the upstream judge prompt of the evaluation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import requests

from research_agent.config import RunConfig
from research_agent.evalset.dataset import (
    CLOSED_BOOK_FILE,
    Attempt,
    Candidate,
    ClosedBookConfig,
    ClosedBookResult,
    EvalsetConfig,
    current,
    load_candidates,
    load_closed_book,
)
from research_agent.evalset.prompts import CLOSED_BOOK_TEMPLATE
from research_agent.llm.client import AccountError, MeteredLLMClient, client_from_config
from research_agent.llm.pricing import PricingTable
from research_agent.upstream import load_script


class ClosedBookChecker:
    """Runs the closed-book check for one candidate; safe to call from several threads."""

    def __init__(
        self, run_config: RunConfig, pricing: PricingTable, subject: str, cfg: ClosedBookConfig
    ) -> None:
        self.run_config = run_config
        self.pricing = pricing
        self.subject = subject
        self.cfg = cfg
        self._evaluator = load_script("eval").Evaluator

    def _attempt(
        self, llm: MeteredLLMClient, judge: MeteredLLMClient, cand: Candidate, temperature: float
    ) -> Attempt:
        prompt = CLOSED_BOOK_TEMPLATE.format(subject=self.subject, question=cand.question)
        result = llm.chat(messages=[{"role": "user", "content": prompt}], temperature=temperature)
        reply = str(result["message"].get("content") or "").strip()
        evaluator = SimpleNamespace(llm_client=judge)
        correct = bool(reply) and (
            self._evaluator.calculate_llm_accuracy(evaluator, reply, cand.answer) == 1.0
        )
        return Attempt(temperature=temperature, reply=reply, correct=correct)

    def check(self, cand: Candidate) -> ClosedBookResult:
        llm = client_from_config(self.run_config.llm, self.pricing)
        judge = client_from_config(self.run_config.judge, self.pricing)
        result = ClosedBookResult(
            candidate_id=cand.candidate_id,
            question=cand.question,
            answer=cand.answer,
            model=self.run_config.llm.model,
        )
        try:
            result.attempts.append(self._attempt(llm, judge, cand, 0.0))
            for _ in range(self.cfg.samples):
                if result.attempts[-1].correct:  # one correct answer is enough to mark it
                    break
                result.attempts.append(self._attempt(llm, judge, cand, self.cfg.temperature))
        except AccountError:
            raise  # stop everything: later calls would fail the same way
        except (requests.RequestException, KeyError) as e:  # recorded; a rerun retries it
            result.error = f"{type(e).__name__}: {e}"[:300]
        result.cost = round(sum(r.cost for r in llm.records + judge.records), 6)
        return result


def run_closed_book(cfg: EvalsetConfig, workers: int = 5) -> list[ClosedBookResult]:
    """Check every candidate that passed the builder's checks and has no current result."""
    candidates = [c for c in load_candidates(cfg) if c.reject_reason is None]
    done = {k for k, v in current(load_closed_book(cfg), candidates).items() if v.error is None}
    pending = [c for c in candidates if c.candidate_id not in done]
    run = cfg.load_run_config()
    checker = ClosedBookChecker(
        run, PricingTable.from_yaml(run.pricing_file), cfg.subject, cfg.closed_book
    )
    new: list[ClosedBookResult] = []
    path = cfg.out / CLOSED_BOOK_FILE
    with ThreadPoolExecutor(max_workers=workers) as pool, open(path, "a", encoding="utf-8") as f:
        futures = [pool.submit(checker.check, c) for c in pending]
        for future in as_completed(futures):
            row = future.result()  # an AccountError propagates and stops the run
            f.write(row.model_dump_json() + "\n")
            f.flush()
            new.append(row)
    return new
