Agent model `qwen3.7-plus-2026-05-26`, judge `qwen-plus-2025-12-01`, n = 131. LLM-Acc and Cont-Acc are over all questions; the McNemar test and the confidence interval (Newcombe, paired) compare each run with `arag_full` question by question.

| Variant | LLM-Acc | Δ vs arag_full | 95% CI of Δ | McNemar p (ref-only / run-only) | Cont-Acc | descriptive (n=43) | direct (n=50) | version (n=38) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_rag | 82.4% | -16.0 pp | [-23.3, -9.9] | 0.000 (21 / 0) | 70.2% | 72.1% | 88.0% | 86.8% |
| **arag_full** | 98.5% | — | — | — | 78.6% | 95.3% | 100.0% | 100.0% |

| Variant | Loops | Tool calls (kw / sem / read) | Repeat read | Repeat search | Hit max loops | Input tokens | Cache hit | Retrieved tokens | Latency P50 / P95 (s) | Agent cost / 100 q (CNY) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_rag | 1.00 | dense retrieval × 1 | — | — | 0.0% | 4,888 | 1.4% | 4,788 | 5.0 / 9.5 | 1.08 |
| arag_full | 3.50 | 2.50 (1.32 / 0.03 / 1.15) | 1.3% | 0.0% | 0.0% | 10,739 | 63.1% | 2,877 | 13.1 / 29.8 | 1.38 |

| Variant | Answers citing a chunk | Citations read in full | Citations seen only as a snippet | Citations never shown | Answers citing a gold evidence chunk |
| --- | --- | --- | --- | --- | --- |
| naive_rag | 96.9% | 100.0% | 0.0% | 0 | 84.7% (n=131) |
| arag_full | 90.1% | 100.0% | 0.0% | 0 | 87.0% (n=131) |

Settings that differ between rows (everything else is identical):

- **naive_rag** (`results/stage4/tidb_baseline/naive_rag`, commit `f74ec9a`): `method="naive_rag"`, `naive_rag.cite_chunks=true`
- **arag_full** (`results/stage4/tidb_baseline/arag_full`, commit `f74ec9a`): `method="arag"`, `naive_rag.cite_chunks=false`
