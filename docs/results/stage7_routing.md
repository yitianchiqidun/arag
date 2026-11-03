Agent model `qwen3.7-plus-2026-05-26`, judge `qwen-plus-2025-12-01`, n = 126. LLM-Acc and Cont-Acc are over all questions; the McNemar test and the confidence interval (Newcombe, paired) compare each run with `arag_full` question by question.

| Variant | LLM-Acc | Δ vs arag_full | 95% CI of Δ | McNemar p (ref-only / run-only) | Cont-Acc | descriptive (n=40) | direct (n=50) | version (n=36) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| routed | 85.7% | -14.3 pp | [-21.5, -8.4] | 0.000 (18 / 0) | 68.3% | 70.0% | 88.0% | 100.0% |
| **arag_full** | 100.0% | — | — | — | 78.6% | 100.0% | 100.0% | 100.0% |
| naive_rag | 84.1% | -15.9 pp | [-23.2, -9.8] | 0.000 (20 / 0) | 69.8% | 75.0% | 88.0% | 88.9% |

| Variant | Loops | Tool calls (kw / sem / read) | Repeat read | Repeat search | Hit max loops | Input tokens | Cache hit | Retrieved tokens | Latency P50 / P95 (s) | Agent cost / 100 q (CNY) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| routed | 1.79 | dense retrieval × 1 | 1.1% | 0.0% | 0.0% | 8,450 | 38.3% | 4,748 | 7.3 / 36.1 | 1.37 |
| arag_full | 3.48 | 2.48 (1.31 / 0.03 / 1.13) | 1.4% | 0.0% | 0.0% | 10,574 | 63.2% | 2,830 | 13.1 / 29.8 | 1.37 |
| naive_rag | 1.00 | dense retrieval × 1 | — | — | 0.0% | 4,888 | 1.5% | 4,787 | 4.9 / 9.5 | 1.08 |

| Variant | Answers citing a chunk | Citations read in full | Citations seen only as a snippet | Citations never shown | Answers citing a gold evidence chunk |
| --- | --- | --- | --- | --- | --- |
| routed | 96.0% | 100.0% | 0.0% | 0 | 87.3% (n=126) |
| arag_full | 89.7% | 100.0% | 0.0% | 0 | 87.3% (n=126) |
| naive_rag | 96.8% | 100.0% | 0.0% | 0 | 85.7% (n=126) |

Settings that differ between rows (everything else is identical):

- **routed** (`results/stage7/routing/routed`, commit `f5a619c`): `method="routed"`, `naive_rag.cite_chunks=true`, `routing.router.enable_thinking=false`, `routing.router.max_retries=4`, `routing.router.max_tokens=300`, `routing.router.model="qwen3.7-flash-2026-07-15"`, `routing.router.temperature=0.0`, `routing.router.timeout_s=300.0`
- **arag_full** (`results/stage4/tidb_baseline_126/arag_full`, commit `f74ec9a`): `method="arag"`, `naive_rag.cite_chunks=false`, `routing.router.enable_thinking=null`, `routing.router.max_retries=null`, `routing.router.max_tokens=null`, `routing.router.model=null`, `routing.router.temperature=null`, `routing.router.timeout_s=null`
- **naive_rag** (`results/stage4/tidb_baseline_126/naive_rag`, commit `f74ec9a`): `method="naive_rag"`, `naive_rag.cite_chunks=true`, `routing.router.enable_thinking=null`, `routing.router.max_retries=null`, `routing.router.max_tokens=null`, `routing.router.model=null`, `routing.router.temperature=null`, `routing.router.timeout_s=null`
