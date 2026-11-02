Agent model `qwen3.7-plus-2026-05-26`, judge `qwen-plus-2025-12-01`, n = 100. LLM-Acc and Cont-Acc are over all questions; the McNemar test and the confidence interval (Newcombe, paired) compare each run with `arag_full` question by question.

| Variant | LLM-Acc | Δ vs arag_full | 95% CI of Δ | McNemar p (ref-only / run-only) | Cont-Acc | 2hop (n=55) | 3hop (n=34) | 4hop (n=11) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_rag | 46.0% | -25.0 pp | [-36.0, -12.9] | 0.000 (33 / 8) | 61.0% | 47.3% | 38.2% | 63.6% |
| **arag_full** | 71.0% | — | — | — | 83.0% | 78.2% | 61.8% | 63.6% |
| arag_full_stage0 | 68.0% | -3.0 pp | [-9.8, +3.8] | 0.549 (7 / 4) | 83.0% | 74.5% | 58.8% | 63.6% |
| wo_keyword | 66.0% | -5.0 pp | [-14.0, +4.1] | 0.383 (13 / 8) | 78.0% | 72.7% | 52.9% | 72.7% |
| wo_semantic | 68.0% | -3.0 pp | [-9.8, +3.8] | 0.549 (7 / 4) | 78.0% | 76.4% | 58.8% | 54.5% |
| wo_read | 69.0% | -2.0 pp | [-9.6, +5.6] | 0.791 (8 / 6) | 77.0% | 76.4% | 58.8% | 63.6% |

| Variant | Loops | Tool calls (kw / sem / read) | Repeat read | Repeat search | Hit max loops | Input tokens | Cache hit | Retrieved tokens | Latency P50 / P95 (s) | Agent cost / 100 q (CNY) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_rag | 1.00 | dense retrieval × 1 | — | — | 0.0% | 5,807 | 0.1% | 5,364 | 4.3 / 20.9 | 1.38 |
| arag_full | 8.98 | 8.24 (4.39 / 1.59 / 2.26) | 9.7% | 0.0% | 26.0% | 68,137 | 75.0% | 7,457 | 28.8 / 75.6 | 6.35 |
| arag_full_stage0 | 9.03 | 8.27 (4.47 / 1.57 / 2.23) | 8.3% | 0.0% | 22.0% | 66,629 | 75.9% | 7,366 | 40.5 / 106.3 | 6.15 |
| wo_keyword | 9.42 | 8.71 (0.00 / 6.05 / 2.66) | 7.3% | 0.0% | 29.0% | 52,291 | 72.2% | 4,952 | 35.8 / 73.0 | 5.37 |
| wo_semantic | 8.75 | 7.97 (6.14 / 0.00 / 1.83) | 4.9% | 0.5% | 22.0% | 68,388 | 72.8% | 8,226 | 30.3 / 80.6 | 6.65 |
| wo_read | 6.68 | 5.71 (4.19 / 1.52 / 0.00) | 31.3% | 0.2% | 3.0% | 136,312 | 68.2% | 25,668 | 33.8 / 84.7 | 13.13 |

Settings that differ between rows (everything else is identical):

- **naive_rag** (`results/stage1/musique100/naive_rag`, commit `77b9fd3`): `agent.search_result="snippet"`, `agent.tools=["keyword_search", "semantic_search", "read_chunk"]`, `method="naive_rag"`
- **arag_full** (`results/stage1/musique100/arag_full`, commit `2665488`): `agent.search_result="snippet"`, `agent.tools=["keyword_search", "semantic_search", "read_chunk"]`, `method="arag"`
- **arag_full_stage0** (`results/stage0/musique100_plus37`, commit `3ae98c7`): `agent.search_result="snippet"`, `agent.tools=["keyword_search", "semantic_search", "read_chunk"]`, `method="arag"`
- **wo_keyword** (`results/stage1/musique100/wo_keyword`, commit `2665488`): `agent.search_result="snippet"`, `agent.tools=["semantic_search", "read_chunk"]`, `method="arag"`
- **wo_semantic** (`results/stage1/musique100/wo_semantic`, commit `2665488`): `agent.search_result="snippet"`, `agent.tools=["keyword_search", "read_chunk"]`, `method="arag"`
- **wo_read** (`results/stage1/musique100/wo_read`, commit `2665488`): `agent.search_result="full_chunk"`, `agent.tools=["keyword_search", "semantic_search"]`, `method="arag"`
