| 意图 | 暴露的工具 | 每次调用的工具定义 token（按意图） | 全部暴露 | 每次少 | 每题调用次数 | 每题少 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| param_lookup | 3 个：keyword_search、read_chunk、semantic_search | 831 | 2,383 | 1,552 | 3.48 | 5,401 |
| research | 3 个：keyword_search、read_chunk、semantic_search | 831 | 2,383 | 1,552 | 3.48 | 5,401 |
| diagnosis | 19 个：search_docs、run_sql、show_variables、slow_queries、explain、lock_waits、transactions、deadlocks、statement_summary、table_stats、table_schema、bindings、cluster_info、search_logs、show_config、export_report、propose_set_global、propose_analyze、propose_create_index | 1,554 | 2,383 | 829 | 8.2 | 6,798 |
| out_of_scope | 0 个：无 | 0 | 2,383 | 2,383 | 1 | 2,383 |

全部暴露时每次调用带 22 个工具定义，共 2,383 token（Qwen3 分词器）。每题调用次数：参数和研究类取阶段 4 A-RAG 的平均轮数，越界只有分类器的一次调用，诊断类取阶段 11 场景运行的平均 LLM 调用数。工具定义在请求前部，通常能命中缓存，所以省下的主要是上下文长度和首字延迟，花费的节省小于 token 数的比例。
