# 阶段 8：故障诊断场景（2026-11-03）

本文件有三部分：首轮场景结果、强制作答的缓存实验（负面结论，D34）、分类器修正后的第二轮（D35）。

助手配置 `configs/assistant.yaml`：分类器 qwen3.7-flash，诊断 Agent qwen3.7-plus，文档检索全部走 A-RAG。集群为本地 tiup playground v8.5.8，示例库 `shop`。每个场景跑一次，场景定义见 `research_agent/diagnosis/scenarios.py`，评测方法见 design_decisions.md D34。

## 首轮场景结果

| 场景 | 路由 | 诊断正确 | 处理建议 | 引用相关文档 | 工具调用 | 未经确认的写操作 | 耗时 (s) | 花费（元） |
| --- | --- | :---: | --- | :---: | ---: | ---: | ---: | ---: |
| 缺少索引导致慢查询 | out_of_scope | ✗ | ✗ 未提出 | ✗ | 0 | 0 | 1.5 | 0.0001 |
| 单条 SQL 内存上限过小 | diagnosis | ✓ | ✓ `SET GLOBAL tidb_mem_quota_query = 1073741824`（批准） | ✓ | 11 | 0 | 83.4 | 0.0628 |
| 悲观锁等待超时 | diagnosis | ✓ | ✓ 未提出修改（没有写操作能解决，正确） | ✗ | 8 | 0 | 40.0 | 0.0220 |
| 统计信息过期导致执行计划变差 | diagnosis | ✓ | ✓ `ANALYZE TABLE orders`（批准）；`CREATE INDEX … (user_id, status, amount)`（拒绝） | ✗ | 8 | 0 | 51.3 | 0.0359 |

诊断正确 3 / 4，处理建议正确 3 / 4，引用相关文档 1 / 4，未经确认的写操作 0，共 0.12 元。

- 统计信息过期一行是修复场景脚本后的运行。第一次运行时场景没有造出故障：之前手动试验留下的统计信息已经包含偏斜数据。那次运行作废，但其中 Agent 无视工具结果、编造“索引不可见”，作为坏例记录（bad_cases.md 第 4 条）。
- 缺少索引失败在路由：分类器把只描述业务症状的问题判为越界（bad_cases.md 第 3 条）。

**补充场景**（不计入上表）：同一个缺索引故障，症状前加一句“我们的订单库跑在 TiDB 上。”

| 场景 | 路由 | 诊断正确 | 处理建议 | 引用相关文档 | 工具调用 | 未经确认的写操作 | 耗时 (s) | 花费（元） |
| --- | --- | :---: | --- | :---: | ---: | ---: | ---: | ---: |
| 缺少索引（症状里说明是 TiDB） | diagnosis | ✓ | ✓ `CREATE INDEX idx_status_created_at ON shop.orders (status, created_at)`（批准）；`ANALYZE TABLE shop.orders`（拒绝） | ✓ | 18 | 0 | 205.5 | 0.2508 |

## 强制作答的缓存（只做的一项改进）

用阶段 2 MuSiQue A-RAG 运行中以强制作答结束的真实对话回放（`research_agent/eval/cache_probe.py`）。每段对话只用于一种做法：先重发循环的最后一次请求作为预热，再发强制作答请求。

| 做法 | 对话数 | 强制作答提示 token（平均） | 缓存命中 | 强制作答花费（平均，元） | 仍然调用工具 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 不带工具，新开用户消息（上游） | 5 | 18,086 | 0% | 0.0383 | 0 |
| 带工具 + `tool_choice: "none"`，新开用户消息 | 5 | 14,936 | 0% | 0.0320 | 0 |
| 带工具 + auto，新开用户消息 | 5 | 20,158 | 6% | 0.0406 | 0 |
| 带工具 + auto，新开用户消息（预热时正常生成，间隔 2 秒） | 3 | 19,183 | 6% | 0.0383 | 0 |
| 带工具 + auto，指令附在最后一条工具结果里 | 3 | 15,374 | 97% | 0.0081 | 2 |

阶段 2 的记录：25 次强制作答的缓存命中都是 0，而它们之前的循环调用有 90% 的输入命中缓存；强制作答共 0.93 元，占整次运行的 15%。

## 第二轮：修正分类器、要求诊断引用文档之后（D35）

### 分类器测试集（`evalsets/router/questions.json`，只测分类器）

修改前（`results/stage8/router_before/`）：

| 组 | 期望 | 题数 | 准确率 | 分类分布 |
| --- | --- | ---: | ---: | --- |
| symptom | diagnosis | 20 | 75%（15） | diagnosis 15, out_of_scope 5 |
| unrelated | out_of_scope | 10 | 100%（10） | out_of_scope 10 |
| param | param_lookup | 10 | 100%（10） | param_lookup 10 |
| version | research | 10 | 90%（9） | research 9, param_lookup 1 |

花费 0.0034 元。

修改后（`results/stage8/router_after/`）：

| 组 | 期望 | 题数 | 准确率 | 分类分布 | 与修改前相同 |
| --- | --- | ---: | ---: | --- | ---: |
| symptom | diagnosis | 20 | 100%（20） | diagnosis 20 | 15 / 20 |
| unrelated | out_of_scope | 10 | 100%（10） | out_of_scope 10 | 10 / 10 |
| param | param_lookup | 10 | 100%（10） | param_lookup 10 | 10 / 10 |
| version | research | 10 | 100%（10） | research 10 | 9 / 10 |

花费 0.0046 元。

### 4 个默认场景重跑（`results/stage8/scenarios_round2/`，花费上限 1.5 元）

“回答引用文档”按最终回答实际写出的块编号或文档文件名统计，首轮结果也按同样的口径重新计算（首轮表中“引用相关文档”一列是按检索结果统计的，口径不同）。

| 场景 | 运行 | 路由 | 诊断正确 | 处理建议 | 回答引用文档 | 引用相关文档 | 工具调用 | 未经确认的写操作 | 耗时 (s) | 花费（元） |
| --- | --- | --- | :---: | --- | :---: | :---: | ---: | ---: | ---: | ---: |
| 缺少索引导致慢查询 | 修改前 | out_of_scope | ✗ | ✗ 未提出 | ✗ | ✗ | 0 | 0 | 1.5 | 0.0001 |
| 缺少索引导致慢查询 | 修改后 | diagnosis | ✓ | ✓ CREATE INDEX idx_status_created_at ON shop.orders (status, created_at)（批准）；ANALYZE TABLE shop.orders（拒绝） | ✓ | ✓ | 13 | 0 | 352.3 | 0.4121 |
| 单条 SQL 内存上限过小 | 修改前 | diagnosis | ✓ | ✓ SET GLOBAL tidb_mem_quota_query = 1073741824（批准） | ✗ | ✗ | 11 | 0 | 83.4 | 0.0628 |
| 单条 SQL 内存上限过小 | 修改后 | diagnosis | ✗ | ✓ SET GLOBAL tidb_mem_quota_query = 1073741824（批准） | ✗ | ✗ | 11 | 0 | 133.4 | 0.1255 |
| 悲观锁等待超时 | 修改前 | diagnosis | ✓ | ✓ 未提出修改（正确） | ✗ | ✗ | 8 | 0 | 40.0 | 0.0220 |
| 悲观锁等待超时 | 修改后 | diagnosis | ✓ | ✓ 未提出修改（正确） | ✓ | ✓ | 7 | 0 | 99.8 | 0.0779 |
| 统计信息过期导致执行计划变差 | 修改前 | diagnosis | ✓ | ✓ ANALYZE TABLE orders（批准）；CREATE INDEX idx_user_id_status_amount ON orders (user_id, status, amount)（拒绝） | ✗ | ✗ | 8 | 0 | 51.3 | 0.0359 |
| 统计信息过期导致执行计划变差 | 修改后 | diagnosis | ✓ | ✓ ANALYZE TABLE orders（批准）；CREATE INDEX idx_user_id_status_amount ON orders (user_id, status, amount)（拒绝） | ✓ | ✓ | 11 | 0 | 228.2 | 0.2471 |

修改前：诊断正确 3 / 4，处理建议正确 3 / 4，回答引用文档 0 / 4，引用相关文档 0 / 4，未经确认的写操作 0，花费 0.1208 元。

修改后：诊断正确 3 / 4，处理建议正确 4 / 4，回答引用文档 3 / 4，引用相关文档 3 / 4，未经确认的写操作 0，花费 0.8626 元。

- 内存上限场景这次判为诊断错误：Agent 找到了根因（修改提议的理由写着“当前值 1048576（1 MB）过低”），但确认执行后的最终回答只写了“已恢复为 1 GiB”，没有复述根因和依据（bad_cases.md 第 6 条）。
- 花费和耗时明显上升：每次 `search_docs` 是一整次 A-RAG，诊断中的开放式问题（如“复合索引的创建原则”）每次约 130 秒、0.13–0.18 元。

