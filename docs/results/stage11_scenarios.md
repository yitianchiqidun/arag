# 阶段 11：harness 之后的故障场景（2026-11-03）

助手配置 `configs/assistant.yaml`（工具在本进程内执行，经 `ToolHarness`）。每轮工具调用上限 16 次，文档检索上限 2 次，LLM 最多 12 步。每个场景只跑一次，结果在 `results/stage11/scenarios_round3/`。“修改前”是阶段 8 第二轮（`results/stage8/scenarios_round2/`，当时没有记录 LLM 步数）。

| 场景 | 运行 | 路由 | 诊断正确 | 处理建议 | 回答引用文档 | 引用相关文档 | 工具调用 | LLM 步数 | 未经确认的写操作 | 耗时 (s) | 花费（元） |
| --- | --- | --- | :---: | --- | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 缺少索引导致慢查询 | 修改前 | diagnosis | ✓ | ✓ CREATE INDEX idx_status_created_at ON shop.orders (status, created_at)（批准）；ANALYZE TABLE shop.orders（拒绝） | ✓ | ✓ | 13 | — | 0 | 352.3 | 0.4121 |
| 缺少索引导致慢查询 | 修改后 | diagnosis | ✓ | ✓ CREATE INDEX idx_status_created_at ON shop.orders (status, created_at)（批准）；ANALYZE TABLE shop.orders（拒绝） | ✓ | ✓ | 10 | 11 | 0 | 139.0 | 0.1606 |
| 单条 SQL 内存上限过小 | 修改前 | diagnosis | ✗ | ✓ SET GLOBAL tidb_mem_quota_query = 1073741824（批准） | ✗ | ✗ | 11 | — | 0 | 133.4 | 0.1255 |
| 单条 SQL 内存上限过小 | 修改后 | diagnosis | ✓ | ✗ 未提出 | ✓ | ✓ | 17 | 12 | 0 | 79.2 | 0.1119 |
| 悲观锁等待超时 | 修改前 | diagnosis | ✓ | ✓ 未提出修改（正确） | ✓ | ✓ | 7 | — | 0 | 99.8 | 0.0779 |
| 悲观锁等待超时 | 修改后 | diagnosis | ✓ | ✓ 未提出修改（正确） | ✗ | ✗ | 4 | 5 | 0 | 133.6 | 0.0932 |
| 统计信息过期导致执行计划变差 | 修改前 | diagnosis | ✓ | ✓ ANALYZE TABLE orders（批准）；CREATE INDEX idx_user_id_status_amount ON orders (user_id, status, amount)（拒绝） | ✓ | ✓ | 11 | — | 0 | 228.2 | 0.2471 |
| 统计信息过期导致执行计划变差 | 修改后 | diagnosis | ✓ | ✓ ANALYZE TABLE orders（批准） | ✓ | ✗ | 5 | 6 | 0 | 90.2 | 0.1167 |
| 报表账号被锁定（需要查日志） | 修改后 | diagnosis | ✓ | ✓ 未提出修改（正确） | ✓ | ✗ | 6 | 7 | 0 | 138.0 | 0.2289 |

修改前：诊断正确 3 / 4，处理建议正确 4 / 4，回答引用文档 3 / 4，引用相关文档 3 / 4，未经确认的写操作 0，花费 0.8626 元。

修改后：诊断正确 5 / 5，处理建议正确 4 / 5，回答引用文档 4 / 5，引用相关文档 2 / 5，未经确认的写操作 0，花费 0.7113 元。

- 新场景“报表账号被锁定”只能靠日志诊断：只读账号读不到 `mysql.user`。Agent 第一次调用就是 `search_logs`，找到了 `ALTER USER report@%` 的 CRUCIAL OPERATION 记录和 “Account is locked” 的登录失败记录。
- 内存上限场景诊断正确，但没有提出修改：第 12 次工具调用才查到被改小的变量，之后又查日志、查文档；第 17 次调用 `propose_set_global` 被“每轮 16 次”的调用预算拒绝，最后由步数用完时的总结给出结论。这是 harness 的设计缺陷，已改为需确认的工具不受调用预算限制（D38），修改后没有再跑。

MCP 模式（阶段 10，`results/stage10/mcp_scenario/`）：内存上限场景诊断正确，提出的修改经确认后由 Agent 执行；7 次工具调用、8 步，80.7 秒，0.083 元。
