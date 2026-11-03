# TiDB 运维诊断助手：架构

阶段 6 的主 Agent。代码在 `research_agent/assistant/`，配置 `configs/assistant.yaml`，设计取舍见 design_decisions.md D31。

## 主图

一轮对话是 LangGraph 主图的一次运行；同一会话是检查点里的同一个 thread。

```mermaid
flowchart LR
    START([用户问题]) --> condense["condense<br/>追问改写成完整问题"]
    condense --> route["route<br/>意图分类"]
    route -- "param_lookup / research" --> retrieve["retrieve<br/>文档检索（A-RAG 或快速通道）"]
    route -- diagnosis --> diagnose["diagnose<br/>工具调用循环"]
    route -- out_of_scope --> refuse["refuse<br/>说明能力范围"]
    diagnose -- "提出修改" --> confirm["confirm<br/>interrupt() 等用户确认"]
    confirm -- "确认 / 拒绝的结果" --> diagnose
    diagnose -- "给出结论" --> check
    retrieve --> check["check<br/>没有证据就不答"]
    check --> END([回答])
    refuse --> END
```

| 节点 | 模型 | 做什么 |
| --- | --- | --- |
| `condense` | router（qwen3.7-flash） | 有历史时，把“那这个参数在 v8.1 里呢？”改写成不看前文也能懂的问题；没有历史时不调用模型 |
| `route` | router | 分成参数查询、跨文档研究、故障诊断、越界四类；输出不合法时按“研究”处理 |
| `retrieve` | A-RAG 的 Agent（qwen3.7-plus） | 由 `Services.choose_channel(question, intent)` 选通道：A-RAG 多轮检索，或普通 RAG（要求引用，D29）。阶段 6 一律走 A-RAG，阶段 7 才按意图分流 |
| `diagnose` | agent（qwen3.7-plus） | 工具调用循环，最多 12 步：12 个只读集群工具、`search_docs`（整个 A-RAG 作为一个工具）、3 个 `propose_*` 修改工具 |
| `confirm` | — | 用 `interrupt()` 暂停整张图，等用户确认或拒绝；确认才执行，结果作为工具结果交还诊断循环 |
| `check` | — | 回答背后没有证据（没有读过或看过片段的引用块，也没有集群数据）就拒答；诊断时集群连不上则明确说明 |
| `refuse` | — | 越界问题，说明能做什么 |

## 数据流

```mermaid
flowchart TB
    subgraph 入口
        cli["CLI<br/>python -m research_agent.assistant"]
        demo["Streamlit 演示<br/>apps/chat_demo.py"]
    end
    app["AssistantApp（app.py）<br/>组装服务、串行化、写轨迹"]
    graph["Assistant（graph.py）<br/>主图 + 事件流"]
    cli --> app
    demo --> app
    app --> graph
    graph --> docs["DocRetriever（retrieval.py）<br/>A-RAG / 普通 RAG"]
    graph --> ro["ReadOnlyTools<br/>agent_ro 账号"]
    graph --> wt["WriteTools → WriteExecutor<br/>root 账号，只在确认后"]
    docs --> index[("TiDB 中文文档索引<br/>579 块，v8.1 + v8.5")]
    ro --> tidb[("本地 TiDB v8.5.8<br/>tiup playground")]
    wt --> tidb
    wt --> audit[("writes.jsonl 审计")]
    graph --> ckpt[("sessions.sqlite<br/>会话检查点")]
    app --> trace[("traces/会话.jsonl")]
```

- **会话**：检查点存在 SQLite（`data/assistant/sessions.sqlite`），`history` 字段按轮追加，追问能看到上文；等待确认的修改也存在检查点里，进程重启后仍能确认。
- **事件流**：节点通过 LangGraph 的 stream writer 发出事件，`Assistant` 再把 `interrupt` 转成 `confirm_required`，最后补一个 `done`。事件类型：`route`、`retrieval_start`、`tool_call`、`tool_result`、`confirm_required`、`answer`、`error`、`done`。阶段 9 的 SSE 接口直接转发这些事件。
- **轨迹**：`AssistantApp` 把每个事件和每次文档检索的完整步骤追加到 `data/assistant/traces/<会话>.jsonl`，可以事后复盘。
- **花费**：每个节点把模型花费累加到状态的 `cost`，随 `answer` 事件返回。

## 安全边界

| 层 | 措施 |
| --- | --- |
| 数据库权限 | 只读工具用 `agent_ro` 账号连接，只有 `shop.*` 的 SELECT 和全局 PROCESS 权限（`research_agent/tidb/setup.py`）；即使 SQL 检查被绕过，也写不了数据 |
| SQL 检查 | 自由 SQL 只允许 SELECT / SHOW / DESC / EXPLAIN，拒绝多语句和写关键字；`EXPLAIN ANALYZE` 只允许跟 SELECT（会真正执行） |
| 执行限制 | 会话级 `max_execution_time`（默认 10 秒），结果最多 200 行 |
| 修改 | 模型只能调用 `propose_*` 生成待确认的修改（语句、影响说明、回滚语句）；只有用户确认后 `WriteExecutor` 才用管理员账号执行，每次决定都写审计日志 |
| 一次一项 | 一轮里只接受第一项修改提议，其余的告诉模型等确认后再提 |

## 运行

```bash
python -m research_agent.assistant --session demo-1 "tidb_mem_quota_query 的默认值是多少？" "那这个参数在 v8.1 里的默认值呢？"
```

```bash
uv run streamlit run apps/chat_demo.py
```

集群（阶段 5，需要先安装 tiup）：`python -m research_agent.tidb start`、`setup`、`verify`。集群没启动时，助手照常回答文档问题，诊断问题会说明连不上集群。

## 一次真实对话（2026-11-03，会话 `demo-1`，没有集群）

| 轮 | 问题 | 路由 | 结果 | 耗时 | 花费（元） |
| --- | --- | --- | --- | ---: | ---: |
| 1 | tidb_mem_quota_query 的默认值是多少？ | 参数查询 | 1 GB，答案正确，但引用的是 explain-joins.md，引用偏弱 | 13.7 s | 0.0117 |
| 2 | 那这个参数在 v8.1 里的默认值呢？ | 参数查询 | 改写为“tidb_mem_quota_query 在 TiDB v8.1 版本中的默认值是多少？”，引用 Chunk 408，1073741824 | 31.9 s | 0.0313 |
| 3 | 与 v8.1.0 相比，v8.5.8 中哪些 TiKV 配置项的默认值发生了变化？ | 跨文档研究 | 列出 v8.4 等版本的 TiKV 默认值变更 | 127.9 s | 0.2172 |
| 4 | 今天北京天气怎么样？ | 越界 | 说明能力范围 | 2.7 s | 0.0002 |
| 5 | 我的订单查询突然变慢了，帮我看看 | 故障诊断 | 没有集群，拒答（之后改为专门说明集群连不上） | 15.5 s | 0.0089 |

共 0.27 元。跨文档研究一轮 128 秒，是阶段 7 分流要解决的问题之一。
