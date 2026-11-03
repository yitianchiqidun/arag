# TiDB 运维诊断助手（基于 A-RAG）

一个面向 TiDB 的运维问答与故障诊断 Agent：用 [A-RAG](https://arxiv.org/abs/2602.03442) 的分层检索在 TiDB 官方中文文档里找依据，用只读账号查看本地 TiDB 集群的实际状态，需要修改集群时先由用户确认再执行。

- **参数和版本问答**：系统变量、配置项、错误码，以及 v8.1 → v8.5 之间的变化。回答带块级引用（文档 › 章节）。
- **故障诊断**：慢查询、内存超限、锁等待、统计信息过期等。12 个只读工具查看集群；3 类写操作（`SET GLOBAL`、`ANALYZE TABLE`、建索引）只生成提议，用户确认后才执行，并写审计日志。
- **服务化**：LangGraph 主图、多轮会话（SQLite 检查点）、流式事件；FastAPI + SSE 接口和 Streamlit 演示页。

本仓库从 A-RAG 的官方实现分叉而来。A-RAG 的源码（`src/arag/`、`scripts/`）没有改动，本项目的代码都在 `research_agent/`，详见文末“与 A-RAG 和 TiDB 文档的关系”。

## 架构

```mermaid
flowchart LR
    user([用户]) --> api["FastAPI + SSE<br/>research_agent/service"]
    demo["Streamlit 演示页"] --> api
    api --> graph
    subgraph graph["LangGraph 主图（research_agent/assistant）"]
        direction LR
        condense["改写追问"] --> route["意图分类"]
        route -- "参数 / 研究" --> retrieve["文档检索<br/>A-RAG"]
        route -- 诊断 --> diagnose["诊断循环"]
        diagnose -- 提议修改 --> confirm["interrupt()<br/>等用户确认"]
        confirm --> diagnose
        route -- 越界 --> refuse["说明能力范围"]
        retrieve --> check["证据检查"]
        diagnose --> check
    end
    retrieve --> docs[("TiDB 中文文档<br/>579 块，v8.1 + v8.5")]
    diagnose --> docs
    diagnose -- 只读账号 --> tidb[("本地 TiDB v8.5.8")]
    confirm -- 确认后，管理员账号 --> tidb
```

节点、事件和安全边界的说明见 [docs/architecture.md](docs/architecture.md)。

## 主要结果

所有数字都来自真实运行，模型为百炼 qwen3.7-plus（非思考模式），裁判为 qwen-plus-2025-12-01。完整数字、出处和局限见 [docs/project_summary.md](docs/project_summary.md)。

**A-RAG 复现**（MuSiQue 100 题，[对比表](docs/results/stage1_musique100_comparison.md)）

| | LLM-Acc | 每 100 题花费 |
| --- | ---: | ---: |
| 普通 RAG | 46.0% | 1.38 元 |
| A-RAG (Full) | 71.0%（论文 GPT-5-mini 全量：74.1%） | 6.35 元 |

- A-RAG 比普通 RAG 高 25 个百分点，p < 0.001。
- 三个工具消融的差距（2–5 个百分点）和同一配置重跑的波动（约 3 个百分点）同一量级，没有检出显著差异。

**TiDB 评测集上的快慢两个通道**（126 题，未经人工复核，[对比表](docs/results/stage4_tidb_baseline.md)）

| | 准确率 | 延迟 P50 / P95 | 每 100 题花费 |
| --- | ---: | ---: | ---: |
| 普通 RAG（要求引用） | 84.1% | 4.9 / 9.5 s | 1.08 元 |
| A-RAG | 100% | 13.1 / 29.8 s | 1.37 元 |
| 按意图分流（参数题走普通 RAG） | 85.7% | 7.3 / 36.3 s | 1.37 元 |

分流准确率低 14 个百分点，花费却没有降低，所以助手默认全部走 A-RAG（[D32](docs/design_decisions.md)）。

**故障诊断场景**（4 个注入的故障，每个跑一次，[结果](docs/results/stage8_scenarios.md)）

| | 诊断正确 | 处理建议正确 | 回答引用文档 | 未经确认的写操作 |
| --- | ---: | ---: | ---: | ---: |
| 首轮 | 3 / 4 | 3 / 4 | 0 / 4 | 0 |
| 修正分类器、要求引用文档后 | 3 / 4 | 4 / 4 | 3 / 4 | 0 |

## 快速开始

需要 Python 3.10+、[uv](https://docs.astral.sh/uv/)、一块 NVIDIA GPU（嵌入模型 Qwen3-Embedding-0.6B；改用 CPU 时把配置里的 `embedding.device` 设为 `cpu`）和百炼（或其他 OpenAI 兼容）API。

```bash
uv sync --extra full --extra research --extra agent --extra tidb --extra service --extra viewer
```

```bash
cp .env.example .env   # 填 ARAG_API_KEY、ARAG_BASE_URL 和 TIDB_RO_PASSWORD
```

构建 TiDB 文档索引（克隆 pingcap/docs-cn 的 release-8.5 和 release-8.1 分支，取运维相关的 65 篇文档）：

```bash
uv run python -m research_agent.ingest fetch configs/corpora/tidb_docs.yaml
```

```bash
uv run python -m research_agent.ingest build configs/corpora/tidb_docs.yaml --ref origin/release-8.5 --out data/tidb_docs/current
```

启动本地 TiDB（需要先安装 [tiup](https://docs.pingcap.com/zh/tidb/stable/tiup-overview)），导入示例库并创建只读账号：

```bash
uv run python -m research_agent.tidb start
```

```bash
uv run python -m research_agent.tidb setup
```

```bash
uv run python -m research_agent.tidb verify
```

使用：

```bash
uv run python -m research_agent.assistant "tidb_mem_quota_query 的默认值是多少？"
```

```bash
uv run uvicorn research_agent.service.app:app --port 8000
```

```bash
curl -N -X POST localhost:8000/chat/stream -H 'Content-Type: application/json' -d '{"message": "订单列表页最近很慢，帮我看看"}'
```

```bash
CHAT_DEMO_API_URL=http://127.0.0.1:8000 uv run streamlit run apps/chat_demo.py
```

接口：`POST /chat`（同步）、`POST /chat/stream`（SSE，事件类型：route、retrieval_start、tool_call、tool_result、confirm_required、answer、error、done）、`POST /chat/confirm`（确认或拒绝待执行的修改）、`GET /sessions/{id}`、`GET /health`。

**Docker**：

```bash
docker compose up -d --build --wait
```

会启动 API（127.0.0.1:8000）和演示页（127.0.0.1:8501），两个服务共用同一个镜像。

- **镜像**：用 CPU 版 torch，容器里只做查询编码；镜像约 2.0 GB，其中 torch 占 0.7 GB。
- **TiDB**：不在 compose 里，仍由 tiup 在宿主机上运行，容器经 `host.docker.internal:4000` 连接。在 Docker Desktop + WSL2 上实测：宿主机网络模式连不上 WSL 里的 127.0.0.1，桥接网络可以。
- **挂载**：文档索引 `data/tidb_docs` 和嵌入模型的 Hugging Face 缓存只读挂载，容器内不重建索引、不下载模型；会话、轨迹和审计日志写到 `data/assistant_docker/`。
- **验收**（2026-11-03，Docker Desktop，Engine 29.8，WSL2）：curl 调流式接口，事件逐步返回；在演示页容器里完成了一次参数对话和一次带写操作确认的诊断。见 `docs/experiments.md` “阶段 9”。

```bash
docker compose exec -T demo python -m research_agent.service.demo_check
```

**换成本地模型**：`configs/assistant_local.yaml` 演示如何把诊断 Agent 换成本地 vLLM 等 OpenAI 兼容服务（`base_url`、`api_key_env`、`local: true` 按零成本计费）。

**测试**：`uv run pytest tests`（全部使用假模型和假数据库）。连接真实集群的测试要先设置 `TIDB_LIVE=1`。CI 见 `.github/workflows/ci.yml`。

**复现评测**：`python -m research_agent.eval suite configs/suites/stage4_tidb_baseline.yaml`（TiDB 评测集），`python -m research_agent.diagnosis`（故障场景，需要本地集群），`python -m research_agent.assistant.router_eval run --out …`（分类器）。

## 关键设计决策

完整记录在 [docs/design_decisions.md](docs/design_decisions.md)，每条按“背景 → 备选方案 → 最终选择 → 理由 → 代价与局限”写。

| 决策 | 内容 |
| --- | --- |
| D1 | 扩展 A-RAG 只用继承和包装，不改源码 |
| D13、D17 | 对比统一用全部题目上的 LLM-Acc，配对 McNemar 检验加置信区间；根据重跑噪声估算样本量 |
| D23、D29 | 从答案中解析块编号作为引用；普通 RAG 也要求引用，两个通道的引用率才能比较 |
| D26 | 增量索引以上一次构建为缓存，验证与全量构建逐块一致 |
| D30 | 评测题的答案由规则从参数手册和发布说明中抽取，并做闭卷过滤 |
| D31 | 一张 LangGraph 图；A-RAG 整体作为一个工具；修改用 `interrupt()` 等确认 |
| D32 | 分类路由不划算，默认全部走 A-RAG；A-RAG 没写引用时，用它读过的块作为依据 |
| D33 | 只读账号的权限范围，用真实集群上的写探测验证 |
| D34、D35 | 故障场景用规则打分；强制作答的缓存方案不可行（负面结论）；改为修正分类器 |
| D36 | 服务化：每会话一轮、并发上限、单轮超时、步数用完时强制给出结论 |

## 目录

```
research_agent/
  ingest/      TiDB 文档清洗、中文切句、按标题切块、增量索引
  agent/       A-RAG 的工具集与提示词（消融开关）
  baselines/   普通 RAG
  eval/        评测运行器、裁判、配对统计、对比表、缓存探测
  evalset/     TiDB 评测集构建（规则抽取、闭卷过滤、自动预审）
  tracing/     轨迹记录与回放
  tidb/        本地集群、只读工具、需确认的写操作
  assistant/   LangGraph 主图、意图分类、文档检索通道
  diagnosis/   故障注入场景与端到端评测
  service/     FastAPI 服务与客户端
apps/          演示页、轨迹查看器、评测集审核页
configs/       运行、评测套件、助手配置
evalsets/      TiDB 评测集（126 题）、分类器测试集（50 题）
docs/          设计决策、实验记录、架构、坏例、结果表
src/, scripts/ A-RAG 原始代码（未改动）
```

## 与 A-RAG 和 TiDB 文档的关系

- **A-RAG**：本仓库从 [Ayanami0730/arag](https://github.com/Ayanami0730/arag)（MIT 许可）分叉，论文为 *A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces*（[arXiv:2602.03442](https://arxiv.org/abs/2602.03442)）。分层检索工具、Agent 循环和系统提示词都来自原作者，原 README 保留在 [docs/upstream_README.md](docs/upstream_README.md)，对原有文件的少量改动（只涉及依赖配置）记录在 [docs/upstream_changes.md](docs/upstream_changes.md)。本项目新增的代码沿用 MIT 许可。
- **TiDB 文档**：检索语料来自 [pingcap/docs-cn](https://github.com/pingcap/docs-cn)（CC BY-SA 3.0），构建时克隆，不随本仓库分发。`evalsets/tidb/` 中的题目和证据含有文档原文摘录，按 CC BY-SA 3.0 分发。
- **TiDB**：本地集群由 tiup playground 启动；本项目与 PingCAP 无关联。
