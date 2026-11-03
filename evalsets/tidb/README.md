# TiDB 参数与版本变更评测集

基于 TiDB 中文官方文档 [`pingcap/docs-cn`](https://github.com/pingcap/docs-cn)（CC BY-SA 3.0）构建的问答评测集，用于阶段 4 的普通 RAG 与 A-RAG 基线对比。题目中的文档原文版权归 PingCAP 所有，本评测集按相同许可（CC BY-SA 3.0）发布。

- 语料：release-8.5 分支（提交 `1b354399`）的 TiDB 核心子集，65 篇（`configs/corpora/tidb_docs.yaml`）；版本对比另用 release-8.1 分支（提交 `d039c26`）。
- 设计决策：`docs/design_decisions.md` D30。

## 三类题

| 题型（`question_type`） | 题数 | 例子 | 答案来源 |
| --- | ---: | --- | --- |
| `direct` 直接问法 | 50 | TiKV 配置项 `readpool.unified.stack-size` 的默认值是多少？ | 参数文档，规则抽取 |
| `descriptive` 描述问法 | 50 | 只描述参数的作用、不出现参数名，问它的默认值 | 参数文档，规则抽取；问题由 qwen3.7-flash 改写 |
| `version` 版本变更 | 38 | 与 v8.1.0 相比，v8.5.8 中 TiDB 系统变量 `tidb_opt_prefer_range_scan` 的默认值有什么变化？ | 发布说明，规则抽取，并与两个版本的参数文档核对 |

## 文件

| 文件 | 内容 |
| --- | --- |
| `drafts.jsonl` | 抽样得到的题目草稿：抽取的事实、证据位置；描述问法附有“审核参考”（说明最相似的 5 个参数） |
| `build_report.json` | 各类事实的数量、版本变更题的丢弃原因和数量 |
| `candidates.jsonl` | 候选题（描述问法改写之后） |
| `meta.json` | 证据块编号对应的索引构建（`chunks.json` 的 sha256） |
| `closed_book.jsonl` | 闭卷过滤的作答记录 |
| `prereview.jsonl` | 描述问法的 LLM 预审结论 |
| `review.json` / `review_first.json` | 人工的最终判断 / 第一次判断（只写一次，用于一致率） |
| `questions.json` | 最终评测集：上游题目格式，另含证据块、证据文档、原文引用 |
| `memorized.json` | 闭卷可答的题，单独存放，不进入主指标 |

## 构建流程

1. `python -m research_agent.evalset.tidb_build configs/evalset/tidb_params_build.yaml draft`：抽取事实、过滤、抽样、定位证据块（不调用 API）。
2. `... rewrite`：改写描述问法（qwen3.7-flash，规则检查不通过时反馈改写一次）。
3. `python -m research_agent.evalset closed-book configs/evalset/tidb_params.yaml`：闭卷过滤（被测模型 qwen3.7-plus，temperature 0 答一次，答错再以 0.7 采样最多 3 次）。
4. `python -m research_agent.evalset prereview configs/evalset/tidb_params.yaml`：描述问法预审（qwen-plus-2025-12-01）。
5. 人工复核：`QA_REVIEW_CONFIG=configs/evalset/tidb_params.yaml uv run streamlit run apps/qa_review.py`。
6. `python -m research_agent.evalset finalize configs/evalset/tidb_params.yaml`：写出 `questions.json` 并更新下面的统计表。

抽取规则、过滤条件和版本核对规则见 D30。

## 审核标准

1. 答案正确：标准答案与证据原文一致（默认值、作用域、取值范围、版本号），必要时改写答案。
2. 问题自然、无歧义：只有一个正确答案；版本变更题比较的是 v8.1.0 和 v8.5.8 两个具体版本。
3. 描述问法唯一：问题只描述参数的作用、不出现参数名，并且只对应这一个参数（对照“审核参考”里列出的相似参数）。
4. 不泄露答案：问题里不出现标准答案。
5. 答案简短明确。

丢弃或修改时记录原因：描述不唯一、答案与原文不符、问题不自然、有歧义、问题泄露答案、其他。

## 审核方式

- 直接问法和版本变更题的答案由规则抽取，人工从两类中合计随机抽检 20 道（种子 2026），确认抽取规则没有错误。
- 描述问法：LLM 预审全部；人工复核预审判为修改或丢弃的题，加上从预审判为保留的题中随机抽检的 10 道（种子 2026）。
- 人工先在看不到预审结论的情况下保存第一次判断，一致率只用第一次判断计算；看到预审后改判的题数单独统计。

### 审核统计（finalize 自动生成）

<!-- evalset-review:start -->
（尚未定稿）
<!-- evalset-review:end -->

## 局限

- 只覆盖 TiDB、TiKV、PD 的参数；版本变更集中在 v8.4.0。
- 证据块编号对应 `meta.json` 记录的索引构建。语料或切块规则变化后，块编号会变，要用 `evidence_docs` 和 `evidence_quotes` 重新定位。
- 上游裁判只比较 `answer` 字段，等价写法需要写进答案文本。
