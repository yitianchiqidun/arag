# 对 A-RAG 原有文件的修改记录

原则：新功能写在 `research_agent/`，通过继承/包装扩展 A-RAG。只有在没有合理替代方案时才改动原有文件，并在此记录。

截至目前，**A-RAG 的 Python 源码（`src/arag/`、`scripts/`）未做任何修改**。改动仅限项目配置文件。

| 日期 | 文件 | 改了什么 | 为什么 | 替代方案及未采用原因 |
| --- | --- | --- | --- | --- |
| 2026-10-30 | `pyproject.toml` | 新增可选依赖组 `research`（pydantic、python-dotenv、torch、类型桩）；wheel 打包加入 `research_agent` 包；原有 `full`/`dev` 依赖组不变 | 新代码需要这些依赖，且要能 `import research_agent` | 单独建一个 `research_agent/pyproject.toml` 作为 uv workspace 成员：仍需改根 `pyproject.toml` 声明 workspace，且多一层包管理，收益不大 |
| 2026-10-30 | `pyproject.toml` | `[tool.uv.sources]` 把 torch 固定到 PyTorch 官方 cu128 源 | 本机 Windows 驱动 572.70 最高只支持 CUDA 12.8；PyPI 默认的 torch 2.15 是 CUDA 13.2 构建，`torch.cuda.is_available()` 为 False | 升级 Windows NVIDIA 驱动后可去掉此项。cu128 版本向后兼容更新的驱动，对其他机器无害。阶段 6 做 Docker 镜像时需重新评估（服务镜像可能只需 CPU 版 torch） |
| 2026-10-30 | `pyproject.toml` | 新增 mypy override：`arag` 包允许 follow untyped imports | 上游没有 `py.typed`，mypy 默认不分析它，导致继承上游类的代码无法做类型检查 | 给上游加 `py.typed`：属于改源码目录，且上游本身类型注解不完整，加了反而报大量错误 |
| 2026-10-30 | `.gitignore` | `data/` 改为 `data/*` 并加 `!data/subsets/` | 固定评测子集需要进版本库以保证可复现；原规则忽略整个 `data/` 目录，git 无法对被忽略目录下的文件取消忽略 | 把子集放到 `data/` 之外：与方案约定的目录结构不一致 |
