# 实验记录

所有花费按百炼**标价**、人民币计（口径见 design_decisions.md D5）。实际账单可能因免费额度、折扣而更低。

## 本月花费台账（2026-10）

| 日期 | 实验 | Agent | 裁判 | 小计 | 本月累计 |
| --- | --- | ---: | ---: | ---: | ---: |
| 2026-10-30 | 环境检查与建索引（本地 GPU，无 API 调用） | 0 | 0 | 0 | 0 |

---

## 2026-10-30 · 环境检查与 MuSiQue 索引构建

- **环境**：WSL2 Ubuntu（内核 6.18），Python 3.12.3，uv 0.12.23；WSL 可用内存 7 GB，16 核。
- **GPU**：NVIDIA GeForce RTX 3060 Laptop GPU，**显存 6144 MiB**（6.0 GiB），sm_86；Windows 驱动 572.70（支持 CUDA ≤ 12.8）。
- **PyTorch**：默认安装的 torch 2.15.0+cu132 无法使用 GPU（驱动太旧），改用 torch 2.11.0+cu128 后 `torch.cuda.is_available() == True`，矩阵乘法验证通过（见 upstream_changes.md）。
- **索引构建**（上游 `scripts/build_index.py`，参数与 README 相同）：

| 项目 | 数值 |
| --- | --- |
| 嵌入模型 | Qwen/Qwen3-Embedding-0.6B（1024 维，fp32） |
| 块数 / 句子数 | 1,354 / 50,767 |
| batch size | 32（脚本默认值） |
| 耗时 | 4 分 26 秒（含模型下载） |
| 峰值显存 | 约 3.1 GB（nvidia-smi 采样） |
| 峰值内存（RSS） | 2.3 GB |
| 索引文件 | `data/musique/index/sentence_index.pkl`，211 MB |

- **结论**：6 GB 显存下 batch 32 留有约 3 GB 余量。之后语料更大时，内存（WSL 7 GB）比显存更早成为瓶颈：pickle 索引会整体加载到内存中。
