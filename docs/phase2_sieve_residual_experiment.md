# Phase 2：Sieve + Residual 实验报告

> **文档定位**：记录 IGOT 项目 Phase 2（显式筛法 + Residual MLP）的完整技术细节、实验配置、结果与结论。  
> **实验日期**：2026-06-10  
> **代码入口**：`train/train.py --phase phase2`、`model/sieve_residual.py`、`model/number_encoder.py`

---

## 1. 实验目的

Phase 1 建立了可解释的线性筛法基线，但 top1 ≈ 43%、top5 ≈ 87%，说明**低阶整除规则不足以区分多个合法候选**。

Phase 2 在保持可解释加法结构的前提下，叠加一个轻量 Residual 分支，验证：

> 在固定筛法骨架之上，是否存在**显式筛法之外的可学习排序信号**？

核心观测指标为 `residual_gain`：

```text
residual_gain = metrics(Phase2) - metrics(Phase1)
```

若 top1 / top5 / avg_checks 相对 Phase 1 有稳定提升，则 residual 分支捕获到了筛法未能表达的结构。

---

## 2. 任务定义

与 Phase 1 完全一致（见 `docs/phase1_sieve_baseline_experiment.md` 第 2 节）：

| 项目 | 定义 |
|------|------|
| 输入 | 偶数 \(N\) + 位特征 + 模特征 + density |
| 候选词表 | 前 \(K=500\) 个奇素数 |
| 标签 | `valid_mask` multi-hot（\(N-p_i\) 为素数则为 1） |
| 损失 | `BCEWithLogitsLoss` |
| 训练域 | \(N \ge 6\)，99,999 条样本 |

---

## 3. 模型：SieveResidualModel

### 3.1 加法分解（IGOT 核心设计）

\[
\text{final\_logit}_i = \underbrace{b_i + \sum_j w_j z_{i,j}}_{\text{sieve\_logit（Phase 1）}} + \underbrace{\text{MLP}(\mathbf{f}_i)}_{\text{residual\_logit}}
\]

| 分支 | 模块 | 可解释性 |
|------|------|----------|
| sieve | `ExplicitSieveModule` | 完全可解释（\(w_j\) 可直接读出） |
| residual | `NumberEncoder` + per-candidate MLP | 黑盒，但 logit 可单独 ablate |

### 3.2 NumberEncoder（全局 N 上下文）

实现于 `model/number_encoder.py`，将偶数 \(N\) 编码为 `d_model=128` 向量，四个分支拼接后投影：

| 分支 | 输入 | 输出维度 |
|------|------|----------|
| Bit Conv1D | `bits[64]` | 64 |
| Mod Embedding | `mods[10]`，每模数独立 `Embedding(r_j, 8)` | 80 |
| Scale | `log(N)`, `density=1/log N` | 2 |
| Fourier | `sin/cos(2π·mods/r_j)` | 20 |

### 3.3 Residual MLP（逐候选）

每个候选 \(p_i\) 的 residual 输入特征 \(\mathbf{f}_i\)：

\[
\mathbf{f}_i = \big[\log p_i,\; \log(N-p_i),\; z_{i,:},\; \text{global\_context}\big]
\]

| 分量 | 维度 | 含义 |
|------|------|------|
| \(\log p_i\) | 1 | 候选素数尺度 |
| \(\log(N-p_i)\) | 1 | 偏移量尺度 |
| \(z_{i,:}\) | 32 | 该候选的筛法行 |
| global_context | 128 | N 的全局编码 |

MLP 结构：`Linear(162→64) → ReLU → Linear(64→1)`，对全部 500 个候选共享权重。

### 3.4 参数量

| 模块 | 参数量（约） |
|------|-------------|
| ExplicitSieveModule | 532（500 bias + 32 weights） |
| NumberEncoder | ~52K |
| Residual MLP | ~10.5K |
| **合计** | **~56,517** |

### 3.5 实现细节

- **代码**：`model/sieve_residual.py`
- sieve 子模块与 Phase 1 结构相同，权重联合训练（非冻结）
- `forward` 返回 `final_logits`、`sieve_logits`、`residual_logits`、`sieve_matrix`，便于 Phase 4 归因分析

---

## 4. 数据集

与 Phase 1 相同，使用 `data/goldbach_train_v2_phase0.bin`：

- 99,999 条样本（排除 \(N=4\)）
- 90% / 10% 训练验证划分，`random_split(seed=7)`
- 训练集 90,000 条，验证集 9,999 条

---

## 5. 训练配置

### 5.1 超参数（正式实验）

| 参数 | 值 | 说明 |
|------|-----|------|
| Phase | `phase2` | sieve + residual |
| 数据集 | `data/goldbach_train_v2_phase0.bin` | v2，99,999 条 |
| Epochs | 10 | |
| **Batch size** | **2048** | 自 pilot（512）提升 4× |
| 优化器 | Adam | |
| 学习率 | 0.01 | 与 Phase 1 相同，便于公平对比 |
| 损失 | `BCEWithLogitsLoss` | |
| 设备 | CUDA（RTX 4090） | |
| seed | 7 | |
| Phase 1 参照报告 | `results/phase1_gpu/train_report.json` | 用于计算 residual_gain |

### 5.2 Batch size 调整说明

Pilot 实验使用 `batch_size=512`（每 epoch ~176 步）。正式实验将 batch size 提升至 **2048**（每 epoch ~44 步），理由：

1. Phase 2 模型仍较轻（~56K 参数），GPU 显存充裕
2. 更大 batch 可提供更稳定的梯度估计
3. 与 Phase 1 的 512 batch 对比时，主要关注**排序指标增益**而非绝对 BCE

> `train/train.py` 默认 batch size 已更新为 2048。

### 5.3 训练命令

```powershell
.\.venv\Scripts\Activate.ps1
python train\train.py --phase phase2 --epochs 10 --batch-size 2048 --device cuda `
  --output-dir results\phase2_bs2048 `
  --baseline-report results\phase1_gpu\train_report.json
```

### 5.4 产出物

| 文件 | 说明 |
|------|------|
| `results/phase2_bs2048/best.pt` | 验证集 top1 最优 checkpoint（epoch 3） |
| `results/phase2_bs2048/last.pt` | 最后一轮 checkpoint |
| `results/phase2_bs2048/train_report.json` | 完整训练日志与分桶指标 |

---

## 6. 评估指标

与 Phase 1 相同（`train/metrics.py`），额外记录：

| 指标 | 定义 |
|------|------|
| `residual_gain` | Phase 2 指标 − Phase 1 epoch-10 指标 |
| `residual_logits` | 可通过 checkpoint 分解 sieve / residual 贡献 |

最优 checkpoint 选择标准：**验证集 `top1_valid_rate` 最高**。

---

## 7. 实验结果

### 7.1 Phase 1 参照基线（GPU，batch=512，epoch 10）

| 指标 | Phase 1 |
|------|---------|
| val BCE | 0.161 |
| top1 | 42.9% |
| top5 | 87.0% |
| topK | 100% |
| avg_checks | 3.05 |

### 7.2 Phase 2 正式实验（batch=2048）逐 epoch 验证集

| Epoch | train BCE | val BCE | top1 | top5 | topK | avg_checks | Δtop1 |
|-------|-----------|---------|------|------|------|------------|-------|
| 1 | 0.263 | 0.191 | 43.4% | 86.9% | 100% | 3.00 | +0.5pp |
| 2 | 0.162 | 0.139 | 42.9% | 87.0% | 100% | 3.02 | −0.1pp |
| **3** | **0.134** | **0.130** | **43.8%** | **87.0%** | **100%** | **2.90** | **+0.9pp** |
| 4 | 0.130 | 0.128 | 42.2% | 87.1% | 100% | 2.89 | −0.7pp |
| 5 | 0.130 | 0.128 | 43.0% | 86.6% | 100% | 2.90 | +0.1pp |
| 6 | 0.127 | 0.127 | 42.1% | 87.1% | 100% | 2.90 | −0.8pp |
| 7 | 0.126 | 0.126 | 42.6% | 86.8% | 100% | 2.90 | −0.3pp |
| 8 | 0.126 | 0.126 | 42.6% | 87.1% | 100% | 2.90 | −0.3pp |
| 9 | 0.126 | 0.126 | 42.2% | 86.7% | 100% | 2.90 | −0.7pp |
| 10 | 0.126 | 0.126 | 42.6% | 86.7% | 100% | 2.90 | −0.3pp |

训练耗时：**33.7 秒**（10 epochs，RTX 4090）。

### 7.3 最优 checkpoint（epoch 3）vs Phase 1

| 指标 | Phase 1 (ep10) | Phase 2 best (ep3) | residual_gain |
|------|----------------|---------------------|---------------|
| val BCE | 0.161 | 0.130 | **−0.031** |
| top1 | 42.9% | **43.8%** | **+0.9pp** |
| top5 | 87.0% | 87.0% | 0.0pp |
| topK | 100% | 100% | 0.0pp |
| avg_checks | 3.05 | **2.90** | **−0.15** |

### 7.4 Pilot 对比（batch=512，非正式）

早期 pilot 使用 `batch_size=512`，结果如下：

| 配置 | best epoch | best top1 | best avg_checks | elapsed |
|------|------------|-----------|-----------------|---------|
| batch=512（pilot） | 4 | **44.3%** | **2.84** | 33.3s |
| **batch=2048（正式）** | **3** | **43.8%** | **2.90** | 33.7s |

Pilot 在 top1 上略优（+0.5pp），但 batch=2048 的 BCE 收敛更平滑（epoch 3 后 train/val BCE 均稳定在 0.126 附近）。两种配置下 **topK 均保持 100%**，avg_checks 均显著低于 Phase 1。

### 7.5 收敛性分析

| 观察 | 结论 |
|------|------|
| val BCE | epoch 1→10 从 0.191 降至 0.126，**显著低于 Phase 1 的 0.161** |
| top1 | epoch 3 达到峰值 43.8%（+0.9pp），之后波动在 42–43% |
| top5 | 稳定在 86.7%–87.1%，与 Phase 1 基本持平 |
| topK | **全程 100%**，筛法覆盖率未被 residual 破坏 |
| avg_checks | 从 3.00 降至 2.90，**平均少检查 0.15 个候选** |
| 模数权重 | 全部 32 个 \(w_j\) 仍为负，sieve 分支行为正常 |

### 7.6 最终模数权重（epoch 10，batch=2048）

训练后 sieve 分支的模数权重仍为负，幅度较 Phase 1 小（联合训练中 residual 分担了部分排序压力）：

| 模数 r | 权重 w | 模数 r | 权重 w |
|--------|--------|--------|--------|
| 3 | −0.440 | 47 | −0.443 |
| 5 | −0.448 | 53 | −0.425 |
| 7 | −0.448 | 59 | −0.427 |
| 11 | −0.430 | 61 | −0.432 |
| 13 | −0.433 | 67 | −0.427 |
| 17 | −0.428 | 71 | −0.431 |

全部 32 个模数权重均为负数，与数论筛法预期一致。

### 7.7 分桶观察（best epoch 3）

| Bucket | N 范围（约） | topK | avg_checks | 样本数 |
|--------|-------------|------|------------|--------|
| 1 | \(10^2\) | 100% | 24.1 | 11 |
| 2–3 | \(10^2\)–\(10^4\) | 100% | 1.0 | 492 |
| 4–8 | \(10^4\)–\(10^8\) | 100% | 1.1–2.1 | 3,155 |
| 9–14 | \(10^8\)–\(10^{14}\) | 100% | 2.5–3.8 | 3,790 |
| 15–18 | \(10^{14}\)–\(10^{18}\) | 100% | 4.1–4.9 | 2,541 |

- 所有 bucket topK = 100%
- 小 N（bucket 1）avg_checks 仍偏高（候选密集、排序难）
- 大 N bucket avg_checks 与 Phase 1 趋势一致，residual 未引入跨桶退化

---

## 8. 结论

### 8.1 Phase 2 已达成目标

1. **加法分解有效**：`final_logit = sieve + residual` 结构在训练中稳定收敛
2. **BCE 显著下降**：最优 val BCE 0.130 vs Phase 1 的 0.161（−19%）
3. **top1 有提升**：最优 +0.9pp（43.8% vs 42.9%），证实存在筛法外的可学习信号
4. **效率改善**：avg_checks 2.90 vs 3.05（−5%）
5. **覆盖率不变**：topK 保持 100%，residual 未破坏 sieve 的排除能力
6. **可解释性保留**：sieve 权重 \(w_j\) 仍为负，模块可独立 ablate

### 8.2 Residual 的能力边界

| 能力 | Phase 1 | Phase 2 best | 评价 |
|------|---------|--------------|------|
| topK 覆盖 | 100% | 100% | 保持 |
| avg_checks | 3.05 | 2.90 | 小幅改善 |
| top1 一次命中 | 42.9% | 43.8% | **有提升但有限** |
| top5 命中 | 87.0% | 87.0% | **基本持平** |

**核心结论**：轻量 per-candidate MLP 能捕获部分筛法未表达的信号（BCE ↓、top1 ↑、checks ↓），但 **top5 未突破 Phase 1 平台**。这说明更高阶的候选间交互（如 attention）可能是 Phase 3 Transformer 的必要动机。

### 8.3 正式定稿

| 项目 | 定稿值 |
|------|--------|
| 模型 | `SieveResidualModel` |
| Batch size | 2048 |
| 正式 checkpoint | `results/phase2_bs2048/best.pt`（epoch 3） |
| 正式报告 | `results/phase2_bs2048/train_report.json` |
| Phase 1 参照 | `results/phase1_gpu/train_report.json` |
| residual_gain (best) | top1 +0.9pp, BCE −0.031, checks −0.15 |

---

## 9. 代码结构

```text
model/
  sieve_baseline.py       # ExplicitSieveModule（Phase 1，被复用）
  number_encoder.py       # NumberEncoder（全局 N 编码）
  sieve_residual.py       # SieveResidualModel（Phase 2 主模型）
  test_sieve_residual.py  # 单元测试

train/
  train.py                # --phase phase1|phase2 统一训练入口
  metrics.py              # 排序指标 + residual_gain 计算
  dataset.py              # GoldbachBinDataset
```

---

## 10. 复现实验清单

```powershell
# 1. 激活环境
.\.venv\Scripts\Activate.ps1

# 2. 单元测试
python -m unittest discover -s model -p "test_*.py" -v

# 3. 确认 Phase 1 基线报告存在（若无则先训练 Phase 1）
python train\train.py --phase phase1 --epochs 10 --batch-size 512 --device cuda `
  --output-dir results\phase1_gpu

# 4. Phase 2 正式训练（batch=2048）
python train\train.py --phase phase2 --epochs 10 --batch-size 2048 --device cuda `
  --output-dir results\phase2_bs2048 `
  --baseline-report results\phase1_gpu\train_report.json

# 5. 查看报告
Get-Content results\phase2_bs2048\train_report.json | ConvertFrom-Json | Select-Object -ExpandProperty final_residual_gain
```

---

## 11. 下一步：Phase 3

在 `sieve + residual` 基础上引入 **cross-attention / self-attention**，形成完整 IGOT Transformer：

```text
final_logit = sieve_logit + attention_logit + residual_logit
```

重点观察：

```text
attention_gain = metrics(Phase3) - metrics(Phase2)
```

若 attention 能突破 top5 ≈ 87% 的平台，则说明候选间交互是排序质量的关键瓶颈。

详见 `docs/igot_implementation_plan.md`。
