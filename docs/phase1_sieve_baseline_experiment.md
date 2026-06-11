# Phase 1：Explicit Sieve Baseline 实验报告

> **文档定位**：记录 IGOT 项目 Phase 1（显式筛法基线）的完整技术细节、实验配置、结果与结论。  
> **实验日期**：2026-06-10  
> **代码入口**：`train/train.py`、`model/sieve_baseline.py`

---

## 1. 实验目的

Phase 1 的目标不是追求最高预测精度，而是建立一个**可解释、可 ablate 的低阶基线**，用于回答：

> 在控制候选小素数词表的前提下，**显式筛法规则**能解释多少排序性能？

该基线将作为后续 Phase 2（Sieve + Residual）和 Phase 3（Transformer）的参照，通过 `residual_gain` / `attention_gain` 量化更高阶结构的真实贡献。

---

## 2. 任务定义

### 2.1 输入与输出

| 项目 | 定义 |
|------|------|
| 输入 | 偶数 \(N\) |
| 候选词表 | 前 \(K=500\) 个奇素数 \(p_i \in \{3,5,7,\ldots\}\) |
| 输出 | 每个候选的 logit：\(\text{score}(N - p_i \text{ 为素数})\) |
| 标签 | `valid_mask[i] = 1` 当且仅当 \(N - p_i\) 为素数（multi-hot） |
| 损失函数 | `BCEWithLogitsLoss`（多标签，非单类 CrossEntropy） |

### 2.2 推理语义

模型只负责**候选排序**；最终数学合法性由确定性 Miller-Rabin 验证（标签生成时已使用同一套素性测试）。

评估时模拟 verifier 流程：按 logit 从高到低检查候选，统计 top-1 / top-5 / top-K 命中率与平均检查次数。

---

## 3. 模型：Explicit Sieve Module

### 3.1 筛法矩阵

对候选 \(p_i\) 和模数基底 \(r_j\)，定义：

\[
z_{i,j} = \mathbb{1}\left[(N - p_i) \bmod r_j = 0\right]
\]

若 \(z_{i,j}=1\)，表示偏移量 \(N-p_i\) 被小素数 \(r_j\) 整除，该候选在数论上更不可能为素数。

### 3.2 线性 logit

\[
\text{logit}_i = b_i + \sum_j w_j \cdot z_{i,j}
\]

| 参数 | 形状 | 含义 |
|------|------|------|
| \(b_i\) | `[500]` | 候选 \(p_i\) 的先验偏置 |
| \(w_j\) | `[32]` | 模数 \(r_j\) 的惩罚权重 |

**无可学习隐藏层**，这是整个 IGOT 架构中可解释性最强的模块。

### 3.3 实现细节

- **代码**：`model/sieve_baseline.py`
- **候选词表**：`first_k_odd_primes(500)`，注册为 buffer（不参与梯度）
- **模数基底**：前 32 个奇素数 \(\{3,5,7,\ldots\}\)，注册为 buffer
- **向量化**：`compute_sieve_matrix(N, candidates, moduli)` 输出 `[B, K, M]`，用 `einsum("bkm,m->bk")` 聚合

### 3.4 数论预期

训练收敛后，若模型学到正常筛法，期望：

\[
w_3, w_5, w_7, \ldots < 0
\]

实验结果符合该预期（见第 8 节）。

---

## 4. 数据集

### 4.1 格式（v2）

| 字段 | 类型 | 说明 |
|------|------|------|
| `even_number` | uint64 | 偶数 \(N\) |
| `modulo_features` | uint8[10] | \(N \bmod \{3,5,\ldots,31\}\) |
| `prime_density` | float32 | \(1/\ln N\) |
| `valid_mask` | 500-bit bitset | multi-hot 标签 |

- 单条样本 **85 字节**；文件头 64 字节
- 生成命令：`python generator/python/generate.py --phase phase0 --format v2`

### 4.2 分桶策略

复用 Phase 0 的 18 桶跨数量级设计（\(10^2 \sim 4\times10^{18}\)）：

- 桶 1–3：全量枚举
- 桶 4–18：对 \(\log N\) 均匀抽样

### 4.3 N=4 的特殊处理

#### 问题

\(N=4\) 的哥德巴赫分解为 \(2+2\)，需要候选 \(p=2\)。但 IGOT 词表仅含**奇素数**（从 3 起），因此对 \(N=4\)：

```text
valid_mask 全为 0  →  coverage@500 无法达到 100%
```

在原始 100,000 条数据集中，\(N=4\) 是**唯一**未覆盖样本（bucket 1 全量枚举的首个偶数）。

#### 解决方案

引入训练域下界常量：

```python
MIN_TRAINING_EVEN = 6  # generator/python/constants.py
```

| 层面 | 处理方式 |
|------|----------|
| 数据生成（v2） | 过滤 \(N < 6\)；bucket 1 少 1 条（无法在同范围内补足），总量 **99,999** |
| PyTorch Dataset | `GoldbachBinDataset(min_even=6)` 二次过滤 |
| Coverage 统计 | `coverage_report.py` 自动排除 \(N < 6\) |

排除 \(N=4\) 后，`coverage@500 = 100\%`（99,999 / 99,999）。

> **说明**：\(N=6\) 可用 \(p=3\)（\(6-3=3\) 为素数），是最小的合法训练样本。

---

## 5. 训练配置

### 5.1 超参数

| 参数 | 值 |
|------|-----|
| 数据集 | `data/goldbach_train_v2_phase0.bin`（99,999 条，排除 N=4） |
| 训练 / 验证划分 | 90% / 10%，`random_split(seed=7)` |
| Epochs | 10 |
| Batch size | 512 |
| 优化器 | Adam |
| 学习率 | 0.01 |
| 损失 | `nn.BCEWithLogitsLoss()` |
| 设备 | CPU（本机无 CUDA 训练环境） |

### 5.2 训练命令

```bash
cd train
python train.py --epochs 10 --batch-size 512 --lr 0.01 --seed 7 \
  --output-dir ../results/phase1_min6
```

### 5.3 产出物

| 文件 | 说明 |
|------|------|
| `results/phase1_min6/best.pt` | 验证集 topK 最优 checkpoint |
| `results/phase1_min6/last.pt` | 最后一轮 checkpoint |
| `results/phase1_min6/train_report.json` | 完整训练日志与分桶指标 |

---

## 6. 评估指标

实现于 `train/metrics.py`。

| 指标 | 定义 |
|------|------|
| `bce_loss` | 多标签 BCE 损失 |
| `top1_valid_rate` | 排序第一的候选 `valid_mask=1` 的比例 |
| `top5_valid_rate` | top-5 内至少有一个合法候选的比例 |
| `topk_valid_rate` | top-K（K=500）内至少有一个合法候选的比例 |
| `avg_checks_to_success` | 按 logit 降序扫描，首次命中合法候选的平均检查次数（1-based） |
| `failure_rate_under_k` | `1 - topk_valid_rate` |
| `by_bucket` | 按 18 个数量级桶分别统计上述指标 |

---

## 7. 实验结果对比

### 7.1 数据集与 Coverage

| 配置 | 样本数 | coverage@500 | 未覆盖样本 |
|------|--------|--------------|------------|
| 含 N=4（原始） | 100,000 | 99.999% | N=4 |
| **排除 N<6（正式）** | **99,999** | **100%** | 无 |

### 7.2 验证集指标（10 epochs）

#### 含 N=4（`results/phase1/`，val=10,000）

| Epoch | val BCE | top1 | top5 | topK | avg_checks |
|-------|---------|------|------|------|------------|
| 1 | 0.229 | 42.2% | 86.6% | 99.99% | 2.99 |
| 5 | 0.175 | 42.7% | 86.5% | 99.99% | 3.00 |
| 10 | 0.161 | 43.3% | 86.7% | 99.99% | 2.97 |

#### 排除 N=4（`results/phase1_min6/`，val=9,999）— **正式基线**

| Epoch | val BCE | top1 | top5 | topK | avg_checks |
|-------|---------|------|------|------|------------|
| 1 | 0.229 | 43.1% | 86.8% | **100%** | 3.07 |
| 5 | 0.175 | 43.2% | 86.8% | **100%** | 3.05 |
| 10 | **0.161** | **42.9%** | **87.0%** | **100%** | **3.05** |

### 7.3 训练曲线（Stage 0 回溯，不重训）

![Phase 1 train/val BCE](../figures/phase1_train_val_bce.png)

![Phase 1 top1/top5 vs epoch](../figures/phase1_top1_top5_vs_epoch.png)

数据来源：`results/phase1_gpu/train_report.json`（GPU，batch=512，10 epoch，固定 LR）。

### 7.4 收敛性分析

| 观察 | 结论 |
|------|------|
| val BCE | epoch 1→10 从 0.229 降至 0.161，仍在缓慢下降但收益递减 |
| topK | 排除 N=4 后**稳定 100%**；含 N=4 时恒为 99.99% |
| top1 / top5 | epoch 3 后稳定在 43% / 87% 附近，**排序上限明显** |
| avg_checks | ~3.05，即平均检查约 3 个候选即可找到合法解 |
| 模数权重 | 全部 32 个 \(w_j\) 均为负，且随 epoch 幅度增大 |

### 7.5 最终模数权重（epoch 10，排除 N=4）

| 模数 r | 权重 w | 模数 r | 权重 w |
|--------|--------|--------|--------|
| 3 | -3.814 | 47 | -3.473 |
| 5 | -3.683 | 53 | -3.465 |
| 7 | -3.629 | 59 | -3.461 |
| 11 | -3.584 | 61 | -3.456 |
| 13 | -3.572 | 67 | -3.445 |
| 17 | -3.552 | 71 | -3.440 |

**全部 32 个模数权重均为负数**，与数论筛法预期一致。

### 7.6 分桶观察（epoch 10）

- **bucket 1**（小 N）：topK=100%，但 `avg_checks` 较高（候选密集、排序更难）
- **bucket 3–6**：`avg_checks` 接近 1（小 N 上合法候选多且靠前）
- **bucket 18**（极大 N）：topK=100%，`avg_checks` ≈ 4.7（合法候选平均更少，但词表仍覆盖）

---

## 8. 结论

### 8.1 Phase 1 已达成目标

1. **可解释性**：模型学到了「被小素数整除则压低分数」的显式规则（\(w_j < 0\)）
2. **覆盖率**：排除 \(N=4\) 后，topK_valid_rate = **100%**
3. **效率**：平均约 **3 次** verifier 检查即可命中合法候选
4. **稳定性**：排序类指标在 epoch 3–5 后趋于稳定

### 8.2 显式筛法的能力边界

| 能力 | 表现 |
|------|------|
| 词表内是否存在合法候选（topK） | **已解决**（100%） |
| 快速定位合法候选（avg_checks ≈ 3） | **良好** |
| 一次命中最优候选（top1 ≈ 43%） | **不足** |
| top-5 命中 | **~87%**，仍有提升空间 |

**核心结论**：线性显式筛法足以完成「排除明显错误候选 + 粗排序」，但无法学到「哪个合法候选更优」。这正是 Phase 2 Residual 模块的动机。

### 8.3 正式基线定稿

| 项目 | 定稿值 |
|------|--------|
| 训练域 | \(N \ge 6\) |
| 样本量 | 99,999 |
| K | 500 |
| 模数基底数 M | 32 |
| 正式 checkpoint | `results/phase1_min6/best.pt` |
| 正式报告 | `results/phase1_min6/train_report.json` |

---

## 9. 代码结构

```text
model/
  sieve_baseline.py      # ExplicitSieveModule + compute_sieve_matrix
  test_sieve_baseline.py # 单元测试

train/
  dataset.py             # GoldbachBinDataset（mmap，min_even 过滤）
  metrics.py             # topK / avg_checks / 分桶指标
  train.py               # Phase 1 训练入口
  smoke_loader.py        # DataLoader 冒烟测试

generator/python/
  labeling.py            # compute_valid_mask
  format_io.py             # v2 格式
  generate.py              # v2 数据生成（含 MIN_TRAINING_EVEN 过滤）

tools/
  coverage_report.py     # coverage@K 统计
  verify_dataset.py        # 数据集审计
```

---

## 10. 复现实验清单

```bash
# 1. 生成 v2 数据集（排除 N<6）
cd generator/python
python generate.py --phase phase0 --format v2 --seed 7 --workers 8

# 2. Coverage 验证（应得 100%）
python tools/coverage_report.py --k 500 --seed 7 --workers 8 \
  --output data/coverage_report_phase0_min6.json

# 3. 数据集审计
python tools/verify_dataset.py --phase phase0 --format v2 --sample-size 10000

# 4. 训练 10 epochs
cd train
python train.py --epochs 10 --batch-size 512 --seed 7 \
  --output-dir ../results/phase1_min6

# 5. 单元测试
cd model && python -m unittest test_sieve_baseline.py -v
cd ../train && python -m unittest test_dataset.py -v
```

---

## 11. 下一步：Phase 2

在固定 Explicit Sieve 的基础上叠加 Residual 分支：

```text
final_logit = sieve_logit + residual_logit
```

重点观察：

```text
residual_gain = metrics(Phase2) - metrics(Phase1)
```

若 residual 能显著提升 top1 / top5，则说明存在显式筛法之外的可学习信号。
