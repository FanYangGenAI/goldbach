# IGOT 实施方案

> **文档定位**：基于 `interpretable_goldbach_offset_transformer.md` 与 Phase 0 讨论整理的可执行开发计划。  
> **目标**：在现有数据生成管线之上，分阶段交付 IGOT（Interpretable Goldbach Offset Transformer）训练与可解释性分析能力。  
> **原则**：先验证任务可解性（coverage），再建 baseline，再叠加 Transformer，最后做机制逆向工程。

---

## 1. 项目目标回顾

本项目**不是**用神经网络证明哥德巴赫猜想，而是构建一个**可解释的数论探针**：

```text
Input:  偶数 N，候选小素数集合 {p₁, p₂, ..., pₖ}
Output: 每个候选 pᵢ 的 logit — score(N - pᵢ is prime)
Verifier: 确定性 64-bit 素性测试保证最终输出的数学正确性
```

模型只负责**候选排序**；数学合法性由 verifier 闭环保证。

最终 logit 必须可分解，以支持机制分析：

```text
logitᵢ = sieve_logitᵢ + attention_logitᵢ + residual_logitᵢ
```

---

## 2. 现状与差距

### 2.1 已有资产

| 模块 | 路径 | 状态 |
|------|------|------|
| 数据生成（Python） | `generator/python/` | 可用，支持 phase0/phase1/phase2 |
| 数据生成（CUDA） | `generator/cuda/` | 代码就绪，Windows 上需 WSL2 或升级 CUDA |
| 确定性素性测试 | `generator/python/miller_rabin.py` | 可用 |
| 26 字节定长格式 | `generator/python/format_io.py` | v1，含 `p1_label` |
| 数据集验证工具 | `tools/verify_dataset.py` | 针对 v1 格式 |
| 方案设计 | `docs/interpretable_goldbach_offset_transformer.md` | 已完成 |

### 2.2 缺失项

| 模块 | 说明 |
|------|------|
| `valid_mask[K]` 标签 | v1 仅保存单一 `p1_label`，不满足 multi-label 训练 |
| Coverage 统计 | 尚未验证「前 K 个候选」对目标 N 分布的覆盖率 |
| PyTorch 训练代码 | 无 |
| IGOT 模型实现 | 无 |
| 可解释性分析工具 | 无 |

### 2.3 关键设计变更（v1 → v2）

| 维度 | v1（现有） | v2（IGOT 需要） |
|------|-----------|----------------|
| 标签 | `p1_label`（单一最小素数） | `valid_mask[K]`（multi-hot） |
| 损失函数 | CrossEntropy（单类） | `BCEWithLogitsLoss`（多标签） |
| 样本大小 | 26 字节 | 85 字节（K=500 时） |
| 推理方式 | 直接预测 p₁ | 对 K 个候选排序 + verifier 逐个验证 |

---

## 3. 开发阶段总览

```text
Phase 0  数据与 coverage 验证     ← 当前阶段，无神经网络
Phase 1  Explicit Sieve Baseline  ← 第一次训练
Phase 2  Sieve + Residual MLP
Phase 3  完整 IGOT Transformer
Phase 4  Mechanistic Interpretability
```

每个 Phase 有明确的**入口条件**和**完成标准**，前一阶段未达标不进入下一阶段。

---

## 4. Phase 0：数据生成与 Coverage 验证

### 4.1 目的

Phase 0 **不训练任何模型**。它回答一个前置问题：

> **对于目标 N 分布，仅使用前 K 个小奇素数作为候选，Ranking 任务对绝大多数 N 是否有解？**

若 coverage 不足，后续所有 `topK_valid_rate`、`avg_checks_to_success` 指标都会失真——模型再强也无法对「词表内无解」的 N 给出正确答案。

### 4.2 核心概念

#### valid_mask

对每个偶数 N，对前 K 个奇素数候选逐一判断：

```text
valid_mask[i] = 1  if is_prime(N - pᵢ) else 0
```

与 v1 `p1_label` 的区别：

- `p1_label` 只记录**全局最小**合法 p
- `valid_mask` 记录**所有**合法候选（通常有多个 1）
- 不要求 pᵢ < N - pᵢ；只要 N - pᵢ 为素数即标记为 1

#### coverage@K

```text
coverage@K = |{ N : ∃ pᵢ in first K primes, is_prime(N - pᵢ) }| / |{ all N }|
```

Coverage 关心的是词表中**是否存在任意合法 p**，而非「最小 p 是否在词表内」。

> 注：现有 `find_min_p1` 搜索范围是 `MAX_P1 = 5000`，最小 p 可能排在第 500 名之后；但只要更大的候选 p 仍合法，coverage@500 仍可能为真。

### 4.3 工作内容

#### 4.3.1 实现 valid_mask 生成

在 `generator/python/` 新增标签函数：

```python
def compute_valid_mask(even_number: int, candidate_primes: Sequence[int]) -> list[int]:
    """Return multi-hot mask: 1 if N - p is prime, else 0."""
```

候选词表：前 K 个奇素数，初始 K=500（3, 5, 7, 11, ..., p₅₀₀）。

对每个 N 需做 K 次 `N - p` 的确定性素性测试。K=500 时计算量可接受，可批量向量化或 GPU 加速。

#### 4.3.2 定义 v2 数据格式

**文件头**：沿用 v1 的 64 字节头，将 `version` 升级为 2。

**单条样本（K=500 时 85 字节）**：

| Offset | 字段 | 类型 | 说明 |
|--------|------|------|------|
| 0–7 | `even_number` | `uint64_t` | 偶数 N |
| 8–17 | `modulo_features` | `uint8_t[10]` | N mod {3,5,7,...,31}，与 v1 兼容 |
| 18–21 | `prime_density` | `float` | 1 / ln(N) |
| 22–84 | `valid_mask` | `uint8_t[63]` | 500-bit bitset，LSB = candidate[0] |

**不单独存储**：

- `sieve_matrix`（K×M）：训练时动态计算，`z[i,j] = 1 if (N-pᵢ) mod rⱼ == 0`
- 64-bit 位流：由 `even_number` 现场展开

#### 4.3.3 Coverage 统计脚本

新增 `tools/coverage_report.py`，输入样本集（或随机采样 N），输出：

| 统计项 | 含义 |
|--------|------|
| `coverage@K` 全局 | 至少有一个合法候选的 N 占比 |
| 按 bucket 的 coverage | 各数量级（10² ~ 4×10¹⁸）分别统计 |
| `valid_count` 分布 | 每个 N 平均有几个合法候选 |
| `min_valid_rank` | 最小合法 p 在 K 候选中的排名分布 |
| `uncovered_samples` | 前 K 候选全失败的 N 样本（供 case study） |

建议先用 10 万条跨数量级样本（复用 `PHASE0_BUCKET_TARGETS` 分桶策略）跑统计，再决定是否扩到 100 万条。

#### 4.3.4 确定最终 K

```text
若某 bucket 的 coverage@500 < 阈值（建议 99%）
  → 尝试 K = 1000, 2000, 5000
  → 记录 coverage vs K 的曲线
  → 选定「coverage 足够高且存储/计算可接受」的 K
```

K 越大，样本越大、模型输出维度越高，但任务更可解。Phase 0 的核心产出之一就是**用数据支撑的 K 决策**。

**实验记录（2026-06-10，已完成）**

命令：

```bash
python tools/coverage_report.py --k 500 --seed 7 --workers 8 \
  --output data/coverage_report_phase0.json
```

样本：Phase 0 分桶随机 100,000 条偶数；候选：前 500 个奇素数（3, 5, 7, ...）。

| 指标 | 结果 |
|------|------|
| 全局 `coverage@500` | **99.999%**（99,999 / 100,000） |
| 未覆盖样本 | **仅 N=4**（bucket 1） |
| bucket 1 coverage | 97.92%（48 条中 1 条为 N=4） |
| bucket 2–18 coverage | **100%** |
| **排除 N<6 后** | **coverage@500 = 100%**（99,999 条，见 `phase1_sieve_baseline_experiment.md`） |
| 平均合法候选数 | 48.1 |
| 最小合法候选平均 rank | 12.4 |
| bucket 18 平均合法候选数 | 23.8（min=5, max=69） |

**结论：采用 K=500，无需扩大词表。**

未覆盖原因：N=4 的哥德巴赫分解为 2+2，需要候选 p=2，而 IGOT 词表仅含**奇素数**（从 3 起）。这是任务定义内的已知边界，不是词表太小。

**训练域定稿**：`MIN_TRAINING_EVEN = 6`，生成/加载/评估均排除 N=4。v2 phase0 总量 99,999（bucket 1 少 1 条）。详见 `docs/phase1_sieve_baseline_experiment.md`。

完整报告：`data/coverage_report_phase0.json`（耗时约 28s，8 workers）。

#### 4.3.5 数据正确性验证

扩展 `tools/verify_dataset.py` 支持 v2：

- 随机抽检 `valid_mask` 每一位与确定性素性测试一致
- 一致性检查：若 `p1_label`（v1）对应 p 在前 K 候选中，则 `valid_mask` 对应位必为 1
- bitset 编解码 round-trip 测试

#### 4.3.6 生成 Phase 0 数据集

产出：

| 文件 | 说明 |
|------|------|
| `data/goldbach_train_v2_phase0.bin` | 10 万条 v2 样本（含 valid_mask） |
| `data/goldbach_manifest_v2_phase0.json` | 元数据：K、coverage 摘要、生成参数 |
| `data/coverage_report_phase0.json` | 分 bucket coverage 详细统计 |

CUDA 生成器（`generator/cuda/`）在 Python 路径验证通过后，再移植 v2 格式以加速全量生成。

### 4.4 Phase 0 完成标准

- [x] 已确定 K 值，有 coverage 数据支撑（**K=500**；唯一例外 N=4，见 4.3.4）
- [x] v2 数据格式规范写入本文档并落地到 `format_io.py`
- [x] 10 万条 v2 样本生成完成，抽检通过（`goldbach_train_v2_phase0.bin`，8.11 MiB，37s）
- [x] Coverage 报告产出（`data/coverage_report_phase0.json`）
- [x] PyTorch `Dataset` 原型能正确加载 `valid_mask`（`train/dataset.py`）

### 4.5 Phase 0 不做的事

- 不训练神经网络
- 不实现 IGOT 完整架构
- 不追求 1 亿条全量数据（小规模验证优先）

---

## 5. Phase 1：Explicit Sieve Baseline

### 5.1 目的

建立**可解释的低阶基线**，回答：「显式筛法能解释多少性能？」

### 5.2 模型

```text
z[i, j] = 1  if (N - pᵢ) mod rⱼ == 0  else 0
sieve_logit[i] = bᵢ + Σⱼ wⱼ · z[i, j]
```

无可学习隐藏层；`bᵢ` 为候选先验偏置，`wⱼ` 为模数 rⱼ 的筛法惩罚权重。训练后预期 `w₃, w₅, w₇, ...` 为明显负数。

### 5.3 训练配置

```python
loss = BCEWithLogitsLoss(sieve_logits, valid_mask.float())
optimizer = Adam(params=[b, w], lr=...)
```

### 5.4 评估指标

| 指标 | 含义 |
|------|------|
| `top1_valid_rate` | 排序第一候选通过 verifier 的比例 |
| `top5_valid_rate` | top-5 内存在合法候选的比例 |
| `topK_valid_rate` | top-K 内存在合法候选的比例 |
| `avg_checks_to_success` | 平均 verifier 检查次数 |
| `BCE_loss` | multi-label 损失 |
| `generalization_by_scale` | 按 N 数量级分桶的泛化 |

### 5.5 交付物

| 路径 | 说明 |
|------|------|
| `model/sieve_baseline.py` | ExplicitSieveModule |
| `train/train.py` | 训练循环 |
| `train/metrics.py` | 评估指标 |
| `train/dataset.py` | v2 DataLoader |
| `results/phase1/` | 训练日志、checkpoint、评估报告 |

### 5.6 完成标准

- [x] Sieve baseline 训练收敛（`train/train.py`，3 epoch smoke 已跑通）
- [x] `wⱼ` 权重符合数论预期（r=3,5,7,... 均为明显负数）
- [x] 基线指标写入报告（`results/phase1/train_report.json`），作为后续 `residual_gain` 的参照

**Phase 1 smoke 结果（v2 phase0，val 10k，3 epochs，CPU）**

| 指标 | epoch 3 |
|------|---------|
| val BCE | 0.191 |
| top1_valid_rate | 42.5% |
| top5_valid_rate | 87.0% |
| topK_valid_rate | 99.99% |
| avg_checks_to_success | 2.97 |

训练命令：

```bash
cd train
python train.py --epochs 5 --batch-size 512
```

---

## 6. Phase 2：Sieve + Residual Network

### 6.1 目的

验证显式筛法之外是否存在可学习的 residual signal。

### 6.2 模型

```text
final_logit = sieve_logit + residual_logit
```

新增 `NumberEncoder`（N 的 bits、mods、log N、density、Fourier 特征）和 per-candidate 的 `residual_head`。

### 6.3 关键对比

```text
residual_gain = metrics(Phase 2) - metrics(Phase 1)
```

若 `residual_gain ≈ 0`，说明低阶筛法已覆盖绝大部分可学习结构，Transformer 价值有限。

### 6.4 完成标准

- [ ] Residual 相对 sieve-only 有统计显著的 `topK_valid_rate` 或 `avg_checks_to_success` 提升
- [ ] 记录 `residual_gain` 按 bucket 的分布

---

## 7. Phase 3：完整 IGOT Transformer

### 7.1 架构模块

按 `interpretable_goldbach_offset_transformer.md` 第 4–11 节实现：

```text
1. Number Encoder
2. Explicit Sieve Module          （复用 Phase 1）
3. Prime Token Builder
4. Modulus Token Builder
5. Prime-Modulus Cross-Attention
6. Prime-Prime Self-Attention
7. Residual Insight Head

final_logit = sieve_logit + attention_logit + residual_logit
```

### 7.2 实现注意点

- **Number Encoder**：Conv1D 沿 64-bit 序列卷积，不要先把 64 bit dense 到 256 维再 reshape 成长度 1 的序列
- **Modulo Encoder**：`N mod rⱼ` 用离散 embedding，不用连续值
- **Fourier features**：显式加入 `sin/cos(2π · (N mod rⱼ) / rⱼ)`
- **Modulus bases**：建议 M=32 个小素数 {3, 5, 7, ..., r₃₂}

### 7.3 推理流程

```python
logits = model(N, candidate_primes, modulus_bases)
ranked_indices = torch.argsort(logits, descending=True)

for idx in ranked_indices[:top_k]:
    p = candidate_primes[idx]
    q = N - p
    if is_prime_64bit(q):
        return p, q

return None  # 词表内无解
```

### 7.4 完成标准

- [ ] 三部分 logit 可独立输出并记录
- [ ] 相对 Phase 2 有 `attention_gain`（attention 模块的增量贡献）
- [ ] 模型支持 module-level ablation（移除 sieve / modulus tokens / 单个 head）

---

## 8. Phase 4：Mechanistic Interpretability

### 8.1 实验清单

| 实验 | 目的 |
|------|------|
| Modulus ablation | 逐个移除 M₃, M₅, M₇...，观察 mod-specific 错误变化 |
| Head-modulus alignment | 分析 attention head 与 modulus token 的对应关系 |
| Head ablation | 逐个 zero-out head，验证因果性（非仅看 attention map） |
| Candidate clustering | Prime-Prime attention 聚类是否对齐 mod 30 / mod 210 |
| Logit attribution | 分解 sieve / attention / residual 三方贡献方差 |
| Fourier spectrum | 固定 pᵢ，将 logit 看作 N 的函数做频谱分析 |

### 8.2 交付物

| 路径 | 说明 |
|------|------|
| `analysis/ablation.py` | 模块/head 消融 |
| `analysis/attention_viz.py` | Head-modulus 热力图 |
| `analysis/logit_decompose.py` | 三方 logit 贡献统计 |
| `analysis/fourier.py` | 周期结构分析 |
| `results/phase4/` | 实验报告 |

### 8.3 完成标准

- [ ] 至少一个 attention head 经 ablation 验证具有明确 mod-r 功能
- [ ] Logit 分解报告：成功预测主要来自哪个模块
- [ ] 形成可发表的 mechanism 发现或「无高阶结构」的负面结论

---

## 9. 推荐目录结构

```text
goldbach/
├── docs/
│   ├── interpretable_goldbach_offset_transformer.md   # 架构设计（已有）
│   └── igot_implementation_plan.md                    # 本文件
├── generator/
│   ├── python/                                        # 扩展 v2 标签生成
│   └── cuda/                                          # 后续移植 v2
├── model/
│   ├── sieve_baseline.py
│   ├── number_encoder.py
│   ├── igot.py
│   └── tokens.py
├── train/
│   ├── dataset.py
│   ├── train.py
│   └── metrics.py
├── analysis/
│   ├── ablation.py
│   ├── attention_viz.py
│   ├── logit_decompose.py
│   └── fourier.py
├── tools/
│   ├── verify_dataset.py                              # 扩展 v2 支持
│   ├── coverage_report.py                             # 新增
│   └── read_sample.py                                 # 扩展 v2 支持
└── data/
    ├── goldbach_train_v2_phase0.bin
    ├── goldbach_manifest_v2_phase0.json
    └── coverage_report_phase0.json
```

---

## 10. 数据生成 Phase 命名对照

仓库中存在两套「Phase」命名，避免混淆：

| 名称 | 含义 | 样本量 |
|------|------|--------|
| **IGOT Phase 0** | valid_mask + coverage 验证 | 10 万（建议） |
| **IGOT Phase 1–4** | 模型训练与可解释性 | — |
| **Generator phase0** | Python 生成器的小规模验证集 | 10 万（`PHASE0_TOTAL`） |
| **Generator phase1** | 桶 1–6 全量枚举 | ~500 万 |
| **Generator phase2** | 全量 1 亿条生产数据 | 1 亿 |

IGOT Phase 0 可复用 Generator phase0 的分桶策略，但需将标签格式从 `p1_label` 升级为 `valid_mask`。

---

## 11. 风险与决策点

| 风险 | 影响 | 应对 |
|------|------|------|
| coverage@500 在大 N 不足 | 模型上限被词表限制 | 增大 K；或接受部分 N 不可解并单独报告 |
| v2 样本体积增大（26B → 85B） | 1 亿条约 8.5 GB | 可接受；或在线计算 valid_mask（生成慢） |
| Residual gain 接近 0 | Transformer 投入可能不值 | Phase 2 结果决定是否继续 Phase 3 |
| Attention 不可解释 | 违背项目核心目标 | 记录负面结果；简化架构而非追求性能 |
| CUDA 生成器 Windows 编译受阻 | 全量生成慢 | WSL2 构建；或 Python 生成器先跑通 |

---

## 12. 近期行动项（Phase 0 拆解）

按优先级排序：

1. [x] **`generator/python/labeling.py`**：新增 `compute_valid_mask(even_number, candidate_primes)`
2. [x] **`generator/python/format_io.py`**：定义 v2 `GoldbachSampleV2` 与 bitset 编解码
3. [x] **`tools/coverage_report.py`**：对随机/现有 N 样本统计 coverage@K
4. [x] **跑 coverage 实验**：K=500，按 18 个 bucket 出报告 → **K=500 确认**
5. [x] **根据 coverage 结果确定 K**，更新本文档第 4.3.4 节
6. [x] **生成 `goldbach_train_v2_phase0.bin`**（10 万条）
7. [x] **`tools/verify_dataset.py`**：扩展 v2 抽检逻辑
8. [x] **`train/dataset.py`**：PyTorch Dataset 原型

预计 Phase 0 完成后，即可启动 Phase 1 的 sieve baseline 训练。

---

## 13. 参考文档

- 架构设计：`docs/interpretable_goldbach_offset_transformer.md`
- 数据生成（v1）：`docs/训练数据集生成最终方案.md`
- 本实施方案：`docs/igot_implementation_plan.md`
