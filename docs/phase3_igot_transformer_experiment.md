# Phase 3：IGOT Transformer 实验报告

> **文档定位**：记录 IGOT 项目 Phase 3（显式筛法 + Cross/Self-Attention + Residual）的完整技术细节、实验配置、SDPA 工程优化、结果与结论。  
> **实验日期**：2026-06-10  
> **代码入口**：`train/train.py --phase phase3`、`model/igot.py`、`model/tokens.py`、`model/attention.py`

---

## 1. 实验目的

Phase 2 证实存在筛法外的可学习信号（BCE ↓、top1 小幅 ↑），但 **top5 ≈ 87% 未突破 Phase 1 平台**。Per-candidate MLP 无法建模候选间的竞争与替代关系。

Phase 3 在保持三路 logit 加法分解的前提下，引入 **Prime-Modulus Cross-Attention** 与 **Prime-Prime Self-Attention**，验证：

> Attention 能否捕获候选–模数、候选–候选的高阶交互，带来可量化的 `attention_gain`？

```text
attention_gain = metrics(Phase3) - metrics(Phase2)
final_logit_i = sieve_logit_i + attention_logit_i + residual_logit_i
```

---

## 2. 任务定义

与 Phase 1/2 完全一致（见 `docs/phase1_sieve_baseline_experiment.md` 第 2 节）：

| 项目 | 定义 |
|------|------|
| 输入 | 偶数 \(N\) + `bits[64]` + `mods[10]` + `density` |
| 候选词表 | 前 \(K=500\) 个奇素数 |
| 标签 | `valid_mask` multi-hot |
| 损失 | `BCEWithLogitsLoss` |
| 训练域 | \(N \ge 6\)，99,999 条样本 |
| 最优 checkpoint 选择 | 验证集 `top1_valid_rate` 最高 |

---

## 3. 模型：IGOTModel

### 3.1 加法分解

\[
\text{final\_logit}_i =
\underbrace{b_i + \sum_j w_j z_{i,j}}_{\text{sieve}}
+ \underbrace{\text{Linear}(\text{AttnToken}_i)}_{\text{attention}}
+ \underbrace{\text{Linear}(\text{AttnToken}_i)}_{\text{residual}}
\]

| 分支 | 模块 | 可解释性 |
|------|------|----------|
| sieve | `ExplicitSieveModule`（Phase 1 复用） | 完全可解释 |
| attention | Cross-Attn + Self-Attn + `attention_head` | 可 ablate、可导出权重 |
| residual | 同上 token + `residual_head` | 可 ablate |

### 3.2 Token 构建

| 模块 | 输出 | 语义 |
|------|------|------|
| `PrimeTokenBuilder` | `[B, K, d]` | 每个候选 \(p_i\) 的上下文 |
| `ModulusTokenBuilder` | `[B, M, d]` | 每个筛法模数 \(r_j\) 的维度 |

Prime token 特征：`prime_id_emb`、\(\log p_i\)、\(\log(N-p_i)\)、\(p_i \bmod r_j\)、\(N \bmod r_j\)、\((N-p_i)\bmod r_j\)、筛法行 \(z_{i,:}\)、全局 N 编码。

Modulus token 特征：模数 id、\(N \bmod r_j\) embedding、淘汰统计、Fourier、全局 N 编码。

### 3.3 注意力栈（浅层设计）

| 层 | 类型 | 层数 | 说明 |
|----|------|------|------|
| Cross-Attention | Q=Prime, K/V=Modulus | **1** | 候选关注模数约束 |
| Self-Attention | Q=K=V=Prime | **1** | 候选间交互，**O(K²)** |

每层结构：`MultiHeadSDPA(4 heads)` + LayerNorm + FFN + LayerNorm。

> **注意**：当前实现为 **1×cross + 1×self**，非深层 Transformer 堆叠。

### 3.4 SDPA / FlashAttention 工程优化

`model/attention.py` 中 `MultiHeadSDPA`：

- **训练默认路径**：`F.scaled_dot_product_attention`（PyTorch 2.5 + CUDA 自动启用 Flash / memory-efficient）
- **分析路径**：`return_attention=True` 时回退手算 softmax，导出 `[B, heads, K, M]` / `[B, heads, K, K]` 权重

相对旧版 `nn.MultiheadAttention`，SDPA 在 batch=2048、K=500 时显著降低峰值显存，使 4090 可训练。

### 3.5 参数量

| 模块 | 参数量（约） |
|------|-------------|
| ExplicitSieveModule | 532 |
| NumberEncoder + Token Builders | ~120K |
| Cross-Attn + Self-Attn + Heads | ~300K |
| **合计** | **~424,000** |

### 3.6 消融接口

`IGOTAblation(sieve/attention/residual)`：将对应分支 logit 置零，供 Phase 4 因果实验。

---

## 4. 数据集

与 Phase 1/2 相同：`data/goldbach_train_v2_phase0.bin`

- 99,999 条；90% / 10% 划分，`seed=7`
- 训练集 90,000；验证集 9,999

---

## 5. 训练配置

### 5.1 正式实验（定稿）

| 参数 | 值 |
|------|-----|
| Phase | `phase3` |
| 注意力实现 | SDPA（Flash 启用） |
| Batch size | **2048** |
| 学习率 | **0.0025** |
| 优化器 | Adam |
| 梯度裁剪 | `max_norm=1.0` |
| Epochs | 10 |
| 设备 | CUDA（RTX 4090） |
| Phase 2 参照 | `results/phase2_bs2048/train_report.json` |

**学习率说明**：batch=2048 时使用 `lr=0.01` 曾在 pilot 中 epoch 7 出现 **NaN**；降至 `0.0025` 后训练稳定。

### 5.2 Pilot 实验记录

| 运行 | 注意力 | batch | lr | 结果 | 产出 |
|------|--------|-------|-----|------|------|
| Pilot A | `nn.MultiheadAttention` | 512 | 0.01 | 10 epoch 完成，top1=43.5% | `results/phase3/` |
| Pilot B | SDPA | 2048 | 0.01 | epoch 7 NaN | `results/phase3_flash_bs2048/`（已覆盖） |
| **正式** | **SDPA** | **2048** | **0.0025** | **10 epoch 稳定** | **`results/phase3_flash_bs2048/`** |

### 5.3 训练命令（正式）

```powershell
$env:PYTHONUNBUFFERED=1
.\.venv\Scripts\Activate.ps1

python train\train.py --phase phase3 --epochs 10 --batch-size 2048 --lr 0.0025 --device cuda `
  --output-dir results\phase3_flash_bs2048 `
  --phase2-report results\phase2_bs2048\train_report.json
```

### 5.4 产出物

| 文件 | 说明 |
|------|------|
| `results/phase3_flash_bs2048/best.pt` | 最优 checkpoint（epoch 10，top1 最高） |
| `results/phase3_flash_bs2048/last.pt` | 最后一轮 |
| `results/phase3_flash_bs2048/train_report.json` | 完整日志 |
| `results/phase3_flash_benchmark.json` | SDPA 显存/速度基准 |

---

## 6. SDPA 显存与速度基准

环境：PyTorch 2.5.1+cu121，RTX 4090，Flash SDPA 已启用。

工具：`tools/benchmark_phase3_memory.py`（须 **单独运行**，勿与训练并行）

| Batch | 单步耗时（稳态） | 峰值显存 | 每 epoch 步数 |
|-------|------------------|----------|---------------|
| 512 | 70.0 ms | 3.4 GB | 176 |
| 1024 | 136.1 ms | 6.8 GB | 88 |
| **2048** | **268.4 ms** | **13.5 GB** | **44** |

**冷启动注意**：batch=2048 首个计时步在无充分 warmup 时可超过 **20s**（cuDNN 自动调优），benchmark 须 `warmup>=8` 才能得到稳态计时。

**与 Phase 2 对比**：Phase 2（无 self-attn）batch=2048 约 33s/10 epoch；Phase 3 SDPA batch=2048 约 **147s/10 epoch**（self-attn O(K²) 是主要开销）。

---

## 7. 实验结果

### 7.1 Phase 2 参照（末 epoch）

| 指标 | Phase 2（`phase2_bs2048`，ep10） |
|------|--------------------------------|
| val BCE | 0.126 |
| top1 | 42.6% |
| top5 | 86.7% |
| topK | 100% |
| avg_checks | 2.90 |

Phase 2 最优（ep3）：top1 **43.8%**，BCE **0.130**，checks **2.90**。

### 7.2 正式实验逐 epoch（SDPA，batch=2048，lr=0.0025）

| Epoch | train BCE | val BCE | top1 | top5 | topK | avg_checks |
|-------|-----------|---------|------|------|------|------------|
| 1 | 0.343 | 0.261 | 42.1% | 86.7% | 100% | 3.15 |
| 2 | 0.162 | 0.130 | 43.1% | 87.1% | 100% | 3.07 |
| 3 | 0.128 | 0.127 | 42.9% | 86.9% | 100% | 3.08 |
| 4 | 0.131 | 0.130 | 43.2% | 87.0% | 100% | 3.04 |
| 5 | 0.128 | 0.127 | 43.2% | 86.4% | 100% | 2.93 |
| 6 | 0.126 | 0.126 | 43.0% | 87.0% | 100% | **2.86** |
| 7 | 0.126 | 0.126 | 43.2% | 86.9% | 100% | 2.89 |
| 8 | 0.126 | 0.125 | 43.3% | 86.8% | 100% | 2.89 |
| 9 | 0.126 | 0.125 | 42.5% | 86.6% | 100% | 2.91 |
| **10** | **0.126** | **0.125** | **43.5%** | **86.8%** | **100%** | **2.89** |

- 总耗时：**146.5s**（每 epoch ~14.5–15.1s）
- 最优 checkpoint：**epoch 10**（top1=43.5%）

### 7.3 attention_gain（Phase 3 ep10 vs Phase 2 ep10）

| 指标 | Phase 2 ep10 | Phase 3 ep10 | attention_gain |
|------|--------------|--------------|----------------|
| val BCE | 0.126 | **0.125** | **−0.0003** |
| top1 | 42.6% | **43.5%** | **+0.81pp** |
| top5 | 86.7% | 86.8% | +0.12pp |
| topK | 100% | 100% | 0 |
| avg_checks | 2.90 | **2.89** | **−0.013** |

### 7.4 三阶段横向对比

| 实验 | 注意力 | batch | 耗时 | best top1 | best top5 | best BCE | best checks |
|------|--------|-------|------|-----------|-----------|----------|-------------|
| Phase 2 | — | 2048 | 34s | **43.8%** | 87.0% | 0.130 | 2.90 |
| Phase 3 pilot | MHA | 512 | 197s | 43.5% | **87.4%** | 0.134 | 3.02 |
| **Phase 3 正式** | **SDPA** | **2048** | **147s** | 43.5% | 86.8% | **0.125** | **2.89** |

### 7.5 收敛性分析

| 观察 | 结论 |
|------|------|
| val BCE | epoch 2 后快速降至 0.125–0.130，低于 Phase 2 末 epoch |
| top1 | 稳定在 43.0%–43.5%，相对 Phase 2 末 epoch **+0.8pp** |
| top5 | 86.4%–87.1% 波动，**未突破 ~87% 平台** |
| topK | 全程 **100%** |
| avg_checks | epoch 5 后稳定在 **2.86–2.93**，略优于 Phase 2 |
| 训练稳定性 | `lr=0.0025` + 梯度裁剪下无 NaN |

---

## 8. 工程问题与排查记录

### 8.1 「长时间无输出」原因

| 原因 | 说明 | 处理 |
|------|------|------|
| stdout 块缓冲 | 后台运行时 epoch 结束才 flush | `print(..., flush=True)` + `PYTHONUNBUFFERED=1` |
| GPU 并行争用 | benchmark 与 train 同时跑导致极慢 | **顺序执行**，禁止并行 |
| cuDNN 冷启动 | batch=2048 首步可达 20s+ | benchmark `warmup>=8`；训练 epoch 1 略慢属正常 |

### 8.2 链路验证工具

`tools/smoke_phase3_pipeline.py`：分步计时（数据集→模型→首 batch→稳态步），用于快速确认链路：

```powershell
$env:PYTHONUNBUFFERED=1
python tools\smoke_phase3_pipeline.py 2048
```

冒烟结果（batch=2048）：首 batch 0.79s，稳态 **258ms/step**，1 epoch 训练 **~15s**。

---

## 9. 结论

### 9.1 Phase 3 已达成目标

1. **完整 IGOT 架构落地**：sieve + cross-attn + self-attn + residual 三路分解可训练、可 ablate
2. **SDPA 工程优化成功**：batch=2048 在 4090 上峰值 13.5GB，训练吞吐优于 batch=512 的 MHA pilot
3. **attention_gain 为正（top1）**：+0.81pp vs Phase 2 末 epoch，BCE 略降，checks 略改善
4. **topK 保持 100%**：attention 未破坏 sieve 覆盖率

### 9.2 Attention 的能力边界

| 能力 | Phase 2 best | Phase 3 正式 best | 评价 |
|------|--------------|-------------------|------|
| topK | 100% | 100% | 保持 |
| avg_checks | 2.90 | 2.89 | 基本持平 |
| top1 | 43.8% | 43.5% | 略低於 Phase 2 最优 ep，但高于 Phase 2 末 epoch |
| top5 | 87.0% | 86.8% | **未突破平台** |

**核心结论**：1 层 cross + 1 层 self-attention 带来了与 Phase 2 同量级的 top1 提升，但 **未能突破 top5 ≈ 87% 的排序平台**。更高收益可能需要：加深 attention、改进 token 特征、更长训练或不同的 residual 分支设计（恢复 Phase 2 的手工特征 MLP）。

### 9.3 正式定稿

| 项目 | 定稿值 |
|------|--------|
| 模型 | `IGOTModel`（SDPA） |
| Batch size | 2048 |
| 学习率 | 0.0025 |
| 正式 checkpoint | `results/phase3_flash_bs2048/best.pt`（epoch 10） |
| 正式报告 | `results/phase3_flash_bs2048/train_report.json` |
| Phase 2 参照 | `results/phase2_bs2048/train_report.json` |
| attention_gain (ep10) | top1 +0.81pp, top5 +0.12pp, checks −0.013, BCE −0.0003 |

---

## 10. 代码结构

```text
model/
  igot.py                 # IGOTModel + IGOTAblation
  tokens.py               # PrimeTokenBuilder, ModulusTokenBuilder
  attention.py            # MultiHeadSDPA, Cross-Attn, Self-Attn
  sieve_baseline.py       # ExplicitSieveModule（复用）
  number_encoder.py       # NumberEncoder（复用）
  test_igot.py            # Phase 3 单元测试

train/
  train.py                # --phase phase3, attention_gain, grad clip

tools/
  benchmark_phase3_memory.py  # SDPA 显存/速度基准
  smoke_phase3_pipeline.py    # 链路分步冒烟测试
```

---

## 11. 复现实验清单

```powershell
# 1. 环境
.\.venv\Scripts\Activate.ps1
$env:PYTHONUNBUFFERED=1

# 2. 单元测试
python -m unittest discover -s model -p "test_*.py" -v

# 3. 链路冒烟（可选）
python tools\smoke_phase3_pipeline.py 2048

# 4. SDPA 基准（单独跑，勿与训练并行）
python tools\benchmark_phase3_memory.py --batch-sizes 512 1024 2048 --warmup 8 --repeats 8

# 5. Phase 3 正式训练
python train\train.py --phase phase3 --epochs 10 --batch-size 2048 --lr 0.0025 --device cuda `
  --output-dir results\phase3_flash_bs2048 `
  --phase2-report results\phase2_bs2048\train_report.json
```

---

## 12. Phase 3 未完成项与深度实验（循序渐进）

基线（1×cross + 1×self、10 epoch、**固定 LR**）已完成；**深度实验尚未开始**。

完整路线见 **`docs/phase3_deep_experiment_plan.md`**。四套方案**分阶段全部验证**：

| Stage | 方案 | 实验内容 |
|-------|------|----------|
| I | A | Cross=**1 固定**，Self L=**1..6** |
| II | B | 在 L_self\* 上，Cross C=**1..3** |
| III | C | **绑深** L_cross = L_self = **1..6** |
| IV | D | 局部网格（可选） |

| 项目 | 基线（本文 §7） | 深度实验 |
|------|----------------|----------|
| Cross / Self | 1 / 1 | 按 Stage 扫描或绑深 |
| Epoch | 10 | **50**（可增至 100） |
| 学习率 | **固定** `lr=0.0025` | **warmup + cosine** |
| 损失曲线 | 仅 JSON | Stage 0 回溯基线 + 每 run 插图写入 md |

**说明**：Stage I 先固定 Cross=1 是为了**控制变量**、优先检验 Self 深度；Cross 深度在 Stage II/III 单独验证。当前 `train.py` 尚无训练 LR warmup（实施 Stage 0 时添加）。

---

## 13. 下一步：Phase 4

Mechanistic Interpretability 实验（见 `docs/igot_implementation_plan.md` 第 8 节）：

| 实验 | 目的 |
|------|------|
| Logit 分解 | 单独评估 sieve / attention / residual 的 top1/top5 |
| Modulus ablation | 移除单个 \(M_j\) token |
| Head-modulus alignment | cross-attn 权重与 mod \(r_j\) 的对齐 |
| Head ablation | 逐个 zero-out attention head |
| Candidate clustering | self-attn 是否形成 mod 30 / mod 210 等价类 |

```text
analysis/ablation.py
analysis/attention_viz.py
analysis/logit_decompose.py
```
