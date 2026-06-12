# WP-F · F-4 回归重训预注册设计

> **文档定位**：F-4 回归重训与 P0 checks 重评的执行规格；开跑前须 commit 本文件（预注册凭证）。  
> **前置**：WP-F 修复部分关账（见 `wpf_summary.md` §7）  
> **版本**：v0.3（2026-06-11，预注册放行版）

---

## 0. 执行摘要

F-4 验证：**WP-F 修复后，Phase 1/2/3 在统计协议下行为符合预期**——bucket 1 病理消除、大 N 行为无意外偏移、为 P1b / WP-C 提供干净基线。

**方法论原则**：修复前后对照必须在**同一数据集、同一指标定义、同一训练预算**下进行；跨数据集（v2 vs v3）的数字不可直接比大小。v3 正式批的对照对象是 **NULL-0 / NULL-1 理论基线**，不是旧报告历史数字。

**v3 数据决策（已拍板）**：

| 项 | 状态 |
|----|------|
| v3 数据集（450k / 50k / 500k test） | **未生成**（N-1 未实施；当前仅有 `goldbach_train_v2_phase0.bin`，99,999 条） |
| F-4a v2 对照批 | **修复副作用的合法证据**（同数据、同 epoch、历史 checkpoint 用新指标重评） |
| F-4b 正式批 | **在 v3 上跑**（N-1 + 数据审计闸门通过后） |
| P1b 冠军复测 | **必须在 v3 上**（不可妥协） |

---

## 1. 两阶段执行计划

```text
N-1 生成 v3 数据
    ↓
v3 数据验收闸门（附录 C 1/2/5/6）── 不通过则全线作废
    ↓
F-4a v2 对照批（可与 N-1 并行）
  · 1–2 seeds × 3 models × 10 epoch（与历史一致）
  · 历史 checkpoint 用 masked 指标重评
  · 验收：§2 判据（含 4a-E 修复前后 BCE / 大 N top1）
    ↓
F-4b 正式回归（v3）
  · 5 seeds × 3 models × 正式 epoch
  · 验收：§3.5 判据（对照 NULL-0 / NULL-1，非历史数字）
    ↓
P0 checks 子项重评（checks 项）
    ↓
P1b 冠军复测（v3 test，NULL-0 版本预注册钉死）
```

---

## 2. F-4a：v2 对照批（修复副作用检验）

### 目的

在**同一 v2 val** 上证明 WP-F 修复：

1. 消除 bucket 1 病理（结构性改进，预期通过）；
2. **不改变**大 N 桶排序行为（4a-E，§2.1 三级规则）；
3. Phase 1 masked BCE 不劣于修复前（4a-F，同指标重评旧 checkpoint）。

**本批是「修复前 vs 修复后」的唯一合法对照**；不得用 v3 数字与 v2 历史数字直接比较。

### 配置

| 项 | 值 |
|----|-----|
| 数据 | `data/goldbach_train_v2_phase0.bin` |
| 划分 | val_fraction=0.1, seed=7（与历史实验一致） |
| seeds | **1–2**（seed=0；可选 seed=1 复现稳健性） |
| epochs | Phase 1/2/3 均为 **10**（与历史正式实验一致，非短 epoch 冒烟） |
| batch | Phase 1: **512**；Phase 2/3: **2048**（与历史一致） |
| Phase 3 lr | **0.0025**（与 `phase3_flash_bs2048` 一致） |
| checkpoint | val **BCE**（masked）最低 |
| 输出 | `results/wpf_regress_f4a/` |

### 历史参照 checkpoint（仅用于同数据重评，不重训）

| phase | 路径 | 原选点依据 |
|-------|------|-----------|
| Phase 1 | `results/phase1/`（epoch 10 末或 best） | 历史 val top1 |
| Phase 2 | `results/phase2_bs2048/best.pt` | 历史 val top1（epoch 3） |
| Phase 3 | `results/phase3_flash_bs2048/best.pt` | 历史 val top1（epoch 10） |

**重评协议**：对上述 checkpoint 在 v2 val 上用 **当前 masked BCE / 分桶 metrics 代码路径** 重新评估（`tools/reval_checkpoint.py`，F-4 实施时编写），产出 `reval_report.json` 作为「修复前」基线。

### 2.1 逐桶偏移判据（4a-E / 4b-C 共用）

大 N 桶 **bucket 7–18** 共 12 桶；F-4a / F-4b 各覆盖 Phase 1/2/3 → **比较总数 N_cmp = 36**。  
逐桶 Δ = top1_new − top1_ref（4a-E：ref = 历史 checkpoint 重评；4b-C：ref = NULL-0，同 val 配对）。  
SE_bucket ≈ √(p(1−p)/n)，n = 该桶 val 样本量，p = 两 top1 的均值。

**三级规则**（三条须同时满足；任一条触发即该判据失败）：

| 级 | 规则 | 裁决 |
|----|------|------|
| **硬上限** | 任一桶 \|Δ\| > **3×SE** | **立即判失败** |
| **软配额** | \|Δ\| ∈ (2×SE, 3×SE] 的桶数 | ≤ **⌊0.10 × N_cmp⌋ = 3**（给二项抽样运气留配额） |
| **系统性偏移** | 全部 N_cmp 个 Δ 的**符号检验** | p < **0.01** → 判失败（防同向小幅偏移漏网） |

\|Δ\| ≤ 2×SE 的桶视为通过，不计入软配额。

> **附加分析（不参与裁决）**：同一 val 上的配对 bootstrap 可作参考；预注册裁决仅采用上表 SE 公式与三级规则。

**实施笔记（多种子聚合）**：N_cmp = 36 的定义是 **每 phase、每桶一次比较**。F-4a 若跑 2 个种子、F-4b 若跑 5 个种子，§2.1 中的 **top1_new 须先对种子取均值**（同 phase、同桶），再与 ref 算 Δ 并套用三级规则。**禁止**逐种子各跑一遍三级规则——否则 N_cmp 翻倍、软配额与符号检验的标定均失效。4a-F / 4b-E 等同理：多种子时先取均值再与 ref 比较。

### 判据（F-4a 完成标准）

| ID | 判据 | 阈值 / 方法 |
|----|------|------------|
| 4a-A | 三模型训练完成，无 NaN | 必须 |
| 4a-B | bucket 1 avg_checks（新模型，v2 val） | ≤ **6** |
| 4a-C | 全桶 topK | **100%** |
| 4a-D | `train_report.json` + 曲线正常产出 | 必须 |
| **4a-E** | **大 N 桶 top1：新 vs 历史 checkpoint 重评** | §2.1 三级规则（N_cmp = 36，v2 val） |
| **4a-F** | **Phase 1 masked BCE：新 vs 历史 checkpoint 重评** | 新 − 旧 ≤ **0.001**（同 v2 val、同 masked 指标定义） |

---

## 3. F-4b：v3 正式回归批（预注册）

### 3.1 数据（依赖 N-1）

| 划分 | 规模 | 文件（计划） |
|------|------|-------------|
| train | 450,000 | `data/goldbach_train_v3_train.bin` |
| val | 50,000 | `data/goldbach_train_v3_val.bin` |
| test | 500,000 | `data/goldbach_train_v3_test.bin`（上锁，F-4 不用于训练选点） |

生成规格见 `igot_implementation_plan_v2.md` §3.1（N-1）。

### 3.2 v3 数据验收闸门（训练前置，不可跳过）

v3 生成后、**任何 F-4b / P1b 训练开始前**，必须先执行 `igot_experiment_plan_v2.1.md` **附录 C** 下列项：

| 附录 C 项 | 内容 |
|-----------|------|
| **1** | 划分泄漏：同一 N 不得跨 train / val / test |
| **2** | 重复样本：同一划分内 N 无重复 |
| **5** | 生成器 RNG：各划分独立播种，无序列相关 |
| **6** | bitset roundtrip：valid_mask 打包/解包（全 0、全 1、交替、随机 1000 例） |

**产出**：`results/wpf_regress/v3_audit_report.json`（含每项 pass/fail 与样本计数）。  
**裁决**：任一项 fail → **v3 作废**，不得开训；50 万 test 是后续所有正式裁决的法庭，此闸不能省。  
审计报告须随各 run 的 `train_report.json` 一并归档。

### 3.3 模型矩阵

| Run ID | phase | 架构 | 超参 |
|--------|-------|------|------|
| `f4-p1` | phase1 | `ExplicitSieveModule`, M=32 | lr=1e-2, batch=2048 |
| `f4-p2` | phase2 | `SieveResidualModel` | lr=1e-2, batch=2048 |
| `f4-p3` | phase3 | C=2, L=2, sequential | lr=0.0025, batch=2048, cosine 50ep |

每个 Run × **5 seeds** `{0,1,2,3,4}` → **15 runs**。

### 3.4 评估协议（与 WP-N 对齐）

| 条款 | 规定 |
|------|------|
| checkpoint 选择 | **val BCE**（masked）最低 |
| 报告指标 | v3 val 全量指标 + 分桶 |
| test | F-4 **不使用** test；test 留给 WP-C / P1b |
| NULL-0 | 与模型**同场次**在 v3 val 上评估（同 metrics 代码路径） |
| NULL-1 | **必须在 v3 train 上重新拟合** isotonic 保序回归后，再在 v3 val 上评估；**不得**复用 v2 校准曲线（数据换了须重标；且只准用 train，纪律同 N-2） |
| 产出 | `results/wpf_regress/{run_id}/seed_{s}/train_report.json` + 曲线 PNG |

### 3.5 预注册判据（F-4b 完成标准）

| ID | 判据 | 阈值 / 方法 |
|----|------|------------|
| 4b-A | bucket 1 avg_checks（v3 val，5-seed mean） | ≤ **6** |
| 4b-B | 全桶 topK | **100%** |
| **4b-C** | **大 N 桶 top1：模型 vs NULL-0** | §2.1 三级规则（N_cmp = 36，v3 val 配对） |
| 4b-D | 15/15 runs 完成，无 NaN | 必须 |
| **4b-E** | **Phase 1 masked BCE（v3 val，5-seed mean）** | ∈ **[NULL-1, NULL-1 + 0.07]**（NULL-1：v3 train 重拟合 → v3 val 评估；上界 0.07 见 `null_model_verification.md` §5.2） |

> **禁止**：将 v3 指标与 v2 历史报告数字直接比较（数据集不同；旧 val 大 N 桶仅数百条，SE ≈ ±2pp，0.5pp 阈值统计不可达）。

### 3.6 P1b 冠军复测（捆绑 F-4b，v3 only）

| 项 | 值 |
|----|-----|
| 配置 | Phase 3：C=2, L=2, sequential（与深度实验同构） |
| seeds | 5 |
| 选点 | val BCE 中位数 seed 的 checkpoint |
| test | **一次性**评估（WP-C 标尺一-b） |
| 预测 | test top1 mean ∈ **NULL-0_ref ± 0.3pp**；44.63% 不可复现 |
| 禁止 | 使用 `results/phase3_depth/stageII/B_C2_L2/best.pt` |

#### NULL-0 版本钉死（预注册）

P1b 裁决所用的 NULL-0 版本在 **开跑 F-4b 前** 写入下表，事后不修改：

| 字段 | 值 |
|------|-----|
| `NULL-0_ref` 版本 | **v1.1**（若移交项 4 交付 **(1)+(2)** 完成并通过专家回归）**否则 v1.0**（交付 (3) 仅阻塞 M-scaling，不阻塞 P1b） |
| 版本记录 | git commit hash of `model/null_model.py` + 文档增补节 |
| 评估场次 | NULL-0_ref 在 **v3 test** 上的评估与 P1b test 评估 **同场次、同脚本、连续执行**（时间戳写入 `p1b_eval_log.json`） |

v1.1 主要影响小 N 桶，会动 test 全集均值；版本不钉死则 ±0.3pp 判据不可复现。

P1b 训练可与 `f4-p3` 合并（同一 5-seed 跑）。

---

## 4. P0 checks 子项重评

| 子项 | 时机 | 数据 |
|------|------|------|
| checks ≤ 0.1 | F-4b 完成后 | **NULL-0_ref**（§3.6 钉死版本）vs **F-4b 重训后** Phase 1 best（v3 val） |
| 预期 | bucket 1 病理消除后 | checks 差 ≤ 0.1 |

top1 / top5 子项留待 WP-C 全标尺（v3 test）。

---

## 5. 工程前置（F-4 开跑前代码清单）

| # | 项 | 状态 |
|---|-----|------|
| 1 | `train.py`：val BCE（masked）选 checkpoint | **待实施** |
| 2 | `train.py`：`--seeds` 多 seed 循环 | **待实施** |
| 3 | `train/dataset.py`：v3 格式 / split 加载 | **待 N-1** |
| 4 | `tools/run_wpf_f4.ps1` 批跑脚本 | **待实施** |
| 5 | `tools/reval_checkpoint.py`：历史 ckpt masked 指标重评 | **待实施**（F-4a 必需） |
| 6 | `tools/v3_data_audit.py`：附录 C 1/2/5/6 | **待实施**（F-4b 闸门） |
| 7 | WP-F 修复代码 | **已完成** |

建议顺序：**N-1** → **v3 审计闸门** → **F-4a**（可与 N-1 并行至审计前）→ **F-4b + P1b**。

---

## 6. 资源预估

| 阶段 | GPU (4090) | 备注 |
|------|------------|------|
| N-1 数据生成 | 0（CPU 一夜） | gmpy2 + 8 workers |
| v3 数据审计 | 0（CPU 分钟级） | 闸门脚本 |
| F-4a 对照批 | ~45–90 min | v2, 1–2 seeds, 10 epoch × 3 phases |
| F-4b 正式 | ~2–3 h | 15 runs, 50 epoch Phase 3 为主耗时 |
| P1b test 评估 | ~5 min | 与 NULL-0_ref 同场次 |

**铁律**：GPU 串行，单进程。

---

## 7. 预注册声明（开跑 F-4b 前填写）

> 提交时间：2026-06-12（git commit: `e453df9` + F-4 工具链后续 commit）  
> `NULL-0_ref` 版本：**v1.0**（v1.1 未落地；移交项 4 仅完成前不切换）  
> 声明：§2 判据（4a-A–F）与 §3.5 判据（4b-A–E）、§3.6 NULL-0 版本在产出任何 F-4b val / P1b test 指标前写定，事后不修改。

---

## 8. 阶段结果（实施后填写）

### F-4a v2 对照批

> 完成时间：____  
> 判据 4a-A–F：____  
> 历史 checkpoint 重评摘要：____

### F-4b 正式

> v3 审计闸门：____（commit / report path）  
> 完成时间：____  
> 15 runs 摘要：____  
> P1b test top1 mean ± sd：____（NULL-0_ref: ____）
