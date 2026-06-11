# WP-F · F-4 回归重训预注册设计

> **文档定位**：F-4 回归重训与 P0 checks 重评的执行规格；开跑前须 commit 本文件（预注册凭证）。  
> **前置**：WP-F 修复部分关账（见 `wpf_summary.md` §7）  
> **版本**：v0.1（2026-06-11）

---

## 0. 执行摘要

F-4 验证：**WP-F 修复后，Phase 1/2/3 在统计协议下行为符合预期**——bucket 1 病理消除、大 N 桶指标在噪声带内、为 P1b / WP-C 提供干净基线。

**v3 数据决策（已拍板）**：

| 项 | 状态 |
|----|------|
| v3 数据集（450k / 50k / 500k test） | **未生成**（N-1 未实施；当前仅有 `goldbach_train_v2_phase0.bin`，99,999 条） |
| F-4 正式批 | **在 v3 上跑**（N-1 完成后） |
| F-4 冒烟批 | **可在 v2 上先跑**（验证修复不破坏训练链路） |
| P1b 冠军复测 | **必须在 v3 上**（不可妥协） |

---

## 1. 两阶段执行计划

```text
N-1 生成 v3 数据 ─────────────────────────────┐
                                               │
F-4a 冒烟（v2, 可选并行于 N-1）              │
  · 1 seed × 3 models × 短 epoch              │
  · 验收：bucket 1 ≤ 6、训练无 NaN、指标可产出   │
                                               ▼
F-4b 正式回归（v3）  ←── 依赖 N-1
  · 5 seeds × 3 models × 正式 epoch
  · 验收：§3 预注册判据
                                               ▼
P0 checks 子项重评（checks 项，v2 或 v3 对照报告）
                                               ▼
P1b 冠军复测（v3 only，可与 F-4b 同批训练）
```

---

## 2. F-4a：v2 冒烟批（可选，建议执行）

### 目的

在 v3 生成完成前，确认 WP-F 修复不破坏训练/评估链路；**不作为正式结论依据**。

### 配置

| 项 | 值 |
|----|-----|
| 数据 | `data/goldbach_train_v2_phase0.bin` |
| 划分 | val_fraction=0.1, seed=7（与历史实验一致） |
| seeds | **1**（seed=0） |
| epochs | Phase 1/2: **3**；Phase 3: **3**（或 1 epoch 冒烟） |
| batch | Phase 1: 512；Phase 2/3: 2048 |
| checkpoint | val **BCE** 最低（若 `train.py` 尚未切换，F-4 实施时一并改） |
| 输出 | `results/wpf_regress_smoke/` |

### 冒烟判据

| ID | 判据 |
|----|------|
| 4a-A | 三模型训练完成，无 NaN |
| 4a-B | bucket 1 avg_checks ≤ **6** |
| 4a-C | topK = **100%**（全桶） |
| 4a-D | `train_report.json` 正常产出 |

---

## 3. F-4b：v3 正式回归批（预注册）

### 3.1 数据（依赖 N-1）

| 划分 | 规模 | 文件（计划） |
|------|------|-------------|
| train | 450,000 | `data/goldbach_train_v3_train.bin` |
| val | 50,000 | `data/goldbach_train_v3_val.bin` |
| test | 500,000 | `data/goldbach_train_v3_test.bin`（上锁，F-4 不用于训练选点） |

生成规格见 `igot_implementation_plan_v2.md` §3.1（N-1）。

### 3.2 模型矩阵

| Run ID | phase | 架构 | 超参 |
|--------|-------|------|------|
| `f4-p1` | phase1 | `ExplicitSieveModule`, M=32 | lr=1e-2, batch=2048 |
| `f4-p2` | phase2 | `SieveResidualModel` | lr=1e-2, batch=2048 |
| `f4-p3` | phase3 | C=2, L=2, sequential | lr=0.0025, batch=2048, cosine 50ep |

每个 Run × **5 seeds** `{0,1,2,3,4}` → **15 runs**。

### 3.3 评估协议（与 WP-N 对齐）

| 条款 | 规定 |
|------|------|
| checkpoint 选择 | **val BCE** 最低 |
| 报告指标 | val 全量指标 + 分桶 |
| test | F-4 **不使用** test；test 留给 WP-C / P1b |
| 产出 | `results/wpf_regress/{run_id}/seed_{s}/train_report.json` + 曲线 PNG |

### 3.4 预注册判据（F-4b 完成标准）

| ID | 判据 | 阈值 |
|----|------|------|
| 4b-A | bucket 1 avg_checks（val，5-seed mean） | ≤ **6** |
| 4b-B | 全桶 topK | **100%** |
| 4b-C | 大 N 桶（bucket ≥ 7）top1 相对修复前历史 | 差异 ≤ **0.5pp**（参照 Phase 2/3 旧报告同桶） |
| 4b-D | 15/15 runs 完成，无 NaN | 必须 |
| 4b-E | Phase 1 val BCE | 应低于修复前同配置（eligibility mask 减少无效位噪声） |

### 3.5 P1b 冠军复测（捆绑 F-4b，v3 only）

| 项 | 值 |
|----|-----|
| 配置 | Phase 3：C=2, L=2, sequential（与深度实验同构） |
| seeds | 5 |
| 选点 | val BCE 中位数 seed 的 checkpoint |
| test | **一次性**评估（WP-C 标尺一-b，单独脚本） |
| 预测 | test top1 mean ∈ NULL-0 ± 0.3pp；44.63% 不可复现 |
| 禁止 | 使用 `results/phase3_depth/stageII/B_C2_L2/best.pt` |

P1b 训练可与 `f4-p3` 合并（同一 5-seed 跑，额外记录 test 评估时间戳）。

---

## 4. P0 checks 子项重评

| 子项 | 时机 | 数据 |
|------|------|------|
| checks ≤ 0.1 | F-4b 完成后 | 用 NULL-0 vs **F-4b 重训后** Phase 1 best（v3 val） |
| 预期 | Phase 1 bucket 1 病理消除后 | checks 差 ≤ 0.1 |

top1 / top5 子项留待 WP-C 全标尺（v3 test）。

---

## 5. 工程前置（F-4 开跑前代码清单）

| # | 项 | 状态 |
|---|-----|------|
| 1 | `train.py`：val BCE 选 checkpoint | **待实施**（当前仍 val top1） |
| 2 | `train.py`：`--seeds` 多 seed 循环 | **待实施** |
| 3 | `train/dataset.py`：v3 格式 / split 加载 | **待 N-1** |
| 4 | `tools/run_wpf_f4.ps1` 批跑脚本 | **待实施** |
| 5 | WP-F 修复代码 | **已完成** |

建议顺序：**N-1 + train.py 协议升级** → F-4a 冒烟（可并行）→ F-4b + P1b。

---

## 6. 资源预估

| 阶段 | GPU (4090) | 备注 |
|------|------------|------|
| N-1 数据生成 | 0（CPU 一夜） | gmpy2 + 8 workers |
| F-4a 冒烟 | ~15 min | v2, 1 seed, 短 epoch |
| F-4b 正式 | ~2–3 h | 15 runs, 50 epoch Phase 3 为主耗时 |
| P1b test 评估 | ~5 min | 冻结后一次性 |

**铁律**：GPU 串行，单进程。

---

## 7. 预注册声明（开跑 F-4b 前填写）

> 提交时间：____（git commit: ____）  
> 声明：§3.4 判据在 F-4b 产出任何 val 指标前写定，事后不修改。

---

## 8. 阶段结果（实施后填写）

### F-4a 冒烟

> 完成时间：____  
> 判据 4a-A–D：____

### F-4b 正式

> 完成时间：____  
> 15 runs 摘要：____  
> P1b test top1 mean ± sd：____
