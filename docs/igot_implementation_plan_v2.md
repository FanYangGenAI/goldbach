# IGOT 后续实验实施方案 v2

> **文档定位**：将 `igot_experiment_plan_v2.1.md` 的科学路线落地为可执行开发/实验计划。  
> **版本**：v2.0（2026-06-11）  
> **前置条件**：Phase 0–3b 已完成（见 §1）；原 `igot_implementation_plan.md` 中 Phase 4「直接做机制逆向」路线**废止**，由本方案替代。  
> **关联文档**：`igot_experiment_plan_v2.1.md`（科学方案）、`phase1_sieve_baseline_experiment.md`、`phase2_sieve_residual_experiment.md`、`phase3_igot_transformer_experiment.md`、`phase3_deep_experiment_plan.md`

---

## 0. 执行摘要

Phase 1–3 的架构搜索表明：在模余特征给定条件下，top1 全部落于 43.8%–44.6% 窄带，与解析数论「天花板假说」高度吻合。后续工作**不再以堆架构提精度为主线**，而是：

1. **WP-N + WP-C**：建立统计可判的评估体系，用零参数 NULL-0 做预注册天花板检验（回答 Q1）；
2. **WP-F**：修复 eligibility、小素数边界、logit 不可辨识三类结构性缺陷；
3. **Phase 4R**：研究全特征网络如何实现 NULL-0（回答 Q2）；
4. **Phase 5**：研究原始比特域内模运算电路的自发涌现（回答 Q3，主创新点）；
5. **Phase 6**：汇总写作。

**硬性闸门**（未满足不得进入下一阶段）：

| 闸门 | 条件 |
|------|------|
| G1 | 附录 B 预注册声明在 WP-C 实验产出数据**之前** commit |
| G2 | WP-N 完成：NULL-0 在旧验证集通过 P0 |
| G3 | WP-F §5.3 可辨识性修复完成 |
| G4 | WP-C 判定树走通（或异常审计完成）后，方可启动 Phase 4R |
| G5 | Phase 4R backbone 来自 v3 冠军复测 5 seeds 的 val BCE 中位 checkpoint，**禁止**使用 `results/phase3_depth/stageII/B_C2_L2/best.pt` |

---

## 1. 已完成阶段（归档，不再扩展）

| 阶段 | 状态 | 关键产出 | 文档 |
|------|------|----------|------|
| Phase 0 | 完成 | v2 数据、`coverage@500=100%`（N≥6） | `igot_implementation_plan.md` §4 |
| Phase 1 | 完成 | `ExplicitSieveModule`，top1≈42.9% | `phase1_sieve_baseline_experiment.md` |
| Phase 2 | 完成 | `SieveResidualModel`，top1≈43.8% | `phase2_sieve_residual_experiment.md` |
| Phase 3 | 完成 | `IGOTModel`（SDPA），top1≈43.5% | `phase3_igot_transformer_experiment.md` |
| Phase 3b | 完成 | 15 runs 深度扫描，best top1=44.63% | `phase3_deep_experiment_plan.md` §13 |

**Phase 3b 遗留项（纳入 WP-C 标尺四，不重训）**：

- 15 runs 按 val BCE ≤ 0.131 分收敛/未收敛组；
- 排序–校准解离散点图；
- P4 正式裁决待 NULL-0 计算后执行。

---

## 2. 总体路线图

```text
[当前] Phase 3b 收尾
   │
   ├─ WP-N  统计基础设施 + Null Model     (~2 天, CPU 为主)  ─┐
   │                                                           │
   ├─ WP-F  模型与数据修复                  (~1.5 天)         ├─ 可并行
   │                                                           │
   └─ 预注册 commit（附录 B）               (开跑 WP-C 前)    ─┘
              │
              ▼
        WP-C  天花板预注册检验              (~1.5 天, ~3h GPU)
              │
              ├─ P0–P4 全部成立 → Q1 定稿
              │                → 启动 Phase 4R / Phase 5（可并行）
              └─ 任一不成立 → 异常审计 §4.6（`igot_experiment_plan_v2.1.md`）
              │
              ▼
        Phase 4R  Track A 机制分析            (~3 天, ~4h GPU)
              │
              ▼
        Phase 5   Track B 原始比特机制        (~1–2 周, ~20–60h GPU)
              │
              ▼
        Phase 6   汇总与写作                  (~1 周)
```

**铁律**：GPU 串行、单进程；每 run 产出 `train_report.json` + 曲线 PNG；各阶段「阶段结果」节填写完毕方可标记完成。

---

## 3. WP-N：统计基础设施与解析 Null Model

> **科学目标**：回答「现有神经网络指标是否在 NULL-0 预测带内」之前，先让 NULL-0 可运行、让 test 集统计可分辨 0.2pp 效应。  
> **对应问题**：Q1 的基础设施。  
> **可与 WP-F 并行**。

### 3.1 入口条件

- [ ] Phase 3b 实验数据已归档；
- [ ] 开发环境可用（Python venv、`gmpy2` 可安装）。

### 3.2 任务清单

#### N-1 数据集 v3 生成

| 项 | 规格 |
|----|------|
| train | 450,000 条 |
| val | 50,000 条 |
| test | 500,000 条（上锁，全程只评一次） |
| 采样 | 18 桶 log-uniform，沿用 `generator/python/buckets.py` |
| 训练域 | N ≥ 6，K = 500 |
| 划分纪律 | **按 N 取值**划分，同一 N 不跨 split |
| 素性测试 | `gmpy2.is_prime`，multiprocessing，8 workers |
| 格式 | v2 85 字节 + 文件头新增 `split_id`、`dedup_hash`（version 升级为 3） |

**代码改动**：

| # | 文件 | 改动 |
|---|------|------|
| 1 | `generator/python/format_io.py` | 新增 `VERSION_V3`、split 元数据字段 |
| 2 | `generator/python/generate.py` | 支持 `--split train/val/test`、按 N 去重、独立 RNG seed |
| 3 | `generator/python/labeling.py` | 确认 valid_mask 与 m=r_j 边界一致 |
| 4 | `tools/verify_dataset.py` | 扩展 v3：划分泄漏、重复 N、bitset roundtrip |
| 5 | `tools/split_dataset.py`（新建） | 从候选池按 N hash 划分并写 manifest |
| 6 | `train/dataset.py` | 支持 v3、按 split 加载、`scale_holdout` / `residue_holdout` 标志位 |

**产出**：

```text
data/goldbach_train_v3_train.bin
data/goldbach_train_v3_val.bin
data/goldbach_train_v3_test.bin      # 上锁清单写入 manifest
data/goldbach_manifest_v3.json
```

**生成命令（草案）**：

```powershell
$env:PYTHONUNBUFFERED=1
.\.venv\Scripts\Activate.ps1
pip install gmpy2

# 一夜批量生成（按 split 顺序执行，各 split 独立 seed）
python generator\python\generate.py --phase phase0 --format v3 --split train --count 450000 --workers 8
python generator\python\generate.py --phase phase0 --format v3 --split val   --count 50000  --workers 8
python generator\python\generate.py --phase phase0 --format v3 --split test  --count 500000 --workers 8

python tools\verify_dataset.py --manifest data\goldbach_manifest_v3.json --audit-splits
```

**降级**：若 test 生成超时，test 降至 200,000（SE ≈ 0.11pp，仍可用）。

#### N-2 NULL-0 / NULL-1 实现

| 文件 | 内容 |
|------|------|
| `model/null_model.py` | `NullModel`：NULL-0 概率与 logit（见 v2.1 附录 A） |
| `model/test_null_model.py` | 单元测试：eligibility、m=r_j、负数取模、与手算对照 |
| `tools/eval_null_model.py` | 在 val/test 上跑全套 metrics |
| `docs/null_model_verification.md` | NULL-0 vs Phase 1/2/3 对照表 |

**NULL-0 核心公式**（实现须在代码注释中写明）：

```text
m = N - p_i
m < 3           → p̂ = 0
m ∈ {r_1..r_M}  → p̂ = 1
∃j: r_j | m     → p̂ = ε (1e-4)
else            → p̂ = min(0.999, e^γ · ln(r_M) / ln(m))
```

**NULL-1**：`sklearn.isotonic.IsotonicRegression`，仅在 train 上拟合；排序指标与 NULL-0 逐样本相同。

#### N-3 评估协议升级

| # | 文件 | 改动 |
|---|------|------|
| 1 | `train/train.py` | checkpoint 选择改为 **val BCE 最低**（废止 val top1）；支持 `--seeds`、`--lr-schedule cosine` |
| 2 | `train/metrics.py` | 确保 NULL-0 与神经网络走同一代码路径、同一 tie-breaking |
| 3 | `tools/stats.py`（新建） | McNemar、bootstrap CI、Holm 校正；最小关注效应量判定 |
| 4 | `tools/eval_test_locked.py`（新建） | test 一次性评估脚本，写时间戳审计日志 |

**评估协议（全部后续实验强制执行）**：

| 条款 | 规定 |
|------|------|
| 模型选择 | val BCE 选 checkpoint |
| test 纪律 | 每个模型 test 只评一次，冻结后评估 |
| 多种子 | 结论表配置 ≥ 5 seeds，报告 mean ± sd |
| 显著性 | top1/top5：McNemar；checks/BCE：bootstrap 10⁴ 次；Holm α=0.01 |
| 最小效应量 | top1/top5：0.3pp；avg_checks：0.05；BCE：0.001 |

### 3.3 验收标准

| ID | 标准 | 验证方式 |
|----|------|----------|
| N-A | v3 三 split 生成完成，审计清单 1–4 通过 | `verify_dataset.py` 报告 |
| N-B | NULL-0 单元测试全绿 | `python -m unittest model.test_null_model` |
| N-C | **P0**：旧 9,999 val 上 NULL-0 与 Phase 2/3 最优 top1/top5/checks 差 ≤ 1pp/1pp/0.1 | `docs/null_model_verification.md` |
| N-D | NULL-1 BCE ≤ 0.127（旧 val） | 同上 |
| N-E | `train.py` 已切换 val BCE 选点 | 代码审查 + 冒烟训练 |

### 3.4 预估资源

| 资源 | 量 |
|------|-----|
| 人力 | ~2 天 |
| GPU | 0（数据生成 CPU，3–8 h/一夜） |
| 磁盘 | ~1M × 85B ≈ 85 MB 数据 + manifest |

### 3.5 阶段结果（实施后填写）

> 完成时间：____  
> P0 结果：____  
> v3 manifest SHA：____  
> 遗留问题：____

---

## 4. WP-F：模型与数据修复

> **科学目标**：消除阻碍公平比较与机制归因的三类管线缺陷。  
> **硬性闸门**：§4.3 完成前不得启动 Phase 4R。  
> **可与 WP-N 并行**。

### 4.1 任务清单

#### F-1 Eligibility Mask（§5.1）

```text
p_i > N - 3  ⇒  final_logit_i = -1e4，loss mask 该位置
```

| # | 文件 | 改动 |
|---|------|------|
| 1 | `model/sieve_baseline.py` | `ExplicitSieveModule` 最外层 eligibility |
| 2 | `model/igot.py` | 全模型 forward 传播 eligibility |
| 3 | `model/sieve_baseline.py` | 审计 `compute_sieve_matrix` 对 m≤0 的行为 |
| 4 | `model/test_sieve_baseline.py` | N ∈ {6, 8, 10, 100} 全候选单元测试 |

**预期效果**：bucket 1 `avg_checks` 从 ~24 降至 ≤ 6。

#### F-2 m = r_j 边界（§5.2）

```text
若 N - p_i ∈ {r_1, ..., r_M}：valid_mask=1，筛模块跳过 z 惩罚
```

| # | 文件 | 改动 |
|---|------|------|
| 1 | `model/sieve_baseline.py` | `is_small_prime(m)` 旁路 |
| 2 | `model/null_model.py` | 与筛模块语义对齐 |
| 3 | `generator/python/labeling.py` | 断言 m=r_j 时 valid_mask=1 |
| 4 | `generator/python/test_labeling.py` | 边界用例 |

#### F-3 Null-anchored Δ 架构（§5.3，Phase 4R 前置）

废弃联合训练三分解归因；新架构：

```text
final_logit_i = logit(NULL-0_i) + Δ_i(θ)    # NULL-0 冻结
```

| # | 文件 | 改动 |
|---|------|------|
| 1 | `model/igot_null_anchored.py`（新建） | `NullAnchoredIGOT`：冻结 null + 可训练 Δ |
| 2 | `train/train.py` | `--phase phase3-na`，Δ L2 正则 λ ∈ {0, 1e-4, 1e-3} |
| 3 | `model/test_igot_null_anchored.py`（新建） | 梯度不流入 NULL-0；Δ≡0 时等价 NULL-0 |

**Stage-wise 冻结（4R 消融备用）**：

```text
Stage A: 训 sieve 或直接用 NULL-0 → 冻结
Stage B: 训 attention → 冻结
Stage C: 训 residual
```

各阶段 test 嵌套似然差 = 分支贡献；**禁止**联合训练模型内部 logit 拆分做归因。

#### F-4 回归测试（§5.4）

在 v3 train/val 上重训三个代表配置各 5 seeds：

| 模型 | phase | 配置 |
|------|-------|------|
| Phase 1 | `phase1` | 线性筛，M=32 |
| Phase 2 | `phase2` | sieve + residual |
| Phase 3 | `phase3` | C=2, L=2, sequential |

```powershell
# 示例（5 seeds 循环）
python train\train.py --phase phase1 --dataset v3 --seeds 0,1,2,3,4 `
  --select-metric val_bce --output-dir results\wpf_regress\phase1
```

**回归预期**：

- bucket 1 avg_checks ≤ 6；
- 全桶 topK = 100%；
- 大 N 桶指标与修复前差异在噪声带内（±0.5pp top1）。

### 4.2 验收标准

| ID | 标准 |
|----|------|
| F-A | Eligibility 单元测试全绿 |
| F-B | m=r_j 标签断言 0 失败 |
| F-C | Null-anchored 模型可训练，Δ L2 正则可扫 |
| F-D | 回归测试 15 runs 完成，bucket 1 病理消除 |

### 4.3 预估资源

| 资源 | 量 |
|------|-----|
| 人力 | ~1.5 天 |
| GPU | ~2 h（回归 15 runs） |

### 4.4 阶段结果（实施后填写）

> 完成时间：____  
> bucket 1 avg_checks（修复后）：____  
> 回归测试摘要路径：____

---

## 5. WP-C：天花板预注册检验

> **科学目标**：正面回答 Q1——给定模余特征，是否存在超出 Hardy–Littlewood 的可学习排序信号。  
> **预期主结局**：零结果（无超额信号）。

### 5.1 入口条件（闸门）

- [ ] **G1**：附录 B 预注册声明已 commit（`docs/preregistration_wpc.md` 或 v2.1 附录 B 填空后提交）；
- [ ] **G2**：WP-N 完成，P0 通过；
- [ ] WP-F F-1/F-2 完成（F-3 可与 WP-C 部分并行，但 P1 训练须在 eligibility 修复后）。

### 5.2 预注册命题（开跑前锁定）

| ID | 命题 | 判定阈值 |
|----|------|----------|
| P0 | NULL-0 对齐旧 val | top1/top5/checks 差 ≤ 1pp/1pp/0.1；NULL-1 BCE ≤ 0.127 |
| P1 | 无超额信号（新 test） | 任一 NN vs NULL-0：top1/top5/checks 优势 < 0.3pp/0.3pp/0.05；vs NULL-1：BCE 优势 < 0.002 |
| P1b | 冠军不可复现 | `p3-B_C2_L2` 5 seeds test top1 均值 ∈ NULL-0 ± 0.3pp；44.63% 为顺序统计假象 |
| P2 | M-scaling 标度律 | top1(M) 与解析预测相对偏差 ≤ 10% |
| P3 | 可交换性 shuffle | 幸存候选打乱后 top1/checks 变化 < 0.3pp/0.05 |
| P4 | 深度无关性 | 已收敛 run（BCE≤0.131）落于 NULL-0 ± 1pp；深度–指标无显著正向 Spearman |
| P5 | Δ ≈ 0 | Null-anchored Δ 排序贡献 < 最小效应量；BCE 贡献 ≤ NULL-1−NULL-0 校准差 |

### 5.3 四个标尺实验

#### 标尺一：全模型 vs NULL-0（§4.1）

| Run | 模型 | 数据 | seeds | 选点 |
|-----|------|------|-------|------|
| `wpc-m1` | Phase 1 线性筛 | v3 | 5 | val BCE |
| `wpc-m2` | Phase 2 sieve+residual | v3 | 5 | val BCE |
| `wpc-m3` | Phase 3 (C=2,L=2) | v3 | 5 | val BCE |
| `wpc-null0` | NULL-0 | v3 test | — | — |
| `wpc-null1` | NULL-1 | v3 test | — | — |
| `wpc-champion` | Phase 3 冠军复测 (P1b) | v3 | 5 | val BCE → test 一次 |

```powershell
python train\train.py --phase phase3 --cross-attn-layers 2 --self-attn-layers 2 `
  --stack-mode sequential --dataset v3 --seeds 0,1,2,3,4 `
  --select-metric val_bce --epochs 50 --lr-schedule cosine `
  --output-dir results\wpc\champion_C2_L2

python tools\eval_test_locked.py --checkpoint results\wpc\champion_C2_L2\seed_2\best.pt `
  --split test --log results\wpc\champion_test_eval.jsonl
```

**产出**：`results/wpc/full_model_vs_null.json`、`docs/figures/wpc_forest_plot.png`

#### 标尺二：M-scaling（§4.2）

| Run ID | M | r_M | 模型 |
|--------|---|-----|------|
| `ms-32` | 32 | 137 | 线性筛 |
| `ms-64` | 64 | 313 | 线性筛 |
| `ms-168` | 168 | 997 | 线性筛 |
| `ms-1229` | 1229 | 9973 | 线性筛 |
| `ms-1229-null` | 1229 | 9973 | NULL-0(M=1229) |

**点预测（test top1）**：

| M | 预测 top1 | 预测 avg_checks |
|---|-----------|-----------------|
| 32 | 43% ± 2pp | 3.0 ± 0.2 |
| 64 | 50% ± 3pp | 2.4 ± 0.2 |
| 168 | 59% ± 3pp | 1.9 ± 0.15 |
| 1229 | 73% ± 4pp | 1.5 ± 0.1 |

**工程**：M=1229 时 z 矩阵分块流式计算，或二参数压缩版 `logit = b(scale) - w·[any_hit]` 互验。

```powershell
python train\train.py --phase phase1 --num-moduli 1229 --dataset v3 --seeds 0,1,2,3,4 `
  --output-dir results\wpc\ms-1229
```

**产出**：`docs/figures/wpc_mscaling_top1_vs_ln_rM.png`（**论文一主图**）

#### 标尺三：可交换性 shuffle（§4.3）

| 项 | 说明 |
|----|------|
| 脚本 | `tools/shuffle_survivor_test.py`（新建） |
| 输入 | Phase 3 最优模型在 test 上的逐样本排序 |
| 操作 | 对每个 N，幸存候选间随机打乱 10 次取均值 |
| 成本 | 零训练，~30 行核心逻辑 |

#### 标尺四：Phase 3 重分析（§4.4，不重训）

| 分析项 | 脚本/产出 |
|--------|-----------|
| 收敛过滤 | BCE ≤ 0.131 → 10 收敛 / 5 未收敛 |
| 主散点图 | 深度 vs top1/top5 + NULL-0 水平线 + ±2SE |
| 深度 Spearman | 已收敛组 |
| 排序–校准解离图 | (val BCE, best top1) 15 点散点 |

```powershell
python tools\reanalyze_phase3_depth.py `
  --reports results\phase3_depth\**\train_report.json `
  --null-baseline results\null\eval_report.json `
  --output docs\figures\wpc_phase3_reanalysis.png
```

**未收敛 run 清单**（预期）：`A_L4`、`A_L5`、`A_L6`、`B_C1_L2`、`C_L4`–`C_L6` 中 BCE 离群者。

### 5.4 判定树

```text
P0–P4 全部成立
  → Q1 定稿：未检出超额信号
  → 撰写 `docs/wpc_ceiling_test_report.md`
  → 解锁 Phase 4R、Phase 5

任一命题显著不成立
  → `igot_experiment_plan_v2.1.md` §4.6 异常审计（8 步）
  → 暂停 Phase 4R/5 直至审计结论
```

### 5.5 验收标准

| ID | 标准 |
|----|------|
| C-A | 预注册 commit 时间戳早于一切 WP-C 数据产出 |
| C-B | 四个标尺脚本/报告齐全 |
| C-C | 判定树结论写入 `docs/wpc_ceiling_test_report.md` |
| C-D | P1b 产出 Phase 4R 候选 backbone 池（5 seeds checkpoint） |

### 5.6 预估资源

| 资源 | 量 |
|------|-----|
| 人力 | ~1.5 天（含分析） |
| GPU | ~3 h |

### 5.7 阶段结果（实施后填写）

> 完成时间：____  
> 判定树结论：____  
> P1b 5-seed test top1 mean ± sd：____  
> Phase 4R backbone（val BCE 中位 seed）：____

---

## 6. Phase 4R：Track A 机制分析

> **科学目标**：回答 Q2——全特征网络用什么内部机制实现（逼近）NULL-0。  
> **前提**：WP-C 判定 Q1 为零结果（或异常审计完成）。WP-F §5.3 完成。

### 6.1 分析对象

| 项 | 规定 |
|----|------|
| 配置 | C_cross=2, L_self=2, sequential（与 Phase 3b 冠军同构，仅为连续性） |
| 数据 | v3 |
| 训练 | WP-C 标尺一 P1b 已完成的 5 seeds |
| **选用 checkpoint** | 5 seeds 中 **val BCE 中位数** 的 seed |
| **禁止** | `results/phase3_depth/stageII/B_C2_L2/best.pt` |

在天花板假说下，任何已收敛配置等价；选 (2,2) 不构成「该深度更优」背书。

### 6.2 实验清单

| ID | 实验 | 方法 | 预注册预期 | 产出 |
|----|------|------|------------|------|
| 4R.1 | Logit 对齐 | final_logit vs NULL-0 逐样本回归 | 幸存候选 R² > 0.95 | `analysis/logit_align.py` |
| 4R.2 | 筛法实现定位 | zero-ablate 模数特征/token | 错误率增幅 ≈ 1/r 排序 | `analysis/modulus_ablation.py` |
| 4R.3 | Head–modulus 对齐 | cross-attn + head ablation | ablation 因果背书，不单靠 attention map | `analysis/head_modulus.py` |
| 4R.4 | 尺度校准电路 | ln N 线性探针 + patching | patching 平移基率不改排序 | `analysis/scale_probe.py` |
| 4R.5 | Self-attn 必要性 | L_self=0 vs 1，5 seeds | 无显著差异 | 复用 `train.py` |
| 4R.6 | 聚类复核 | prime-prime attention vs residue class | 若 4R.5 冗余，ARI 跨 seed 低 | `analysis/cluster_residue.py` |

**因果证据等级**：每项标注 observational / ablation / patching。

### 6.3 代码结构（新建）

```text
analysis/
  __init__.py
  logit_align.py
  modulus_ablation.py
  head_modulus.py
  scale_probe.py
  cluster_residue.py
  common.py              # 加载 backbone、NULL-0、metrics 复用
```

### 6.4 验收标准

| ID | 标准 |
|----|------|
| 4R-A | `docs/phase4r_mechanism_report.md` 六项实验均有结果节 |
| 4R-B | 4R.3/4R.4 含 patching 或 ablation 因果证据 |
| 4R-C | Null-anchored Δ 验证 P5（排序贡献 < 最小效应量） |

### 6.5 预估资源

| 资源 | 量 |
|------|-----|
| 人力 | ~3 天 |
| GPU | ~4 h |

### 6.6 阶段结果（实施后填写）

> 完成时间：____  
> 4R.1 R²：____  
> 4R.5 结论：____

---

## 7. Phase 5：Track B 原始比特机制发现

> **科学目标**：回答 Q3——仅给原始比特，网络能否自发形成模运算/筛法电路，按什么顺序涌现。  
> **定位**：本项目对 mech-interp 社区的**主创新点**；可与 Phase 4R 并行（WP-C 判定后）。

### 7.1 任务与架构

| 项 | 规定 |
|----|------|
| 任务 | **逐对二分类**：(N 的 64 bit, 候选 index) → P(N−p_i 素数) |
| N 输入 | 64 bit-position token + 位置编码 |
| p_i 输入 | 1 个 id embedding（**不含** p 的数值特征） |
| 架构 | 1–4 层 Transformer，d_model=128，4 heads |
| 训练 | batch ≥ 4096 pair，AdamW，wd ∈ {0.01, 0.1, 1.0}，3×10⁴–3×10⁵ steps，cosine LR |
| 上限参照 | NULL-0(M = 模型实际学会的 z_eff) |

### 7.2 代码结构（新建）

```text
model/
  bit_transformer.py       # Track B 主模型
  test_bit_transformer.py

train/
  train_bit_pair.py        # 逐对训练入口
  bit_pair_dataset.py      # (N_bits, cand_idx, label) 采样

analysis/
  probe_timeline.py        # 残差流 mod-r 线性探针，每 1k steps
  fourier_bits.py          # bit embedding / 首层 attention 谱分析
  residue_holdout.py       # 留出 (N mod 3, N mod 5) 单元泛化
  activation_patch.py      # 因果 patching
```

### 7.3 监测与分析

| # | 监测项 | 频率 | 预注册预期 |
|---|--------|------|------------|
| 1 | mod-r 线性探针涌现时间线 | 每 1k steps | mod 3 最早；按 1/r 排序；伴随 loss 台阶 |
| 2 | Fourier 谱 | 训练结束 | mod 3 奇偶、mod 5 周期 4 等 |
| 3 | residue-holdout 泛化 | 阶段性 | 规则学习 → 泛化；查表 → 崩塌 |
| 4 | activation patching | 探针显著后 | 输出按筛法逻辑翻转 |
| 5 | 天花板复核 | 最终 | vs NULL-0(z_eff)，不超解析值 |

### 7.4 风险与降级

| 风险 | 缓解 |
|------|------|
| 主任务太难 | 辅助任务：先训 (N−p_i) mod 3；或 N 缩至 32 bit |
| 无 grokking | wd × batch × steps 小网格；时间线仍记录缓坡 |
| 探针假阳性 | patching 背书；随机初始化探针基线 |

### 7.5 训练命令（草案）

```powershell
python train\train_bit_pair.py --depth 2 --d-model 128 --heads 4 `
  --batch-size 4096 --weight-decay 0.1 --steps 100000 `
  --dataset v3 --probe-every 1000 `
  --output-dir results\phase5\depth2_wd0.1
```

### 7.6 验收标准

| ID | 标准 |
|----|------|
| 5-A | `docs/phase5_trackB_report.md` 含涌现时间线、Fourier、patching、holdout、z_eff 曲线 |
| 5-B | 主图：各 mod 涌现时刻 vs 1/r |
| 5-C | 至少一个 mod 探针结论有 patching 因果背书 |

### 7.7 预估资源

| 资源 | 量 |
|------|-----|
| 人力 | 1–2 周 |
| GPU | 20–60 h（可夜间排队） |

### 7.8 阶段结果（实施后填写）

> 完成时间：____  
> 最优 wd / depth：____  
> z_eff：____

---

## 8. Phase 6：汇总与写作

### 8.1 两条论文线

| 论文 | 内容 | 主图 | 启动条件 |
|------|------|------|----------|
| **一：预注册伪随机性检验** | NULL-0 + P0–P5 + M-scaling | top1 vs ln(r_M)；森林图 | WP-C 判定树走通 |
| **二：筛法电路自发形成** | Track B 涌现 + Track A 对照 | 涌现时刻 vs 1/r；Fourier | Phase 5 主扫描完成 |

### 8.2 交付清单

| 文件 | 说明 |
|------|------|
| `docs/wpc_ceiling_test_report.md` | WP-C 完整报告 |
| `docs/phase4r_mechanism_report.md` | Track A 机制 |
| `docs/phase5_trackB_report.md` | Track B 机制 |
| `docs/null_model_verification.md` | NULL-0 验证 |
| `docs/preregistration_wpc.md` | 预注册存档 |

### 8.3 验收标准

- [ ] 三份实验报告「阶段结果」节无占位符；
- [ ] 论文一图表可从 `docs/figures/wpc_*.png` 直接引用；
- [ ] 全部 test 评估日志可审计（附录 C 第 8 项）。

---

## 9. 推荐目录结构（完成后）

```text
goldbach/
├── docs/
│   ├── igot_experiment_plan_v2.1.md          # 科学总方案
│   ├── igot_implementation_plan_v2.md        # 本文件
│   ├── preregistration_wpc.md                # 预注册（WP-C 前 commit）
│   ├── null_model_verification.md
│   ├── wpc_ceiling_test_report.md
│   ├── phase4r_mechanism_report.md
│   ├── phase5_trackB_report.md
│   └── figures/
│       ├── wpc_forest_plot.png
│       ├── wpc_mscaling_top1_vs_ln_rM.png
│       └── wpc_phase3_reanalysis.png
├── model/
│   ├── null_model.py                         # WP-N
│   ├── igot_null_anchored.py                 # WP-F
│   └── bit_transformer.py                    # Phase 5
├── analysis/                                 # Phase 4R + 5
├── tools/
│   ├── eval_null_model.py
│   ├── eval_test_locked.py
│   ├── stats.py
│   ├── shuffle_survivor_test.py
│   └── reanalyze_phase3_depth.py
├── data/
│   ├── goldbach_train_v3_train.bin
│   ├── goldbach_train_v3_val.bin
│   ├── goldbach_train_v3_test.bin            # 上锁
│   └── goldbach_manifest_v3.json
└── results/
    ├── null/
    ├── wpc/
    ├── wpf_regress/
    └── phase5/
```

---

## 10. 资源总表

| 阶段 | 人力 | GPU (4090) | 依赖 |
|------|------|------------|------|
| WP-N | 2 天 | 0 | — |
| WP-F | 1.5 天 | ~2 h | 可与 WP-N 并行 |
| WP-C | 1.5 天 | ~3 h | G1, G2, F-1/F-2 |
| Phase 4R | 3 天 | ~4 h | G3, G4, G5 |
| Phase 5 | 1–2 周 | 20–60 h | G4 |
| Phase 6 | ~1 周 | 0 | 4R 或 5 部分完成 |
| **合计** | **~3 周** | **< 70 h** | |

---

## 11. 未决问题（执行中决定）

| # | 问题 | 决策时机 |
|---|------|----------|
| 1 | M-scaling 是否推到 M=9592 | ms-1229 与预测吻合度 |
| 2 | Track B 是否加 RoPE | wd 网格首轮结果 |
| 3 | 论文一是否含 residue-holdout OOD 变体 | 篇幅需要 |
| 4 | test 是否从 500k 降至 200k | v3 生成吞吐实测 |

---

## 12. 附录：预注册声明模板

> **提交时间**：____（git commit: ____）  
> **声明**：以下预测在对应实验产生任何数据之前写定，事后不修改。

完整命题列表见 `igot_experiment_plan_v2.1.md` 附录 B（P0–P5）。开跑 WP-C 前将填妥版本提交为 `docs/preregistration_wpc.md`。

---

## 13. 附录：泄漏审计清单

WP-N 完成时执行 1–4；WP-C 异常时执行全部 1–8。详见 `igot_experiment_plan_v2.1.md` 附录 C。

---

## 14. 与旧方案的关系

| 旧方案 (`igot_implementation_plan.md`) | 本方案 |
|----------------------------------------|--------|
| Phase 4 直接 head ablation / logit 三分解 | **废止**；改为 WP-C → 4R → 5 |
| val top1 选 checkpoint | **废止**；改 val BCE |
| `p3-B_C2_L2/best.pt` 为 Phase 4 backbone | **废止**；改 P1b 5-seed 中位 BCE |
| 99,999 样本 / 9,999 val | 升级为 v3 1M 划分 |
| 寻找筛法外高阶 pattern | 重构为 Q1 预注册检验 + Q2/Q3 机制发现 |
