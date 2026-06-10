# Phase 3 深度实验方案（循序渐进：Attention 深度 + 长训练）

> **文档定位**：在 Phase 3 基线之上，**分阶段**验证 Cross-Attn / Self-Attn 深度与训练预算对排序上限的影响。  
> **状态**：方案已对齐讨论结论，**待代码就绪后按阶段实施**（尚未开跑）。  
> **关联文档**：`docs/phase3_igot_transformer_experiment.md`（基线）、`docs/igot_implementation_plan.md`（总路线图）

---

## 0. 设计原则

### 0.1 循序渐进，方案都要验证

深度实验**不是只做一种架构**，而是按阶段依次验证讨论中的多种方案，前一阶段结论决定后一阶段的超参：

```text
Stage 0  工程就绪（可配置深度、LR warmup+cosine、损失曲线）
    ↓
Stage I  方案 A：Cross=1 固定，Self L=1..6          → 得到 L_self*
    ↓
Stage II 方案 B：在 L_self* 上，Cross C=1..3        → 得到 C_cross*
    ↓
Stage III方案 C：L_cross = L_self = 1..6（绑深）     → 对比「绑深」vs「解耦最优」
    ↓
Stage IV 方案 D：局部网格（可选）                    → 仅在有希望区域细扫
    ↓
汇总     损失曲线写入 md、定稿 Phase 3、衔接 Phase 4
```

**原则**：每阶段**严格串行**跑实验（禁止 benchmark 与 train 并行）；每阶段结束写 **§阶段结果** 再进入下一阶段。

### 0.2 为何 Stage I 先固定 Cross=1？

| 理由 | 说明 |
|------|------|
| 假设优先级 | top5 平台更可能来自 **候选–候选**（Self）而非 **候选–模数**（Cross） |
| 与 Sieve 分工 | Cross 与显式 sieve + token 内模特征重叠较多；Self 是 sieve/Phase2 未覆盖的能力 |
| 控制变量 | 先只动 Self 深度，结论可归因于 L_self |
| 算力 | 6 run vs 全网格 36 run |

**这不代表 Cross 只需一层**。Stage II 专门验证 Cross 深度；Stage III 验证两种深度是否应绑定。四套方案**都会做**。

### 0.3 学习率与 Warmup（当前 vs 目标）

| 项目 | 当前 `train/train.py` | 深度实验（各 Stage 统一） |
|------|------------------------|---------------------------|
| 学习率 | **`--lr` 固定常数**，无 scheduler | **warmup + cosine 衰减** |
| 训练 LR warmup | **无** | **有**（见 §3） |
| Benchmark warmup | 有（cuDNN 计时用） | 保留，与训练无关 |
| 梯度裁剪 | `max_norm=1.0` | 保留 |

历史 Phase 1/2/3 均为**固定 LR**；深度实验一律改用动态 LR，并在 `train_report.json` 记录每 epoch 的 `learning_rate`。

---

## 1. 实验动机（基线回顾）

Phase 3 基线（1×cross + 1×self，10 epoch，固定 lr）：

- top1 相对 Phase 2 约 **+0.8pp**；top5 **未突破 ~87%**
- 10 epoch **不足以判断收敛**；1 层 self **不足以排除深度不足**

要回答：

1. **加深 Self** 能否抬高 top5/top1？（Stage I）  
2. 在最佳 Self 深度下，**加深 Cross** 是否还有收益？（Stage II）  
3. **Cross 与 Self 同深绑定** 是否优于「解耦最优组合」？（Stage III）  
4. 在有望区域是否需要 **二维细扫**？（Stage IV，可选）

---

## 2. 两种 Attention 的语义（实验对照）

| 模块 | 交互 | 复杂度 | 主要假设 |
|------|------|--------|----------|
| **Cross-Attn** | 候选 → 模数 | O(K·M)，M=32 | 模数约束的软版本；与 sieve 部分重叠 |
| **Self-Attn** | 候选 → 候选 | O(K²)，K=500 | 竞争、替代、等价类；**top5 瓶颈主假设** |

通用堆叠模板（解耦深度，Stage I / II 使用）：

```text
prime_tokens, modulus_tokens = build_tokens(...)
for c in range(C_cross):
    prime_tokens = cross_attn(prime_tokens, modulus_tokens)
for s in range(L_self):
    prime_tokens = self_attn(prime_tokens)
attention_logits = attention_head(prime_tokens)
residual_logits  = residual_head(prime_tokens)
final_logits = sieve_logits + attention_logits + residual_logits
```

绑定深度（Stage III）：令 `C_cross = L_self = L`，两种循环各执行 L 次（先 Cross 块再 Self 块，或按块交替——**实施时默认：L 次 `(cross → self)` 单元**，见 §5.3）。

---

## 3. 训练策略（各 Stage 共用）

### 3.1 LR：Warmup + Cosine

```text
1. Linear warmup：前 W 步，lr 从 0 → base_lr
2. Cosine decay：warmup 后至训练结束，lr 从 base_lr → eta_min（默认 base_lr × 0.01）
3. 每 optimizer step 更新 scheduler；梯度裁剪 max_norm=1.0
```

| 超参 | 建议 |
|------|------|
| `warmup_steps` | **max(500, 1 × steps_per_epoch)** |
| `base_lr` | 随 batch 调整（见各 Stage 表） |
| `epochs` | **50**（主训练）；若 val 指标仍缓降可增至 100 |
| 早停 | 可选：`patience=10`，监控 val `top1_valid_rate` |

### 3.2 需在 `train.py` 新增的 CLI

```text
--cross-attn-layers INT    # C_cross，默认 1
--self-attn-layers INT     # L_self，默认 1
--warmup-steps INT
--lr-schedule {none,cosine}
--eta-min FLOAT
```

`train_report.json` 须含：`cross_attn_layers`、`self_attn_layers`、`lr_schedule`、`warmup_steps`、每 epoch `learning_rate`、完整 `history`（供曲线）。

---

## 4. Stage 0：工程就绪（先于一切实验）

| # | 任务 | 产出 |
|---|------|------|
| 0.1 | `IGOTModel` 支持 `cross_attn_layers`、`self_attn_layers` | `model/igot.py` |
| 0.2 | LR warmup + cosine | `train/train.py` |
| 0.3 | `tools/plot_train_curves.py` | PNG → `docs/figures/` |
| 0.4 | **回溯**基线曲线（不重训） | Phase 1/2/3 基线 md 插图 |
| 0.5 | `tools/smoke_phase3_pipeline.py` 支持 `--cross-layers` / `--self-layers` | 冒烟 |
| 0.6 | 单元测试更新 | `test_igot.py` |

**Stage 0 完成门**：L=1,C=1 的 2-epoch smoke 无 NaN；能从 json 生成曲线图。

---

## 5. Stage I — 方案 A：固定 Cross=1，扫描 Self 深度

### 5.1 目的

隔离 **L_self** 效应；得到 **L_self\***（best val top1，辅以 top5/BCE）。

### 5.2 矩阵

| Run ID | C_cross | L_self | 建议 batch | epochs | 输出目录 |
|--------|---------|--------|------------|--------|----------|
| `p3-A-L1` | 1 | 1 | 2048 | 50 | `results/phase3_depth/stageI/A_L1/` |
| `p3-A-L2` | 1 | 2 | 2048 | 50 | `.../A_L2/` |
| `p3-A-L3` | 1 | 3 | 1024 | 50 | `.../A_L3/` |
| `p3-A-L4` | 1 | 4 | 1024 | 50 | `.../A_L4/` |
| `p3-A-L5` | 1 | 5 | 512 | 50 | `.../A_L5/` |
| `p3-A-L6` | 1 | 6 | 512 | 50 | `.../A_L6/` |

每 run 前：smoke + benchmark 确认 batch（**禁止**猜 batch）。

### 5.3 Stage I 交付物

- 表：`L_self` vs best top1 / top5 / BCE / checks  
- 图：`phase3_stageI_self_depth_curve.png`（深度 vs 指标）  
- 每 run：`phase3_stageI_A_L{k}_train_val_bce.png`  
- **决策**：记录 **L_self\***；若 L≥4 无边际收益，Stage II 仍从 L_self\* 出发（可为 2 或 3）

### 5.4 与基线关系

`phase3_flash_bs2048`（10ep、固定 lr）仅作**浅训练参照**；`p3-A-L1` 为 Stage I 中与基线同架构的**长训练 + 动态 lr** 正式对照。

---

## 6. Stage II — 方案 B：固定 L_self\*，扫描 Cross 深度

### 6.1 目的

在 Stage I 选出的 **L_self\*** 上，验证 **加深 Cross** 是否带来额外 top5/top1（模数–候选软交互是否不足）。

### 6.2 矩阵（示例：若 L_self\*=3）

| Run ID | C_cross | L_self | epochs | 输出目录 |
|--------|---------|--------|--------|----------|
| `p3-B-C1` | 1 | L_self\* | 50 | `results/phase3_depth/stageII/B_C1_L3/` |
| `p3-B-C2` | 2 | L_self\* | 50 | `.../B_C2_L3/` |
| `p3-B-C3` | 3 | L_self\* | 50 | `.../B_C3_L3/` |

- **C_cross 扫描范围**：**1, 2, 3**（若 C=3 仍有收益且显存允许，可加 C=4）  
- batch / base_lr：按 L_self\* 与 C_cross 合计深度 smoke 后确定  

### 6.3 Stage II 交付物

- 表：`C_cross` vs 指标（在固定 L_self\* 下）  
- 图：`phase3_stageII_cross_depth_at_Lselfstar.png`  
- **决策**：记录 **C_cross\***（解耦最优）

### 6.4 进入条件

Stage I 完成且 **L_self\*** 已写入本文档 §阶段结果。

---

## 7. Stage III — 方案 C：Cross 与 Self 绑深（L=1..6）

### 7.1 目的

验证「**同深度堆叠**」是否优于 Stage I/II 的「**解耦最优 (C_cross\*, L_self\*)**」。

### 7.2 堆叠约定

每个深度 L，重复 L 次：

```text
prime_tokens = cross_attn(prime_tokens, modulus_tokens)
prime_tokens = self_attn(prime_tokens)
```

即 L 个 **(cross → self)** 单元（与 Stage I 的「先全部 cross 再全部 self」不同，绑定方案按单元交替更自然）。

### 7.3 矩阵

| Run ID | L（=C_cross=L_self） | 建议 batch | epochs | 输出目录 |
|--------|----------------------|------------|--------|----------|
| `p3-C-L1` | 1 | 2048 | 50 | `results/phase3_depth/stageIII/C_L1/` |
| `p3-C-L2` | 2 | 1024 | 50 | `.../C_L2/` |
| … | … | … | 50 | … |
| `p3-C-L6` | 6 | 512 | 50 | `.../C_L6/` |

### 7.4 Stage III 交付物

- 表：绑深 L vs 指标  
- **对比行**：Stage III best vs Stage II 解耦最优 `(C_cross*, L_self*)`  
- 图：`phase3_stageIII_tied_depth_curve.png`

### 7.5 进入条件

Stage II 完成。

---

## 8. Stage IV — 方案 D：局部二维网格（可选）

### 8.1 目的

**不做 6×6 全网格**。仅在 Stage I–III 显示的 **有希望区域** 做补充实验，例如：

- 若 L_self\*=4 且 C=2 优于 C=1，可试 `(C=2, L=4)` vs `(C=3, L=4)`  
- 若绑深 L=4 接近解耦最优，可在 L∈{3,4,5}、C∈{2,3} 做 **≤6 个点** 的局部网格  

### 8.2 进入条件

Stage III 完成，且 Stage I–III 汇总显示 **解耦 vs 绑深 结论不一致或边界模糊**。

若 Stage II 已明确「C=1 足够」且 Stage III 绑深不优于解耦最优，**可跳过 Stage IV**。

---

## 9. 损失曲线与文档更新规范

### 9.1 每个 run 必须产出

- `train_report.json`（含完整 history）  
- `docs/figures/<run_id>_train_val_bce.png`  
- 可选：`top1_top5_vs_epoch.png`

### 9.2 写入 md 的时机

| 时机 | 更新 |
|------|------|
| Stage 0 后 | Phase 1/2/3 **基线**曲线插入各实验 md §7 |
| 每 Stage 结束后 | 本文件新增 **§阶段结果 — Stage X** |
| 全部完成后 | `phase3_igot_transformer_experiment.md` 深度结论定稿；`igot_implementation_plan.md` §7.5 勾选完成项 |

### 9.3 图片命名

```text
docs/figures/
  phase1_train_val_bce.png
  phase2_train_val_bce.png
  phase3_baseline_train_val_bce.png
  phase3_stageI_A_L{1..6}_train_val_bce.png
  phase3_stageI_self_depth_metrics.png
  phase3_stageII_cross_depth_*.png
  phase3_stageIII_tied_depth_*.png
```

---

## 10. 执行顺序与资源预估

```text
Stage 0（代码）     ~0.5–1 天
Stage I（6 runs）   ~1.5–4 h GPU
Stage II（3 runs）  ~0.5–2 h GPU
Stage III（6 runs） ~2–5 h GPU
Stage IV（≤6 runs） 视需要
```

**铁律**：同一时刻仅 **一个** 训练或 benchmark 进程占用 GPU。

---

## 11. 代码改动清单（Stage 0）

| # | 改动 | 文件 |
|---|------|------|
| 1 | `cross_attn_layers`：堆叠 `PrimeModulusCrossAttention` | `model/igot.py` |
| 2 | `self_attn_layers`：堆叠 `PrimePrimeSelfAttention` | `model/igot.py` |
| 3 | 可选 `stack_mode`：`sequential`（先全 cross 后全 self）vs `paired`（L 次 cross→self） | `model/igot.py` |
| 4 | LR warmup + cosine | `train/train.py` |
| 5 | CLI：`--cross-attn-layers`、`--self-attn-layers`、`--stack-mode` | `train/train.py` |
| 6 | 绘图脚本 | `tools/plot_train_curves.py` |
| 7 | 顺序批跑（按 Stage 分脚本） | `tools/run_phase3_stageI.ps1` 等 |

---

## 12. Phase 3 深度实验总完成标准

- [ ] Stage 0 工程就绪；基线损失曲线已写入 Phase 1/2/3 md  
- [ ] Stage I（方案 A）完成；**L_self\*** 已确定  
- [ ] Stage II（方案 B）完成；**C_cross\*** 已确定  
- [ ] Stage III（方案 C）完成；绑深 vs 解耦最优已对比  
- [ ] Stage IV：已执行或书面说明跳过理由  
- [ ] 全局汇总：最优配置、框架极限结论、Phase 4 backbone 选定  

---

## 13. 阶段结果（实施后填写）

### 13.1 Stage I — 方案 A

_（待填写：L_self\*、表格、曲线链接、文字结论）_

### 13.2 Stage II — 方案 B

_（待填写：C_cross\*、在 L_self\* 下的增益）_

### 13.3 Stage III — 方案 C

_（待填写：绑深最优 L；是否超过解耦最优）_

### 13.4 Stage IV — 方案 D

_（待填写：或「跳过：原因」）_

### 13.5 全局最优配置（Phase 4 backbone）

| 项目 | 定稿值 |
|------|--------|
| C_cross | _待填_ |
| L_self | _待填_ |
| stack_mode | _待填_ |
| checkpoint | _待填_ |

---

## 14. 预览命令（Stage 0 完成后，Stage I 首 run）

```powershell
$env:PYTHONUNBUFFERED=1
python train\train.py --phase phase3 `
  --cross-attn-layers 1 --self-attn-layers 1 --stack-mode sequential `
  --epochs 50 --batch-size 2048 --lr 0.0025 `
  --lr-schedule cosine --warmup-steps 44 `
  --device cuda --output-dir results\phase3_depth\stageI\A_L1 `
  --phase2-report results\phase2_bs2048\train_report.json
```

---

## 15. 与 Phase 4 的衔接

Phase 4 机制分析（head-modulus 对齐、ablation、聚类）统一使用 **§13.5 全局最优配置** 的 checkpoint，而非 Phase 3 基线 10-epoch 权重。
