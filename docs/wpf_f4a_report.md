# WP-F · F-4a v2 对照批实验报告

> **实验日期**：2026-06-12  
> **数据**：`goldbach_train_v2_phase0.bin`（val 9,999，seed=7 划分）  
> **训练 seed**：0  
> **历史参照**：`phase1_gpu` / `phase2_bs2048` / `phase3_flash_bs2048` best.pt（masked 指标重评）

---

## 1. 执行摘要

| 判据 | 结果 |
|------|------|
| 4a-A 训练完成无 NaN | **PASS** |
| 4a-B bucket 1 avg_checks ≤ 6 | **PASS**（**1.00**） |
| 4a-C 全桶 topK = 100% | **PASS** |
| 4a-D 报告产出 | **PASS** |
| **4a-E** 大 N 三级规则 | **FAIL**（3 桶硬上限越界） |
| **4a-F** Phase 1 masked BCE | **PASS**（Δ = −0.00029） |

修复后 bucket 1 病理消除；Phase 1 BCE 不劣于历史重评。大 N 逐桶 top1 在 3/36 比较上超过 3×SE（均为稀疏桶），需专家裁定是否归因于 val_bce vs 历史 val_top1 选点差异或抽样噪声。

---

## 2. 历史 checkpoint masked 重评（修复前权重 + 现行评估路径）

| Phase | masked BCE | top1 | 来源 |
|-------|------------|------|------|
| 1 | 0.1605 | 43.15% | `phase1_gpu/best.pt` |
| 2 | 0.1306 | 43.85% | `phase2_bs2048/best.pt` |
| 3 | 0.1266 | 43.45% | `phase3_flash_bs2048/best.pt` |

---

## 3. F-4a 重训（WP-F 修复后，val BCE 选点）

| Phase | masked BCE | top1 | bucket 1 checks | 耗时 |
|-------|------------|------|-----------------|------|
| 1 | 0.1602 | 43.59% | **1.00** | 78 s |
| 2 | 0.1266 | 44.15% | — | 81 s |
| 3 | 0.1264 | 44.15% | — | 1771 s |

产出：`results/wpf_regress_f4a/{phase1,phase2,phase3}/seed_0/`

---

## 4. 4a-E 三级规则详情

- **N_cmp** = 36（bucket 7–18 × 3 phases）
- **硬上限越界**（|Δ| > 3×SE）：**3 桶**
  - phase2 · bucket 17：Δ = +5.67pp
  - phase3 · bucket 7：Δ = −6.21pp
  - phase3 · bucket 10：Δ = +6.77pp
- **软配额**（(2×SE, 3×SE]）：2 / 3（未超限）
- **符号检验** p = 0.618（≥ 0.01，无系统性同向漂移）

完整逐桶表：`results/wpf_regress_f4a/f4a_criteria_report.json`

---

## 4.1 4a-F

| 项 | 值 |
|----|-----|
| 新 Phase 1 masked BCE | 0.1602 |
| 历史重评 BCE | 0.1605 |
| Δ | **−0.00029**（≤ 0.001 ✓） |

---

## 5. 下一步

- **F-4b** 正式批已在 v3 审计 PASS 后启动（§7 钉死 `NULL-0_ref = v1.0`）
- 4a-E 失败项提交专家：是否在 F-4b 对照 NULL-0 框架下视为可接受噪声，或需补充 seed=1 稳健性 / 选点对齐分析
