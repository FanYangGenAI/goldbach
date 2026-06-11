# WP-F · F-1 Eligibility Mask 实验报告

> **实验日期**：2026-06-11  
> **代码入口**：`model/sieve_baseline.py`、`model/sieve_residual.py`、`model/igot.py`、`train/train.py`  
> **冒烟脚本**：`tools/verify_eligibility_fix.py`

---

## 1. 目的

修复 `p_i > N - 3`（即 `m = N - p_i < 3`）的非法候选进入排序前列的问题，消除 bucket 1 `avg_checks` 病理（修复前 Phase 2 参照 **24.1**）。

---

## 2. 实现

```text
p_i > N - 3  ⇒  logit_i = -1e4，BCE loss 中 mask 掉该位置
```

| 模块 | 改动 |
|------|------|
| `sieve_baseline.py` | `compute_eligibility_mask`、`apply_eligibility_mask`；筛法 forward 应用 |
| `sieve_residual.py` | `final_logits` 再 mask |
| `igot.py` | `final_logits` 再 mask（防止 attention/residual 抬高非法位） |
| `train/train.py` | `_masked_bce_loss`：仅在 eligible 位置平均 BCE |

---

## 3. 验收结果

### 3.1 单元测试

`model/test_sieve_baseline.py`：N ∈ {6, 8, 10, 100} 全候选，非法位 logit = −10⁴。

全部 model 测试：**21/21 通过**。

### 3.2 bucket 1 冒烟（未训练 SieveResidualModel，val seed=7）

| 指标 | 修复前（Phase 2 参照） | 修复后（冒烟） |
|------|------------------------|----------------|
| bucket 1 avg_checks | **24.1** | **1.27** |
| bucket 1 样本数 | 11 | 11 |
| 验收（≤ 6） | ✗ | **✓** |

原始 JSON：`results/wpf/eligibility_smoke.json`

> 冒烟使用随机初始化权重，全局 top1 无意义；bucket 1 改善完全来自 eligibility 硬约束，与训练无关。

---

## 4. 结论

F-1 **达成验收**：bucket 1 `avg_checks` 从 ~24 降至 **1.27**（≤ 6）。

**下一步**：F-2（m = r_j 边界旁路）→ F-4 回归重训后重评 P0 checks 子项。
