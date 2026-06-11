# WP-F · F-2 / F-1b 实验报告

> **实验日期**：2026-06-11  
> **前置**：F-1 eligibility 出口层 mask 已通过专家审核

---

## F-2：m = r_j 小素数边界旁路（方案甲：单一事实源）

### 设计声明

采用 **方案甲**：`build_sieve_matrix()` 在 raw 整除矩阵上应用 F-2 旁路，作为 **唯一** 对外 `z`。

- sieve / residual / prime token / modulus 淘汰统计 **共用同一份** `sieve_outputs["sieve_matrix"]`
- `z` 语义 = **筛杀标志**，非原始整除事实
- m = r_j 候选在可学习模型中为普通幸存者；NULL-0 给 p̂ = 1 属确定性窗口，Phase 4R 不对齐为异常

### 实现

| 文件 | 改动 |
|------|------|
| `model/sieve_baseline.py` | `build_sieve_matrix()` 单一事实源 |
| `generator/python/test_labeling.py` | 断言 modulus 基偏移 `valid_mask = 1` |

### 验收

- `test_small_prime_bypass_clears_sieve_row`
- `test_modulus_base_offsets_are_valid`
- `test_phase2_uses_bypassed_sieve_matrix` / `test_phase3_uses_bypassed_sieve_matrix`

---

## F-1b：Attention 内部 eligibility key-padding mask

### 问题（专家尾巴一）

F-1 仅在 `final_logits` 出口压死非法位；IGOT 中非法 prime token 仍参与 **self-attention** 的 K/V，乱码模特征污染 bucket 1–3 合法候选表征。

### 实现（方案 a，推荐）

| 层 | 处理 |
|----|------|
| `MultiHeadSDPA` | 支持 `attn_mask`（SDPA + manual 路径） |
| `PrimePrimeSelfAttention` | `prime_eligibility`：Q/K 均须 eligible 才参与注意力；输出再乘 mask |
| `PrimeModulusCrossAttention` | cross 后清零 ineligible 行（modulus K/V 仍干净） |
| `igot.py` | 将 `eligibility_mask` 传入 attention stack |

### m≤0 取模审计豁免（有条件 → F-1c 后正式关闭）

| 阶段 | 状态 |
|------|------|
| F-1b 初稿 | 有条件豁免（modulus 淘汰统计路径未验证） |
| **F-1c 完成后** | **正式豁免**，见 `docs/wpf_f1c_report.md` §4 |

### 验收

- 全部 model 单元测试通过（含 IGOT forward / backward / attention weights）

---

## 推进顺序（专家微调）

```text
F-2 + F-1b（本报告）→ F-4 回归重训（一次）→ P0 checks 子项重评
```

F-3（Null-anchored Δ）仍排在 Phase 4R 前，不与本次合批。
