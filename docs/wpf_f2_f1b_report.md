# WP-F · F-2 / F-1b 实验报告

> **实验日期**：2026-06-11  
> **前置**：F-1 eligibility 出口层 mask 已通过专家审核

---

## F-2：m = r_j 小素数边界旁路

### 实现

当 `m = N - p_i ∈ {r_1, …, r_M}` 时，该候选为素数，**不得**因 `m % r_j == 0` 被筛惩罚。

| 文件 | 改动 |
|------|------|
| `model/sieve_baseline.py` | `apply_small_prime_sieve_bypass`：将对应 sieve 行置零 |
| `generator/python/test_labeling.py` | 断言 modulus 基偏移 `valid_mask = 1` |

### 验收

- `test_small_prime_bypass_clears_sieve_row`：N=10, p=7, m=3 时 raw z 行非零、bypass 后为零
- `test_modulus_base_offsets_are_valid`：标签生成端 m=r_j 均为 1

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

### m≤0 取模审计关闭方式（专家尾巴二）

选用 **key-padding mask (a)**：非法 token 整体从 self-attention 剔除，乱码 z 特征不再经 K/V 泄漏。

**正式结论**：§5.1 负数取模审计项在出口 + 注意力双路径封堵后 **豁免关闭**；无需另补 m≤0 单测（F-1 已有 N∈{6,8,10,100} 出口层测试）。

### 验收

- 全部 model 单元测试通过（含 IGOT forward / backward / attention weights）

---

## 推进顺序（专家微调）

```text
F-2 + F-1b（本报告）→ F-4 回归重训（一次）→ P0 checks 子项重评
```

F-3（Null-anchored Δ）仍排在 Phase 4R 前，不与本次合批。
