# WP-F · F-1c 实验报告：Modulus 淘汰统计 eligible-only 聚合

> **实验日期**：2026-06-11  
> **前置**：F-1b self-attn / cross-attn 封堵  
> **阻塞项**：m ≤ 0 取模审计**有条件豁免**在本项验证通过后方可落账

---

## 1. 问题

`ModulusTokenBuilder` 的淘汰统计原先对 **全部 K=500 候选** 聚合 `z` 列：

```python
eliminated_count = sieve_matrix.sum(dim=1)  # 含非法候选的乱码 z
```

非法候选（m < 3）的 PyTorch 负数取模标志会污染 modulus token → cross-attn K/V → 合法候选表征。

**漏网路径**：

```text
非法 z 列 → 淘汰统计 → modulus token → cross-attn K/V → 合法 prime 表征
```

---

## 2. 实现

| 文件 | 改动 |
|------|------|
| `model/tokens.py` | `compute_elimination_stats(z, eligibility)`：仅在 eligible 候选上 sum |
| `model/tokens.py` | `ModulusTokenBuilder.forward` 新增 `eligibility_mask` 参数 |
| `model/igot.py` | 传入 `prime_eligibility` |

### 跨候选聚合审计（专家要求核对）

| 模块 | 是否跨候选聚合 | 结论 |
|------|----------------|------|
| `NumberEncoder` | 否，仅依赖 N | 干净 |
| `PrimeTokenBuilder` | 否，每候选独立行 | 干净 |
| `ModulusTokenBuilder` 淘汰统计 | **是**（已修 F-1c） | eligible-only |
| `PrimePrimeSelfAttention` | 是，但 F-1b mask | 已封堵 |

---

## 3. 验收

| 测试 | 内容 |
|------|------|
| `test_eligible_only_at_n100` | N=100 时 `compute_elimination_stats` 与手算 eligible-only 一致，且 ≠ 全候选聚合 |
| `test_phase2_uses_bypassed_sieve_matrix` | Phase 2 `sieve_matrix` ≡ `build_sieve_matrix` |
| `test_phase3_uses_bypassed_sieve_matrix` | Phase 3 同上 |

---

## 4. m ≤ 0 取模审计豁免状态

| 封堵层 | 状态 |
|--------|------|
| F-1 出口 logit mask | ✓ |
| F-1b self-attn key-padding | ✓ |
| F-1b cross-attn 行清零 | ✓ |
| **F-1c modulus 淘汰统计** | **✓（本报告）** |

**裁决**：四项齐备后，m ≤ 0 取模审计 **正式豁免关闭**（乱码 z 不再经任何 K/V 路径泄漏）。

---

## 5. F-2 下游 z 版本声明（方案甲，与 F-1c 一并落账）

公开 API：`build_sieve_matrix()` = raw 整除 + F-2 小素数旁路。

- **语义**：`z` 是 **筛杀标志**，不是原始整除事实
- **单一事实源**：`ExplicitSieveModule` 返回的 `sieve_matrix` 供 sieve / residual / prime token / modulus 统计共用
- **已知限制**：m = r_j 候选在可学习模型中仅享受「普通幸存者」待遇，而 NULL-0 给 p̂ = 1（确定性窗口）；Phase 4R 对齐分析时记为表达力限制，非 bug
