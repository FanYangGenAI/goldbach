# WP-F 修复总账（F-1 / F-1b / F-1c / F-2）

> **文档定位**：WP-F 管线修复部分的关账总账；机制归因（F-3）与回归重训（F-4）见 `wpf_f4_design.md`。  
> **关账日期**：2026-06-11（F-1c 专家审核通过）  
> **关联**：`wpf_f1_eligibility_report.md`、`wpf_f2_f1b_report.md`、`wpf_f1c_report.md`

---

## 1. 修复动机

Phase 1–3 训练与评估管线存在三类结构性缺陷，阻碍公平 null 对照与 Phase 4R 机制归因：

1. 非法候选（`p_i > N - 3`）进入排序与 loss；
2. 小素数边界 `m = r_j` 被筛惩罚；
3. IGOT 内部 attention / modulus 聚合路径泄漏乱码特征。

WP-F 修复部分（F-1 / F-1b / F-1c / F-2）**已全部见底**。

---

## 2. 分项摘要

| 子项 | 内容 | 报告 | 状态 |
|------|------|------|------|
| **F-1** | 出口 logit = −10⁴ + BCE 仅合法位平均 | `wpf_f1_eligibility_report.md` | ✓ |
| **F-1b** | self-attn key-padding；cross 后清零 ineligible 行 | `wpf_f2_f1b_report.md` | ✓ |
| **F-1c** | modulus 淘汰统计 eligible-only 聚合 | `wpf_f1c_report.md` | ✓ |
| **F-2** | `build_sieve_matrix()` 单一事实源 + m=r_j 旁路 | `wpf_f2_f1b_report.md`、`wpf_f1c_report.md` §5 | ✓ |
| **F-3** | Null-anchored Δ 架构 | — | 未开始（Phase 4R 前） |
| **F-4** | 回归重训 + P0 checks 重评 | `wpf_f4_design.md` | 未开始 |

### F-1 关键数字

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| bucket 1 avg_checks（Phase 2 参照） | 24.1 | **1.27**（冒烟） |
| NULL-0 交叉校验 bucket 1 | — | **1.00**（N-2 独立实现） |

---

## 3. m ≤ 0 取模审计：四层封堵台账（正式豁免关闭）

| 层 | 机制 | 子项 |
|----|------|------|
| 1 | 出口 mask | F-1：`apply_eligibility_mask`，logit = −10⁴ |
| 2 | self-attention | F-1b：`prime_eligibility` key-padding，非法 token 不参与 Q/K |
| 3 | cross-attention | F-1b：cross 输出后清零 ineligible prime 行 |
| 4 | modulus 聚合 | F-1c：`compute_elimination_stats(z, eligibility)`，淘汰统计不含非法候选 |

**裁决**（2026-06-11）：四层齐备，乱码 z 不再经任何 K/V 路径泄漏；**m ≤ 0 取模语义审计正式豁免关闭**。  
非法位在出口与 attention 双路径封堵后，原始 PyTorch 负数取模结果不再影响 loss、排序或可学习表征。

### 跨候选聚合排他性审计

| 模块 | 跨候选聚合？ | 结论 |
|------|-------------|------|
| `NumberEncoder` | 否（仅 N） | 干净 |
| `PrimeTokenBuilder` | 否（逐候选行） | 干净 |
| `ModulusTokenBuilder` 淘汰统计 | 是 | **F-1c 已修**（eligible-only） |
| `PrimePrimeSelfAttention` | 是 | **F-1b 已封堵** |

---

## 4. F-2 语义声明与表达力限制（方案甲）

### 单一事实源

```text
build_sieve_matrix() = compute_sieve_matrix(raw) + apply_small_prime_sieve_bypass()
```

`ExplicitSieveModule` 返回的 `sieve_matrix` 供以下消费者**共用同一份 z**：

- 线性筛 logit（Phase 1）
- residual 特征（Phase 2）
- prime token 特征（Phase 3）
- modulus 淘汰统计（Phase 3）

### 语义

- **`z` = 筛杀标志**，不是原始整除事实。
- `compute_sieve_matrix()` 保留为 raw 整除诊断接口；**下游不得直接调用**。

### 表达力限制（Phase 4R 备案）

当 `m = N - p_i ∈ {r_1, …, r_M}` 时：

| 对象 | 行为 |
|------|------|
| **NULL-0** | p̂ = **1**（确定性窗口） |
| **可学习模型** | m=r_j 行 z 旁路后仅享受「普通幸存者」待遇 |

此为**已知表达力限制**，不是 bug。Phase 4R 对齐分析时若见该区间系统性偏离，优先对照 NULL-0 确定性窗口，而非当作模型异常。

---

## 5. 代码与测试索引

| 路径 | 职责 |
|------|------|
| `model/sieve_baseline.py` | eligibility、F-2、`build_sieve_matrix` |
| `model/attention.py` | F-1b attention mask |
| `model/tokens.py` | F-1c `compute_elimination_stats` |
| `model/igot.py` | F-1b/F-1c 接线 |
| `train/train.py` | `_masked_bce_loss` |
| `model/test_sieve_baseline.py` | F-1、F-2 单测 |
| `model/test_tokens.py` | F-1c、下游 z 版本断言 |
| `tools/verify_eligibility_fix.py` | F-1 bucket 1 冒烟 |

**当前单测**：model 包 **25/25** 通过。

---

## 6. 移交事项（WP-F 修复 → 后续工作包）

| # | 移交项 | 责任方 | 截止 |
|---|--------|--------|------|
| 1 | **F-4 回归重训**（见 `wpf_f4_design.md`） | WP-F | v3 就绪后正式批；可先 v2 冒烟 |
| 2 | **P0 checks 子项重评**（checks 项，post F-4） | WP-C | F-4 完成后 |
| 3 | **P1b 冠军复测** | WP-C | **必须在 v3 上**；与 F-4 正式批合并 |
| 4 | **NULL-0 v1.1 确定性窗口交叉验证** | WP-N / WP-C | **M-scaling 标尺跑之前** |
| 5 | **F-3 Null-anchored Δ** | WP-F | Phase 4R 启动前 |

### 移交项 4 说明：NULL-0 v1.1 确定性窗口

当前 NULL-0 已对 `m ∈ {r_1,…,r_M}` 设 p̂ = 1；可学习管线经 F-2 对同行 z 旁路。  
在 **M-scaling 标尺**（WP-C 标尺二）开跑前，须完成一次书面交叉验证：

- NULL-0 在 `m = r_j` 的 p̂ / logit 与标签 `valid_mask` 一致率；
- `build_sieve_matrix` 旁路后 z 行全零与 NULL-0 筛杀语义一致；
- 记录于 `docs/null_model_verification.md` 增补节（或 `null_model_v11_window.md`）。

此项不阻塞 F-4 冒烟，但 **阻塞 M-scaling 正式标尺**。

---

## 7. WP-F 修复部分关账声明

```text
F-1  ✓   F-1b ✓   F-1c ✓   F-2 ✓   →  修复管线关账
F-3  待 Phase 4R 前
F-4  待执行（设计见 wpf_f4_design.md）
```
