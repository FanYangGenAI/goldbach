# IGOT 后续实验总方案 v2.1：从架构搜索到预注册检验与机制发现

> **文档定位**：Phase 3 深度实验(Stage I–IV)收尾后的总体实验路线。本方案将项目科学目标重构为三个可证伪的研究问题,并给出每个实验的预注册预测、统计协议、判定树与产出标准。
> **版本**：v2.1(2026-06-11)
> **前置依赖**:`phase3_deep_experiment_plan.md` Stage I–IV **已全部完成**,结果按本方案 §4.4 重新分析

**v2.1 修订记录**(基于 Phase 3 终局数据):

| # | 修订 | 位置 | 动因 |
|---|------|------|------|
| 1 | §4.4 增加收敛过滤(val BCE ≤ 0.131 分组) | §4.4 | 5 个深层 run 出现 BCE 离群(0.137–0.279),源于固定 lr 下深模型优化困难,须与收敛 run 分组分析 |
| 2 | §4.4 增补"排序–校准解离"分析项 | §4.4 | 实测发现 BCE 未收敛的 run(如 `p3-A_L4`,BCE=0.279)top1 仍达 44.28%——排序饱和先于校准收敛,纳入论文一证据链 |
| 3 | P4 正式裁决挂钩 NULL-0 | §4.4、附录 B | 非正式预测带上沿(top1 ≤ 44.5%)被 2 个 run 擦边超出 ≤ 0.13pp,在实测种子方差(同构重复 run 差 0.48pp)与顺序统计量解释范围内;正式判定以 NULL-0 ± 1pp 为准 |
| 4 | 标尺一增加"冠军复测"条目(命题 P1b) | §4.1、附录 B | 全局最优 `p3-B_C2_L2`(top1=44.63%)系约 750 次带噪测量取最大值的产物,须 5 种子复测检验可复现性 |
| 5 | Phase 4R backbone 取用规则收紧 | §6 | 禁止直接使用现有 `best.pt`(自带选择偏差),须取冠军复测中 val BCE 中位数的 checkpoint |

**Phase 3 终局数据摘要**(详见 `phase3_deep_experiment_plan.md` §13):15 个配置(深度 1–12 层、解耦/绑深两种堆叠),best top1 全部落于 43.79%–44.63%(均值 44.15%,标准差 0.26pp);best top5 全部落于 86.45%–87.71%,平台未破;Stage III 绑深 top1 随深度单调微降(44.47% → 43.80%);同构重复 run(`p3-A_L2` vs `p3-B_C1_L2`,同为 C=1/L=2/batch 2048)实测种子方差:Δtop1 = 0.48pp,ΔBCE = 0.096。
> **关联文档**:`phase1_sieve_baseline_experiment.md`、`phase2_sieve_residual_experiment.md`、`phase3_igot_transformer_experiment.md`

---

## 0. 执行摘要

Phase 1–3 的全部实验结果(含深度实验 15 个配置:top1 ∈ [42.9%, 44.6%],top5 ∈ [86.5%, 87.7%],跨三个数量级参数量、1–12 层深度与多种架构)与解析数论的天花板预测高度吻合:**在模余特征给定的条件下,筛法幸存候选的素性近似可交换 Bernoulli 事件,任何架构的排序增益预期为零。**

据此,本方案做三件事:

1. **把"天花板假说"变成可检验的预注册命题**(WP-N + WP-C):实现零参数解析 null model,扩建测试集到统计可判的规模,用 M-scaling、shuffle 检验、预测带四个标尺对假说做系统检验;
2. **修复阻碍 Phase 4 归因分析的三个结构性缺陷**(WP-F):logit 分解退化、residual 的 z 泄漏、小 N 的 eligibility/取模问题;
3. **把机制研究拆为两条有独立发表价值的轨道**(Phase 4R + Phase 5):Track A 研究"网络如何实现已知数论"(筛法 + Hardy–Littlewood 密度的内部实现),Track B 研究"网络能否从原始比特自发形成模运算电路"(grokking 式机制发现,**本项目主创新点**)。

**核心原则:每个实验在执行前写下定量预测;每个结论附带统计检验;零结果(无超额信号)是预期主结局,且本身构成对素数伪随机性的一次系统经验检验。**

---

## 1. 科学问题重构

原方案的核心问题("模型能否发现筛法之外的高阶 pattern")在统计上不可直接操作。重构为三个可证伪问题:

| # | 研究问题 | 可证伪形式 | 预期答案 | 对应工作包 |
|---|---------|-----------|---------|-----------|
| **Q1** | 给定模余特征,候选排序是否存在超出 Hardy–Littlewood 密度的可学习信号? | 任意模型在大测试集上的指标是否显著超过解析 null model(NULL-0) | **否**(超出即为重大异常,需审计) | WP-N, WP-C |
| **Q2** | 全特征神经网络通过什么内部机制实现(或逼近)null model? | 网络 logit 与 NULL-0 logit 的对齐度;消融后的可预测崩塌模式 | 网络 = 筛法电路 + 尺度校准电路 | Phase 4R |
| **Q3** | 仅给原始比特时,网络能否自发形成模运算/筛法电路?形成顺序与结构是什么? | 残差流线性探针、电路分析、留出剩余类泛化 | 能,且按 1/r 效应量排序先后涌现 | Phase 5 |

Q1 的零结果 + Q2/Q3 的机制结论,共同构成两篇独立成文的产出(见 §8)。

---

## 2. 总体路线图

```text
[当前] Phase 3 Stage I–IV 跑完
   │
   ├─ WP-F  模型与数据修复 ──────────┐   (~1.5 天,可与 WP-N 并行)
   │                                  │
   ├─ WP-N  统计基础设施 + Null Model ┤   (~2 天)
   │                                  ▼
   └─ WP-C  天花板预注册检验 ◄────────┘   (~1 天 GPU + 0.5 天分析)
              │
              ├─ 判定树 §4.5:全部落在预测带 → Q1 定稿(零结果)
              │                超带且通过审计 → 异常深挖(§4.6)
              ▼
        Phase 4R  Track A 机制分析(网络如何实现 null)   (~3 天)
              │
              ▼
        Phase 5   Track B 原始比特机制发现(主创新)      (~1–2 周)
              │
              ▼
        Phase 6   汇总与写作
```

**硬性闸门**:Phase 4R 不得在 WP-F 的可辨识性修复(§5.3)完成前启动;WP-C 的预注册声明(附录 B)必须在对应实验产出数据前提交至 repo(git commit 时间戳即为预注册凭证)。

---

## 3. WP-N:统计基础设施与解析 Null Model

### 3.1 数据集重构(v3)

当前 99,999 条样本、9,999 条验证集的配置,top1 标准误 ≈ 0.5pp,无法分辨任何 < 1pp 的效应。重构如下:

| 划分 | 规模 | 用途 | 采样 |
|------|------|------|------|
| train | 450,000 | 训练 | 18 桶 log-uniform(沿用) |
| val | 50,000 | checkpoint 选择、早停 | 同上,与 train 不相交 |
| **test** | **500,000** | **最终报告,全程上锁** | 同上,与 train/val 不相交 |

- **test 集 top1 标准误 ≈ 0.07pp**,可分辨 0.2pp 级效应;
- **去重与划分纪律**:按 **N 的取值** 划分(不是按行),保证同一 N 不跨划分出现;生成后运行去重审计(附录 C 第 1 项);
- **生成预算**:1M 样本 × 500 候选 = 5×10⁸ 次 64-bit 素性测试。纯 Python/sympy 不可行;改用 `gmpy2.is_prime`(每次 ~2–5 μs)+ multiprocessing,8 worker 预计 **3–8 小时**,一夜可完成。若吞吐仍不足,test 降至 200,000(SE ≈ 0.11pp,仍够用);
- **格式**:沿用 v2 的 85 字节结构,新增文件头字段 `split_id`、`dedup_hash`;
- **OOD 切片(标注,不单独生成)**:
  - `scale-holdout` 标志:N ∈ (10¹⁶, 4×10¹⁸] 的 test 子集,用于评估"只在 N ≤ 10¹⁶ 上训练"的外推变体(可选实验);
  - `residue-holdout` 标志:预留 3 个 (N mod 3, N mod 5) 组合单元,训练变体时从 train 中剔除,test 中保留,检验剩余类泛化。

### 3.2 解析 Null Model(NULL-0 / NULL-1)

#### 3.2.1 数学定义

对偶数 N、候选 p_i,记 m = N − p_i,模数基底为前 M 个奇素数 r_1=3, …, r_M(M=32 时 r_M=137):

```text
NULL-0 概率:

p̂_i = 0                                    若 m < 3(不合法候选)
p̂_i = 1                                    若 m ∈ {r_1, ..., r_M}(m 本身是基底小素数)
p̂_i = ε                                    若 ∃j: r_j | m 且 m ≠ r_j(被筛除;ε = 10⁻⁴)
p̂_i = min(0.999, e^γ · ln(r_M) / ln(m))    否则(幸存候选)

其中 e^γ = 1.7810724...(γ 为 Euler–Mascheroni 常数)
```

**理论依据**:N 为偶数、p_i 为奇素数,故 m 必为奇数;"m 与 r_1…r_M 全部互素"等价于"m 与 2·3·…·r_M 互素"。由 Mertens 定理,与 ≤ z 的全部素数互素的整数中素数的密度为 (1/ln m)·(e^{−γ}/ln z)⁻¹ = e^γ ln z / ln m。

**两个重要说明(写入代码注释)**:

1. **奇异级数不需要显式出现**。Hardy–Littlewood 因子 𝔖(N) 决定的是有多少候选能通过筛(幸存者数量),而这一信息已完整包含在筛矩阵 z 中;**条件于幸存,素性概率只依赖 m 的大小,与 N 的小素因子结构无关**。这是 NULL-0 不含 N 项的原因,也是 Phase 1 排序指标与理论吻合的机制。
2. **二阶修正**:Mertens 渐近在 u = ln m / ln r_M 较小时有 Buchstab 修正。本项目 N ≥ 10²、r_M=137 时 u ∈ [1, 9],小 N 端公式偏差可达 10–20%。这只影响 BCE/校准,不影响同一 N 内幸存者的排序(因子共享)。NULL-1 用于吸收它。

#### 3.2.2 NULL-1:经验再校准变体

```text
NULL-1 = isotonic_regression(NULL-0 概率 → 经验频率),仅在 train 上拟合
```

- 排序指标(top1/top5/checks)与 NULL-0 **逐样本完全相同**(单调变换);
- BCE/校准指标取 NULL-1,作为"已知理论 + 完美校准"的下界;
- 实现:`sklearn.isotonic.IsotonicRegression`,一维,拟合开销可忽略。

#### 3.2.3 交付物

| 文件 | 内容 |
|------|------|
| `model/null_model.py` | NULL-0/NULL-1 实现(参考实现见附录 A) |
| `results/null/eval_report.json` | 在 val 与 test 上的全套指标(与 metrics.py 完全同协议) |
| `docs/null_model_verification.md` | NULL-0 指标 vs Phase 1/2/3 全部已有结果的对照表 |

**验收标准(预注册预测,见附录 B 命题 P0)**:在现有 9,999 条验证集上,NULL-0 的 top1/top5/avg_checks 与 Phase 2/3 最优结果之差 ≤ 1pp / 1pp / 0.1;NULL-1 的 BCE ≤ 0.127。若 NULL-0 显著**劣于**已训模型(差 > 2pp),优先排查 NULL-0 实现 bug(尤其负数取模与 m=r_j 边界),其次才考虑模型存在超额信号。

### 3.3 评估协议(全部后续实验强制执行)

| 条款 | 规定 |
|------|------|
| **模型选择** | 以 **val BCE** 选 checkpoint(方差小、近似单调)。排序指标只记录、不参与选择。废止"val top1 最高"标准 |
| **报告纪律** | test 集每个模型只评估一次,在该模型全部训练与选择流程冻结后进行。test 结果不得反馈回任何决策 |
| **多种子** | 进入结论表的每个配置 ≥ 5 seeds,报告 mean ± sd。(单 run 12 分钟,成本可忽略) |
| **显著性** | top1/top5:对 NULL-0 的逐样本配对 McNemar 检验;avg_checks/BCE:逐样本配对 bootstrap(10⁴ 次重采样)95% CI。族错误率:对每张结论表做 Holm 校正,α = 0.01 |
| **最小关注效应量** | top1/top5:0.3pp;avg_checks:0.05;BCE:0.001。小于此值即使显著也判定为"无实际意义" |
| **决策阈值** | 任何"配置 A 优于配置 B"的结论,须同时满足:效应量 ≥ 最小关注值、Holm 校正后 p < 0.01、≥ 5 seeds 方向一致 |
| **预注册** | 每个工作包开跑前,预测表 commit 进 repo(附录 B 为 WP-C 的版本) |

---

## 4. WP-C:天花板预注册检验

本工作包是对 Q1 的正面回答,由四个互相独立的标尺组成。**附录 B 的预测表必须在 4.1–4.4 任何一项产出数据之前提交。**

### 4.1 标尺一:全模型 vs NULL-0 对照

把以下模型在新 test 集上与 NULL-0 逐样本配对比较:

| 模型 | 来源 |
|------|------|
| Phase 1 线性筛(重训于 v3 train) | 既有代码 |
| Phase 2 sieve+residual(重训) | 既有代码 |
| Phase 3 全局最优配置(Stage I–IV 选出的 backbone,重训) | 既有代码 |
| NULL-0 / NULL-1 | WP-N |

**预注册命题 P1**:每个神经网络模型与 NULL-0 在 top1/top5/avg_checks 上的差异 < 最小关注效应量(0.3pp / 0.3pp / 0.05);BCE 与 NULL-1 之差 < 0.002。

**冠军复测(v2.1 新增,命题 P1b)**:Phase 3 全局最优配置 `p3-B_C2_L2`(C=2, L_self=2,原始 best top1 = 44.63%)在 v3 数据上以 §3.3 协议重测:5 seeds、val BCE 选 checkpoint、test 集一次性评估。
**预测**:5 种子 test top1 均值落在 NULL-0 ± 0.3pp 内,44.63% 不可复现(该值为 15 runs × 50 epochs ≈ 750 次带噪测量取最大值的顺序统计量假象)。成本 ~1 h GPU。此项同时产出 Phase 4R 的 backbone checkpoint(见 §6)。

### 4.2 标尺二:M-scaling(信息杠杆的解析可预测性)

天花板理论的最强检验:增加模数个数 M 是唯一已知的信息增量,且收益按 Mertens 因子 ln(r_M) 解析可算。

#### 4.2.1 实验矩阵

| Run | M(奇素数个数) | r_M | q 相对 M=32 倍率 ln(r_M)/ln(137) | 模型 |
|-----|----|------|------|------|
| `ms-32` | 32 | 137 | 1.00 | 线性筛(Phase 1 结构) |
| `ms-64` | 64 | 313 | 1.17 | 同上 |
| `ms-168` | 168 | 997 | 1.40 | 同上 |
| `ms-1229` | 1229 | 9973 | 1.87 | 同上 |
| `ms-1229-null` | 1229 | 9973 | 1.87 | NULL-0(M=1229 版,零参数) |

#### 4.2.2 工程预算

筛矩阵 z 的规模:batch 2048 × K 500 × M 1229 = 1.26×10⁹ bool ≈ 1.26 GB(uint8)。两个缓解方案任选:

1. **分块**:沿 M 维分 8 块流式计算与累加(线性模型只需 Σ w_j z_ij,无需驻留全矩阵);
2. **二参数压缩版**:理论预测全部 w_j 等效(任何一次命中都近乎判死),故可用 `logit = b(scale) − w·[任意命中]` 的幸存指示模型替代,只存每候选 1 bit。建议两版都跑,互为校验。

#### 4.2.3 预注册命题 P2

记 top1(M) 为各 run 的 test top1。**预测:top1(M) 与按 NULL-0 公式(将 r_M 代入)计算的解析值的相对偏差 ≤ 10%**;具体点预测(以现 18 桶混合分布计):

| M | 预测 top1 | 预测 avg_checks |
|---|-----------|------------------|
| 32 | 43% ± 2pp | 3.0 ± 0.2 |
| 64 | 50% ± 3pp | 2.4 ± 0.2 |
| 168 | 59% ± 3pp | 1.9 ± 0.15 |
| 1229 | 73% ± 4pp | 1.5 ± 0.1 |

**含义**:若 P2 成立,一个数千参数的线性模型将在排序指标上全面超越 Phase 3 的全部 attention 配置(top1 73% vs 44%),"性能杠杆是信息而非架构"即被定量钉死;同时 top1 随 ln(r_M) 的标度律本身就是对 Cramér 图景的一次跨 16 个数量级的实验验证——**这张图是论文一的主图**。

### 4.3 标尺三:可交换性 shuffle 检验

**操作**:取 Phase 3 最优模型在 test 上的输出。对每个 N,保持非幸存候选的排序不变,把幸存候选(z 行全零且 eligible)之间的相对排序**随机打乱**(10 次独立 shuffle 取均值)。

**预注册命题 P3**:shuffle 前后 top1 / avg_checks 之差 < 0.3pp / 0.05。

**含义**:这是"幸存者之间无信息可分"的最直接演示——如果训练了 42 万参数的模型对幸存者的排序与随机排序统计等价,排序信息不存在的结论不需要任何理论背书。成本:零训练,纯评估脚本,~30 行。

### 4.4 标尺四:Phase 3 Stage I–IV 结果重分析(v2.1 修订)

Stage I–IV 已跑完(15 runs),**不重训**,做四件事:

1. **收敛过滤**:按 val BCE ≤ 0.131 把 run 分为"已收敛"(10 个)与"未收敛"(5 个:`A_L4`、`A_L5`、`A_L6`、`B_C1_L2`、`C_L4`–`C_L6` 中 BCE 离群者)。未收敛组归因于固定 base_lr 下深模型的优化困难(全部集中于深层 + 小 batch 组合),单独标注,不进入主分析;
2. **主散点图**:已收敛 run 的 (深度, best top1/top5) 散点,叠加噪声带(±2×SE)与 NULL-0 水平线;
3. **深度单调性**:对已收敛组做深度–指标 Spearman 检验。注意 Stage III 原始数据呈**负向**趋势(L=1 → 6 时 top1 从 44.47% 降至 43.80%),预期过滤掉未收敛 run 后趋势减弱——若仍显著为负,结论记为"深度在该任务上是优化负资产",与天花板假说一致且更强;
4. **排序–校准解离图(新增,论文一素材)**:全部 15 个 run 的 (val BCE, best top1) 散点。预期形态:BCE 横跨 0.125–0.28,top1 压在 ~44% 水平线——**排序饱和先于校准收敛**,直接证明排序所需信息(幸存者优先)在训练极早期即已穷尽,后续训练只在打磨概率刻度。

**预注册命题 P4**:已收敛 run 全部落在 NULL-0 ± 1pp 带内;深度–指标 Spearman 相关无显著正向关系。
**判定状态(2026-06-11)**:15 run 实测 top1 ∈ [43.79%, 44.63%](spread 0.84pp,σ = 0.26pp),top5 平台未破;非正式预测带上沿(44.5%)被 2 个 run 擦边超出 ≤ 0.13pp,处于实测种子方差(0.48pp)与顺序统计量(~750 次带噪测量取最大值,预期虚高 ~1pp)的解释范围内。**正式裁决待 WP-N 产出 NULL-0 在旧验证集上的 top1 后执行**——这是 WP-N 列为最高优先级的直接原因。

### 4.5 判定树

```text
P0–P4 全部成立
   → Q1 定稿:在模余特征下未检出超额信号;
     预注册检验通过 → 论文一的实证主体完成;
     进入 Phase 4R(机制定位为"网络如何实现 NULL-0")

任一命题不成立(模型显著优于 NULL-0,或 M-scaling 偏离 >10%)
   → 进入 §4.6 异常处理流程(先审计后庆祝)

NULL-0 显著优于全部模型
   → 训练或特征管线存在缺陷(优先排查 eligibility/取模/收敛);
     修复后重测
```

### 4.6 异常处理流程(若检出超额信号)

按顺序执行,任何一步失败即判定为假阳性:

1. **泄漏审计**:跑附录 C 全部清单(重复 N 跨划分、生成器 RNG 相关性、mask 字节序、负数取模语义);
2. **复现**:换 5 个新 seed + 重新生成一份独立 test 集,效应需双重复现;
3. **定位**:超额增益按桶、按 (N mod 3, N mod 5, N mod 7) 单元、按幸存者数量分层,找出效应集中区;
4. **特征归因**:用 NULL-anchored Δ 模型(§5.3)隔离增益来源特征;
5. **数学化**:若以上全部通过,把效应表述为关于 N、p、m 的显式命题,人工在 10⁶ 量级独立样本上验证,再考虑联系数论同行。
6. **(预期概率说明)**:此分支被触发且通过审计的先验概率极低;若发生,它优先级压倒本方案其余全部内容。

---

## 5. WP-F:模型与数据修复

以下修复均为 Phase 4R 的前置条件,可与 WP-N 并行开发。

### 5.1 Eligibility Mask(确定性,零参数)

```text
p_i > N − 3  ⇒  final_logit_i = −∞(实现为 −10⁴),且训练 loss 中 mask 掉该位置
```

- 修复 bucket 1 的 avg_checks = 24.1 病理(候选 p_i > N 被排入前列);
- 实现位置:`ExplicitSieveModule` 内,作为最外层硬约束;
- 同步审计 `compute_sieve_matrix` 对 m ≤ 0 的行为(PyTorch `%`/`fmod` 对负数的符号语义与 Python 不同),并以单元测试覆盖 N ∈ {6, 8, 10, 100} 的全候选。

### 5.2 m = r_j 边界(小素数即素数)

```text
若 N − p_i ∈ {r_1, ..., r_M}:该候选 valid_mask 应为 1,且不得被筛惩罚
```

- 在筛模块中加显式旁路:`is_small_prime(m) ⇒ 跳过 z 惩罚`;
- 在标签生成端加断言:对全部 m = r_j 的样本核对 valid_mask = 1(若 Phase 0 生成器已正确,此为零改动验证)。

### 5.3 可辨识性修复:Null-anchored Δ 架构(Phase 4R 的 backbone)

废弃"sieve + attention + residual 三个可学习分支联合训练"的分解(已证实不可辨识:Phase 2 中 w 从 −3.8 塌缩至 −0.43;Phase 3 中 attention_head 与 residual_head 读同一 token,二线性头之和等价于一头)。新架构:

```text
final_logit_i = logit(NULL-0_i) + Δ_i(θ)
                ─────冻结,零参数─────   ───唯一可学习部分───
```

| 设计点 | 规定 |
|--------|------|
| 锚 | NULL-0 logit,**不参与训练**。由构造保证:Δ ≡ 已知理论之外的全部内容 |
| Δ 分支输入 | 完整特征(bits、mods、z、token、attention 栈均可)——因为锚已冻结,Δ 重新实现筛法不再有梯度收益,泄漏问题由训练目标而非特征裁剪解决 |
| Δ 的正则 | 对 Δ 加 L2(λ 扫 {0, 10⁻⁴, 10⁻³}),倾向 Δ → 0 的解;λ=0 组用于检验正则是否掩盖了真实信号 |
| 归因语义 | E[Δ²]、Δ 对各指标的边际贡献,即为"超出 Hardy–Littlewood 的信号总量"的直接度量 |

**预注册命题 P5**:Δ 分支在 test 上对排序指标的边际贡献 < 最小关注效应量;对 BCE 的贡献 < NULL-1 − NULL-0 的校准差(即 Δ 学到的只是 Buchstab 二阶修正)。

> 若仍需保留三分解形态供 Phase 4R 消融,采用 **stage-wise 冻结**:先训纯筛(或直接用 NULL-0),冻结;再训 attention 分支,冻结;最后训 residual。各阶段产物在 test 上的嵌套似然差即为各分支贡献,**禁止再用联合训练模型的内部 logit 拆分做归因**。

### 5.4 回归测试

WP-F 完成后,在 v3 数据上重训 Phase 1/2/3 三个代表模型各 5 seeds,确认:bucket 1 avg_checks 降至 ≤ 6;全桶 topK 维持 100%;其余指标与修复前差异在噪声带内(修复不应改变大 N 行为)。

---

## 6. Phase 4R:Track A 机制分析——网络如何实现 Null Model

**重新定义的目标**:不再寻找"residual insight"(Q1 已由 WP-C 回答),而是回答 Q2——一个全特征 Transformer 用什么电路实现筛法与密度校准。这是"神经网络重新发现经典算法"类型的机制研究,零结果不存在(机制总是存在,问题只是它是什么)。

**分析对象(v2.1 收紧)**:WP-F 修复后、v3 数据上按 §4.1 冠军复测(P1b)重训的 **(C_cross=2, L_self=2)** 配置,取 5 seeds 中 **val BCE 中位数**的 checkpoint。**禁止直接使用 `results/phase3_depth/stageII/B_C2_L2/best.pt`**——该权重经由"50 epoch 内取 val top1 最大"双重选择产生,自带向上偏差,且训练于未修复 WP-F 缺陷的管线,会系统性污染 4R.1 的 logit 对齐分析与 4R.2 的小 N 桶错误率统计。配置沿用 (2,2) 仅出于与既有决策的连续性;在天花板假说下任何已收敛配置等价,此选择不构成"该深度更优"的背书。

| # | 实验 | 方法 | 预注册预期 |
|---|------|------|-----------|
| 4R.1 | **Logit 对齐** | 网络 final_logit 与 NULL-0 logit 的逐样本回归 | 幸存候选上 R² > 0.95;偏离集中在小 N 与 Buchstab 修正区 |
| 4R.2 | **筛法实现定位** | 逐个 zero-ablate 模数特征/modulus token,测 mod-r 特异错误率 | 删除 M_r 后,仅 r 整除相关错误上升,幅度 ≈ 1/r 排序 |
| 4R.3 | **Head–modulus 对齐** | cross-attn 权重分布 + head ablation 因果验证 | 存在专用或组合 head;对齐结论必须以 ablation 后 Δmod-r 错误率背书,attention map 单独不作为证据 |
| 4R.4 | **尺度校准电路** | 在残差流上对 ln N 做线性探针;patch 不同 N 的 scale 特征 | 存在单方向编码 ln N,patching 它平移输出基率而不改排序 |
| 4R.5 | **Self-attention 必要性** | L_self = 0 vs 1 的配对对比(5 seeds, test) | 无显著差异(与 Q1 一致:候选间交互无信息) |
| 4R.6 | **聚类复核** | prime-prime attention 聚类 vs residue class 对齐 | 若 4R.5 判定 self-attn 冗余,聚类结构预期为训练噪声;以 seed 间聚类一致性(ARI)检验 |

交付:`docs/phase4r_mechanism_report.md`,每项实验一节,含因果证据等级标注(observational / ablation / patching)。

---

## 7. Phase 5:Track B 原始比特机制发现(主创新点)

### 7.1 动机与定位

Track A 中模余特征是喂给模型的,"模型使用模结构"是同义反复。Track B 撤掉全部数论特征,只给原始比特,研究问题变为真问题:**梯度下降能否、以何种顺序、以何种内部表示,从二进制表示中自发构造出模运算电路与筛法?** 这是与 grokking / modular-arithmetic 文献(Power et al.; Nanda et al.)直接对话的实验,也是本项目对 mech-interp 社区的主要贡献面。

### 7.2 任务与架构

| 项目 | 规定 |
|------|------|
| 任务形式 | **逐对二分类**:输入 (N 的 64 bit, 候选 p_i 的 index),输出 P(N − p_i 为素数)。放弃 500 路并行输出,使电路分析对象最小化 |
| 输入表示 | N:64 个 bit-position token(每位一个 token,含位置编码);p_i:1 个候选 token(id embedding,**不含** p 的数值特征) |
| 架构 | 1–4 层 Transformer,d_model 128,4 heads;深度作为扫描维 |
| 训练 | 大 batch(≥ 4096 pair)、AdamW、**weight decay ∈ {0.01, 0.1, 1.0} 扫描**(grokking 对 wd 敏感)、3×10⁴–3×10⁵ steps、cosine LR |
| 性能上限 | **NULL-0(M→该模型实际学会的最大模数)**;模型 BCE 收敛值反推其内部筛深度 z_eff = exp(BCE 反解),作为单一标量进度指标 |

### 7.3 监测与分析(训练全程)

1. **涌现时间线(progress measures)**:每 1k steps,在残差流各层对 N mod r(r = 3,5,7,…,31)做线性探针,记录每个模数电路的可解码时刻。**预注册预期:按效应量 1/r 排序,mod 3 最早涌现,且涌现伴随 val loss 台阶下降**;
2. **Fourier 结构**:对 bit-position embedding 与首层 attention 做谱分析。理论预期:2^k mod r 的周期模式(mod 3 → 位置奇偶交替;mod 5 → 周期 4;mod 7 → 周期 3)应出现在权重中;
3. **留出剩余类泛化**:训练时剔除若干 (N mod 3, N mod 5) 单元(§3.1 residue-holdout),测试模型在未见单元上的表现。规则学习 → 泛化;查表 → 崩塌;
4. **因果验证**:activation patching——把"N mod 3 = 0"样本的探针方向 patch 进"N mod 3 = 1"样本,预期输出按筛法逻辑翻转;
5. **天花板复核**:最终模型 vs NULL-0(以 z_eff 为基底)逐样本配对——Track B 模型同样不应超过其学会的筛深度对应的解析值(Q1 在原始比特域的复测)。

### 7.4 风险与降级方案

| 风险 | 缓解 |
|------|------|
| 主任务太难,电路不涌现 | 课程降级:先训辅助任务"预测 (N−p_i) mod 3 = 0"单模数版,确认可学后逐步合并;或缩小 N 范围至 32 bit |
| grokking 不出现(直接泛化或始终欠拟合) | wd / batch / steps 三维小网格;涌现时间线在非 grokking 形态下同样有效(只是台阶变缓坡) |
| 探针假阳性 | 全部探针结论须 activation patching 背书;探针基线:对随机初始化模型跑同一探针 |

### 7.5 交付

`docs/phase5_trackB_report.md`:涌现时间线图(主图)、Fourier 谱、patching 因果表、residue-holdout 泛化表、z_eff 学习曲线。

---

## 8. 产出与写作计划

两条独立成文的线,**任一 Q1 结局下都成立**:

| 论文 | 内容 | 主图 | 目标场合 |
|------|------|------|---------|
| **一:预注册的伪随机性检验** | NULL-0 方法论 + P0–P5 检验结果 + M-scaling 标度律(10² 到 4×10¹⁸ 跨 16 个数量级) | top1 vs ln(r_M) 标度图;全模型 vs NULL-0 配对差异森林图 | Experimental Mathematics / NeurIPS MATH-AI workshop |
| **二:筛法电路的自发形成** | Track B 涌现时间线 + 电路逆向(辅以 Track A 的全特征对照) | 各模数电路涌现时刻 vs 1/r;Fourier 谱 | mech-interp 方向(ICLR/NeurIPS 主会或 workshop) |

写作启动条件:论文一在 WP-C 判定树走通后即可动笔(Phase 4R/5 并行);论文二在 Phase 5 主扫描完成后。

---

## 9. 资源预估

| 工作包 | 人力 | GPU(4090) |
|--------|------|-----------|
| WP-N(数据 v3 + null model + 协议) | 2 天 | 0(数据生成走 CPU,一夜) |
| WP-F(三项修复 + 回归) | 1.5 天 | ~2 h(回归重训 15 runs) |
| WP-C(四个标尺) | 1 天 | ~3 h(M-scaling 5 runs × 5 seeds 为主) |
| Phase 4R | 3 天 | ~4 h |
| Phase 5 | 1–2 周 | ~20–60 h(长训练 + 三维小网格,可夜间排队) |
| **合计** | **~3 周** | **< 70 h** |

铁律沿用:GPU 串行,单进程;每 run 产出 train_report.json + 曲线 PNG;每工作包结束更新对应 md 的"阶段结果"节,**模板占位符未填写完毕不得标记完成**(修复 Phase 3 文档中 L_self* = 0/1 矛盾、§13.5 提前定稿一类问题)。

---

## 10. 未决问题(执行中决定)

1. M-scaling 是否值得推到 M = 9592(r_M ≈ 10⁵,倍率 2.34)?——视 ms-1229 与预测的吻合度;
2. Track B 是否需要 RoPE/特殊位置编码以利于周期模式表达?——以 wd 网格首轮结果定;
3. 论文一是否纳入 residue-holdout 的 OOD 训练变体?——视篇幅,非核心证据链。

---

## 附录 A:NULL-0 参考实现

```python
# model/null_model.py
import math
import numpy as np

E_GAMMA = 1.7810724179901979  # exp(Euler–Mascheroni)

class NullModel:
    """零参数 Hardy–Littlewood / Mertens 解析预测器 (NULL-0)。

    p_hat(m) 含义: P(m 为素数 | m 与 2, r_1..r_M 全部互素), m = N - p_i。
    奇异级数 S(N) 无需显式出现: 它只决定幸存者数量, 该信息已在筛指示中。
    """

    def __init__(self, candidate_primes, modulus_bases, eps=1e-4):
        self.cands = np.asarray(candidate_primes, dtype=np.int64)   # [K]
        self.mods = np.asarray(modulus_bases, dtype=np.int64)       # [M]
        self.mod_set = set(int(r) for r in self.mods)
        self.eps = eps
        self.log_zM = math.log(float(self.mods[-1]))                # ln r_M

    def predict_proba(self, N):
        """N: python int (任意精度)。返回 [K] float64 概率。"""
        m = N - self.cands                                          # [K], 可为负
        p = np.empty(len(self.cands), dtype=np.float64)

        ineligible = m < 3                                          # 5.1 eligibility
        p[ineligible] = 0.0

        for i in np.flatnonzero(~ineligible):
            mi = int(m[i])
            if mi in self.mod_set:                                  # 5.2 m = r_j 边界
                p[i] = 1.0
                continue
            # 注意: 用 python int 取模, 规避张量负数取模语义问题
            if any(mi % int(r) == 0 for r in self.mods):
                p[i] = self.eps                                     # 被筛除
            else:
                p[i] = min(0.999, E_GAMMA * self.log_zM / math.log(mi))
        return p

    def predict_logits(self, N):
        p = np.clip(self.predict_proba(N), 1e-9, 1 - 1e-9)
        return np.log(p / (1.0 - p))


# NULL-1: 仅在 train 上拟合的一维保序回归再校准
# from sklearn.isotonic import IsotonicRegression
# iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
# iso.fit(null0_probs_train, labels_train)   # 展平为逐候选样本
# null1_probs = iso.predict(null0_probs_eval)
```

接入方式:把 `predict_logits` 的输出送入既有 `train/metrics.py` 的同一评估函数,保证与神经网络模型协议逐字节一致。

---

## 附录 B:WP-C 预注册声明(开跑前 commit 本节)

> 提交时间:____(git commit hash:____)
> 声明:以下预测在对应实验产生任何数据之前写定,事后不修改;偏离预测带的结果按 §4.6 流程处理。

- **P0**(NULL-0 对齐既有结果):在 Phase 1–3 的 9,999 条旧验证集上,NULL-0 与已训最优模型的 top1/top5/avg_checks 差 ≤ 1pp / 1pp / 0.1;NULL-1 BCE ≤ 0.127。
- **P1**(无超额信号):新 test 集上,任一神经网络模型相对 NULL-0 的 top1/top5/avg_checks 优势 < 0.3pp / 0.3pp / 0.05,且相对 NULL-1 的 BCE 优势 < 0.002(Holm 校正,α=0.01)。
- **P1b**(冠军不可复现,v2.1 新增):`p3-B_C2_L2` 配置经 5 seeds、val BCE 选点、test 一次性评估后,test top1 均值落在 NULL-0 ± 0.3pp 内;原始 44.63% 判定为顺序统计量假象。
- **P2**(M-scaling 标度律):top1(M) 与解析预测的相对偏差 ≤ 10%;点预测:M=64 → 50%±3pp,M=168 → 59%±3pp,M=1229 → 73%±4pp。
- **P3**(可交换性):幸存候选随机重排后,top1 / avg_checks 变化 < 0.3pp / 0.05。
- **P4**(深度无关性,v2.1 修订):Phase 3 Stage I–IV 中已收敛 run(val BCE ≤ 0.131)全部落在 NULL-0 ± 1pp;深度–指标无显著正向 Spearman 相关。*判定状态:15 run 数据已产出(top1 ∈ [43.79%, 44.63%],σ = 0.26pp),裁决待 NULL-0 计算后执行;本预测带写定于 Stage I 仅 1 run 完成时(见对话记录/commit 历史)。*
- **P5**(Δ ≈ 0):Null-anchored Δ 分支对排序指标的边际贡献 < 最小关注效应量;对 BCE 的贡献不超过 NULL-1 − NULL-0 的校准差。

---

## 附录 C:泄漏与缺陷审计清单(异常触发时全跑;WP-N 完成时跑 1–4)

1. **划分泄漏**:同一 N 值是否出现在多于一个划分?(按值 hash 全量比对)
2. **重复样本**:同一划分内 N 是否重复?重复是否来自枚举桶与抽样桶的交叠?
3. **负数取模**:`compute_sieve_matrix` 对 m ≤ 0 的输出与 python `%` 语义逐位核对(N ∈ {6,8,10,100} 全候选单元测试)
4. **m = r_j 标签**:全部 m ∈ {3,...,r_M} 的样本 valid_mask = 1 断言
5. **生成器 RNG**:train/test 的 N 采样是否共享 RNG 流(序列相关 → 分布泄漏)?各划分独立播种
6. **bitset 字节序**:valid_mask 打包/解包 roundtrip 测试(全 0、全 1、交替、随机 1000 例)
7. **评估对齐**:NULL-0 与神经网络模型是否使用同一 metrics 代码路径、同一 tie-breaking 规则(logit 相等时按候选 index)?
8. **best checkpoint 路径**:报告中的 test 指标是否确由冻结后的单次评估产生(核对评估日志时间戳 > 训练结束时间戳)?
