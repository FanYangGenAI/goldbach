# Interpretable Goldbach Offset Transformer (IGOT)

## 1. 项目定位

本项目的目标不是用神经网络直接证明哥德巴赫猜想，也不是训练一个单纯过拟合的 `N -> (p, q)` 映射模型，而是构建一个可解释的神经网络框架，用来研究：

> 当模型能够稳定判断哪些小素数 `p_i` 使得 `N - p_i` 为素数时，模型内部是否自然形成了类似筛法、模空间几何、候选素数交互、傅里叶周期结构等可解释机制。

因此，本项目的核心不是“预测答案”，而是：

> 训练一个神经网络数论探针，并通过逆向工程分析它学到了什么。

我们将模型定义为一个 **Goldbach Offset Predictor / Candidate Ranker**：

```text
Input:  偶数 N，以及候选小素数集合 {p_1, p_2, ..., p_K}
Output: 每个候选 p_i 是否满足 N - p_i 为素数的 logit / probability
Verifier: 使用确定性素性测试对模型输出进行最终验证
```

其中候选集合初始可设为前 500 个奇素数：

```text
p_i ∈ {3, 5, 7, 11, ..., p_500}
```

模型最终输出为一个 500 维 logits 向量：

```text
logits[i] = score(N - p_i is prime)
```

这个设计保留了原始方案中的“小素数词表 + verifier 闭环”优势，同时将训练目标从单一分类改为多标签判断，更适合解释模型内部机制。

---

## 2. 核心研究问题

本项目关注以下问题：

1. 模型是否能学习到显式筛法规则？
2. 模型是否能在显式筛法之外学习到模数组合结构？
3. Attention head 是否会自然对应某些模数约束，例如 mod 3、mod 5、mod 7？
4. Candidate prime token 之间是否会形成等价类或替代关系？
5. 模型 residual 部分是否包含低阶筛法无法解释的高阶 pattern？
6. 模型 logit 是否可以被 Fourier / periodic basis 解释？
7. 当模型预测成功时，它主要依赖哪些模块：显式筛法、attention 交互，还是 residual insight？

---

## 3. 为什么不建议继续使用纯 MLP

原始方案使用：

```text
N_bits + modulo_features + density -> Fusion MLP -> 500-way classifier
```

这个结构工程上可行，但有三个问题：

### 3.1 可解释性弱

MLP 很快会把所有输入特征混合到隐藏空间里。即使模型预测效果很好，我们也很难判断：

```text
模型到底是学到了 mod 3 / mod 5 筛法？
还是学到了候选素数之间的交互？
还是只是在记忆训练分布？
```

### 3.2 输出类别之间缺乏显式交互

原始的 500-way classifier 把 500 个候选小素数当成 500 个独立输出维度，但这些候选之间其实有结构关系：

```text
p_i mod r_j 的相似性
候选 p_i 与 p_k 在模空间中的等价类
某个候选失败后其他候选的替代关系
候选集合整体在某个 N 下的稀疏/密集程度
```

MLP 不容易显式表达这些结构。

### 3.3 不适合做机制逆向工程

我们希望做的是 mechanistic interpretability，而不是单纯 feature importance。纯 MLP 缺少清晰的模块边界，不利于做：

```text
head ablation
modulus token ablation
prime-token attention analysis
logit attribution
residual decomposition
```

因此，下一版模型应该从一开始就以“可解释模块化”为核心设计原则。

---

## 4. 总体架构：IGOT

建议模型命名为：

> **Interpretable Goldbach Offset Transformer (IGOT)**

整体结构如下：

```text
Input:
    even number N
    candidate primes p_1 ... p_K
    modulus bases r_1 ... r_M

Modules:
    1. Number Encoder
    2. Explicit Sieve Module
    3. Prime Token Builder
    4. Modulus Token Builder
    5. Prime-Modulus Cross-Attention
    6. Prime-Prime Self-Attention
    7. Residual Insight Head
    8. Deterministic Verifier

Output:
    K-dimensional logits

Training Label:
    valid_mask[i] = 1 if N - p_i is prime else 0
```

最终 logit 分解为：

```text
logit_i = sieve_logit_i + attention_logit_i + residual_logit_i
```

这个分解非常重要，因为它允许我们分析：

```text
最终预测成功来自哪里？
- 显式筛法？
- Attention 学到的高阶交互？
- Residual 中的未知 pattern？
```

---

## 5. 模块一：Number Encoder

Number Encoder 负责编码偶数 `N` 的全局信息。

### 5.1 输入特征

建议包含：

```text
N_bits:              64-bit binary representation
N_mods:              N mod r_j for small primes r_j
log_N:               log(N)
prime_density:       local prime density estimate
Fourier_features:    sin/cos features over modular residues
```

### 5.2 Binary Encoder

如果继续使用 1D Conv，应真正沿着 bit 序列卷积：

```text
N_bits: [B, 64]
-> unsqueeze: [B, 1, 64]
-> Conv1D over length 64
-> pooling / projection
```

不建议先把 64 bit dense 到 256 维再 reshape 成 `[B, 256, 1]`，因为那样 Conv1D 的序列长度为 1，无法真正学习 bit-level 局部模式。

### 5.3 Modulo Encoder

对每个小素数模数 `r_j`，将 `N mod r_j` 作为离散类别输入 embedding：

```text
e_j = Embedding_j(N mod r_j)
```

这是合理的，因为模余值不是连续数值，其大小本身没有普通实数意义。

### 5.4 Fourier Feature Encoder

建议显式加入周期特征：

```text
sin(2π * (N mod r_j) / r_j)
cos(2π * (N mod r_j) / r_j)
```

原因是模空间本质上是周期空间。Fourier-like representation 有助于模型表达周期规律，也方便后续做频谱分析。

---

## 6. 模块二：Explicit Sieve Module

这是整个架构中最重要的可解释模块。

对于每个候选小素数 `p_i` 和每个筛法基底小素数 `r_j`，计算：

```text
z[i, j] = 1 if (N - p_i) mod r_j == 0 else 0
```

如果 `z[i, j] = 1`，说明 `N - p_i` 被 `r_j` 整除。在绝大多数情况下，该候选应该被压低分数。

### 6.1 显式筛法 logit

定义：

```text
sieve_logit[i] = b_i + Σ_j w_j * z[i, j]
```

其中：

```text
b_i: candidate prior bias
w_j: modulus r_j 的筛法惩罚权重
```

训练后，如果模型学到正常筛法机制，我们预期：

```text
w_3, w_5, w_7, ... 都应该是明显负数
```

### 6.2 这个模块的作用

Explicit Sieve Module 提供一个强可解释 baseline：

```text
模型最低限度应该学会：
如果 N - p_i 被小素数整除，则降低 p_i 的成功概率。
```

如果后续 Transformer 模块无法显著超过这个 baseline，说明模型主要只是重新发现了低阶筛法。

---

## 7. 模块三：Prime Token Builder

将每个候选小素数 `p_i` 构造成一个 token。

每个 token 表示一个问题：

```text
对于当前 N，候选 p_i 是否使得 N - p_i 为素数？
```

### 7.1 Prime Token 特征

每个 prime token 可以包含：

```text
prime_id_embedding(p_i)
log(p_i)
log(N - p_i)
p_i mod r_j
N mod r_j
(N - p_i) mod r_j
sieve indicators z[i, j]
global N context from Number Encoder
```

### 7.2 Token 语义

这样构造后，模型不是处理一个抽象的 500 维输出，而是在处理 500 个有明确语义的 token：

```text
Token_i = candidate p_i under current N
```

这为后续 attention 分析提供了基础。

---

## 8. 模块四：Modulus Token Builder

除了 prime tokens，建议显式加入 modulus tokens。

例如：

```text
[M_3], [M_5], [M_7], [M_11], ..., [M_r]
```

每个 modulus token 表示一个筛法维度。

### 8.1 Modulus Token 特征

每个 modulus token `M_j` 可以包含：

```text
r_j identity embedding
N mod r_j
number of candidates eliminated by r_j
elimination pattern summary over p_i
Fourier feature of N mod r_j
```

### 8.2 Modulus Token 的解释价值

加入 modulus tokens 后，我们可以观察：

```text
哪些 prime token 关注 M_3？
哪些 prime token 关注 M_5？
某个 attention head 是否专门负责 mod 7？
是否存在组合 head 同时关注 M_3, M_5, M_7？
```

这使得 Transformer attention 更容易被解释为数论机制。

---

## 9. 模块五：Prime-Modulus Cross-Attention

这是推荐的核心 attention 模块。

设计为：

```text
Query: Prime Tokens
Key:   Modulus Tokens
Value: Modulus Tokens
```

即：

```text
Prime tokens attend to modulus tokens
```

其语义是：

```text
每个候选 p_i 在判断自己是否有效时，关注了哪些模数约束？
```

### 9.1 可解释假设

如果某个候选 `p_i` 满足：

```text
N ≡ p_i mod 3
```

那么：

```text
N - p_i ≡ 0 mod 3
```

模型应该通过 `M_3` token 压低该 candidate 的 logit。

如果训练后发现某个 attention head 大量关注 `M_3`，且 ablation 该 head 后 mod-3 相关错误显著增加，那么可以认为该 head 具有 mod-3 筛法功能。

---

## 10. 模块六：Prime-Prime Self-Attention

在 Prime-Modulus Cross-Attention 之后，加入 Prime-Prime Self-Attention。

设计为：

```text
Prime tokens attend to other prime tokens
```

其语义是：

```text
候选 p_i 与候选 p_k 之间是否存在竞争、替代、等价类或组合关系？
```

### 10.1 可能发现的结构

Prime-Prime attention 可能发现：

```text
1. p_i 和 p_k 在模空间中等价
2. p_i 被某些小模数排除后，p_k 成为更优候选
3. 某些候选集合在特定 residue class 下共同活跃
4. 前 500 个小素数可以被模型聚类成若干模空间等价类
```

### 10.2 Candidate Grouping Analysis

训练后可以对 attention pattern 做 clustering，并检查 cluster 是否对应：

```text
p_i mod 3
p_i mod 5
p_i mod 30
p_i mod 210
p_i mod product(r_1...r_m)
```

如果 cluster 与 residue class 高度一致，说明模型内部确实形成了模空间组织结构。

---

## 11. 模块七：Residual Insight Head

最终输出不应只由 Transformer 直接给出，而应拆成三部分：

```text
final_logit[i] = sieve_logit[i] + attention_logit[i] + residual_logit[i]
```

### 11.1 三部分含义

```text
sieve_logit[i]:
    显式筛法模块给出的可解释低阶规则。

attention_logit[i]:
    Prime-Modulus 与 Prime-Prime attention 学到的交互结构。

residual_logit[i]:
    显式规则和 attention 之外仍然存在的补充模式。
```

### 11.2 Residual 的研究价值

如果模型性能主要来自 sieve_logit，说明神经网络只是重新发现了筛法。

如果 attention_logit 带来明显增益，说明 candidate-modulus 或 candidate-candidate 交互有价值。

如果 residual_logit 在控制显式筛法和 attention 之后仍有显著增益，则可能存在更值得研究的高阶 pattern。

---

## 12. 训练标签设计

原始方案中使用单一 `p1_label`，并用 CrossEntropyLoss 训练。但这不适合本任务。

因为对于一个偶数 `N`，可能存在多个小素数 `p_i` 满足：

```text
N - p_i is prime
```

因此，标签应该是 multi-hot vector：

```text
valid_mask[i] = 1 if is_prime(N - p_i) else 0
```

### 12.1 推荐损失函数

使用：

```python
BCEWithLogitsLoss
```

而不是：

```python
CrossEntropyLoss
```

原因是：

```text
CrossEntropyLoss 假设只有一个正确类别。
BCEWithLogitsLoss 允许多个候选同时为真。
```

---

## 13. 数据结构建议

原始 `26-Byte` 结构中只保存了一个 `p1_label`。新架构建议改为保存 `valid_mask`。

### 13.1 推荐结构

```text
even_number:     uint64
N_mods:          uint8[M]
prime_density:   float32
valid_mask:      bitset[K]
```

如果：

```text
K = 500
```

那么 `valid_mask` 需要 500 bits，约 63 bytes。

### 13.2 是否需要保存 sieve_matrix

不一定需要。

`sieve_matrix` 可以训练时动态计算：

```text
z[i, j] = 1 if (N - p_i) mod r_j == 0 else 0
```

因为 `K × M` 的规模不大，例如：

```text
500 × 32 = 16000 boolean values per sample
```

在 GPU 上可以批量向量化计算。

---

## 14. 推理流程

推理阶段：

```text
1. 输入偶数 N
2. 构造候选小素数集合 p_1...p_K
3. 计算 N 的 bits、mods、density、Fourier features
4. 构造 prime tokens 和 modulus tokens
5. 模型输出 K 个 logits
6. 按 logits 从高到低排序候选 p_i
7. 逐个调用 deterministic primality verifier 检查 N - p_i
8. 找到第一个合法 pair 后返回
```

伪代码：

```python
logits = model(N, candidate_primes, modulus_bases)
ranked_indices = torch.argsort(logits, descending=True)

for idx in ranked_indices[:top_k]:
    p = candidate_primes[idx]
    q = N - p
    if is_prime_64bit(q):
        return p, q

return None
```

注意：模型输出不直接作为数学真理。最终合法性必须由确定性 verifier 判断。

---

## 15. Verifier 设计

由于目标区间在 64-bit 整数范围内，可以使用 deterministic Miller-Rabin 或其他确定性 64-bit primality test。

注意 Miller-Rabin 一般是概率性测试，但对于 64-bit 范围可以使用已知 witness bases 使其确定化。

Verifier 的角色是：

```text
神经网络负责排序候选；
Verifier 负责保证输出 pair 的数学正确性。
```

这也是整个系统避免“神经网络幻觉”的关键。

---

## 16. 实验指标

不建议只看分类 accuracy。

推荐指标包括：

| 指标 | 含义 |
|---|---|
| `vocab_coverage@K` | 前 K 个候选小素数中是否存在合法 p |
| `top1_valid_rate` | top-1 输出是否通过 verifier |
| `top5_valid_rate` | top-5 内是否存在合法候选 |
| `topK_valid_rate` | top-K 内是否存在合法候选 |
| `avg_checks_to_success` | 平均需要 verifier 检查多少个候选 |
| `failure_rate_under_K` | K 次以内找不到合法候选的比例 |
| `BCE_loss` | multi-label 预测损失 |
| `calibration_error` | 输出概率是否校准 |
| `residual_gain` | 相比显式筛法 baseline 的提升 |
| `generalization_by_scale` | 按 N 的数量级分桶测试泛化 |

其中最重要的是：

```text
topK_valid_rate
avg_checks_to_success
residual_gain
```

---

## 17. Baseline 设计

为了判断 Transformer 是否真的发现了新结构，必须建立强 baseline。

### 17.1 Baseline 0: Explicit Sieve Only

```text
logit_i = b_i + Σ_j w_j * z[i, j]
```

没有神经网络，只使用显式筛法特征。

作用：判断低阶筛法能解释多少性能。

### 17.2 Baseline 1: Sieve + Small Residual Network

```text
logit_i = sieve_logit_i + residual_mlp_i
```

作用：判断显式筛法之外是否存在 residual signal。

### 17.3 Baseline 2: Sieve + Prime-Modulus Transformer

```text
logit_i = sieve_logit_i + attention_logit_i + residual_logit_i
```

作用：研究 attention 是否发现更高阶的 candidate-modulus / candidate-candidate 交互。

---

## 18. Interpretability 实验设计

### 18.1 Modulus Ablation

逐个删除模数特征：

```text
remove M_3
remove M_5
remove M_7
remove all modulus tokens
remove explicit sieve module
```

观察：

```text
BCE loss
Top-K valid rate
Avg checks to success
Mod-specific error rate
```

如果删除 `M_3` 后，所有与 mod-3 排除相关的错误显著增加，说明模型确实依赖该模数。

---

### 18.2 Head-Modulus Alignment

分析每个 attention head 从 prime tokens 到 modulus tokens 的注意力分布。

目标是发现：

```text
Head A 是否主要关注 M_3？
Head B 是否主要关注 M_5？
Head C 是否同时关注 M_3, M_5, M_7？
```

但需要注意：

```text
Attention map 不是因果解释。
```

必须配合 head ablation 或 causal mediation 验证。

---

### 18.3 Head Ablation

对每个 attention head 做 ablation：

```text
zero out head output
or gate head output with scalar 0
```

观察性能变化：

```text
ΔBCE_loss
ΔtopK_valid_rate
Δmod-specific error
```

如果某个 head 被删除后，只在 mod-5 相关样本上明显恶化，那么该 head 可能承担 mod-5 功能。

---

### 18.4 Candidate Grouping Analysis

对 Prime-Prime attention matrix 做聚类。

检查 cluster 是否对应：

```text
p_i mod 3
p_i mod 5
p_i mod 30
p_i mod 210
```

如果 attention cluster 与 residue classes 对齐，说明模型在候选小素数集合上形成了模空间几何结构。

---

### 18.5 Logit Attribution

对最终 logit 做分解：

```text
final_logit[i]
= sieve_logit[i]
+ attention_logit[i]
+ residual_logit[i]
```

统计不同模块的贡献：

```text
Variance explained by sieve module
Variance explained by attention module
Variance explained by residual module
```

这可以回答：

```text
模型成功主要来自哪里？
```

---

### 18.6 Fourier Spectrum Analysis

对固定候选 `p_i`，将模型输出看作 `N` 的函数：

```text
s_i(N) = logit for candidate p_i
```

在某个模空间上分析：

```text
N mod M
```

其中：

```text
M = product of selected small primes
```

然后对 `s_i(N mod M)` 做 Fourier decomposition。

如果模型学到的是周期性模结构，则频谱中应出现明显的周期成分。

---

## 19. 可能发现的高阶 insights

### 19.1 单一模数筛法

最基础的结构：

```text
N ≡ p_i mod r_j
=> N - p_i ≡ 0 mod r_j
```

模型应该压低该候选。

---

### 19.2 模数组合结构

模型可能发现某些组合 residue class 具有特殊性质：

```text
(N mod 3, N mod 5, N mod 7, ...)
```

这些组合可能决定前 500 个小素数中 valid candidate 的密度。

---

### 19.3 候选素数替代关系

例如：

```text
当 p = 3 被排除时，p = 5 或 p = 11 在某些 residue class 下更可能成功。
```

这类关系适合由 Prime-Prime attention 捕捉。

---

### 19.4 候选素数等价类

某些候选 `p_i` 在多个模数下具有类似行为，可能形成等价类。

模型可能在 attention pattern 中自动聚类这些候选。

---

### 19.5 尺度相关结构

随着 `N` 增大，`N - p_i` 为素数的概率大致受到 `log N` 影响。

如果模型学到了合理的数论先验，应该能在 bias 或 residual calibration 中体现这种尺度效应。

---

### 19.6 显式筛法之外的 residual pattern

最值得关注的是：

```text
在控制小素数筛法之后，residual module 是否仍然稳定提升性能？
```

如果有，这部分 residual pattern 才是真正值得深入分析的高阶 insight 候选。

---

## 20. 推荐开发路线

### Phase 0: 数据生成与 coverage 验证

目标：

```text
生成 N 的 valid_mask[500]
统计前 500 小素数对不同 N 范围的 coverage
```

关键问题：

```text
对于多少比例的偶数 N，前 500 小素数中存在合法 p？
```

如果 coverage 不足，需要扩展候选词表，例如：

```text
K = 1000, 2000, 5000
```

---

### Phase 1: Explicit Sieve Baseline

训练：

```text
logit_i = b_i + Σ_j w_j * z[i, j]
```

目标：

```text
建立低阶筛法基线
```

---

### Phase 2: Sieve + Residual Network

训练：

```text
final_logit = sieve_logit + residual_logit
```

目标：

```text
确认是否存在显式筛法之外的可学习 signal
```

---

### Phase 3: Sieve + Prime-Modulus Transformer

加入：

```text
prime tokens
modulus tokens
prime-modulus cross attention
prime-prime self attention
```

目标：

```text
研究 attention 是否能发现高阶交互结构
```

---

### Phase 4: Mechanistic Interpretability

进行：

```text
head-modulus alignment
head ablation
candidate clustering
logit attribution
Fourier spectrum analysis
residual decomposition
```

目标：

```text
逆向工程模型内部机制
```

---

## 21. PyTorch 模型伪代码

```python
class IGOT(nn.Module):
    def __init__(self, candidate_primes, modulus_bases, d_model=128):
        super().__init__()
        self.candidate_primes = candidate_primes
        self.modulus_bases = modulus_bases

        self.number_encoder = NumberEncoder(d_model)
        self.prime_token_builder = PrimeTokenBuilder(d_model)
        self.modulus_token_builder = ModulusTokenBuilder(d_model)

        self.prime_mod_cross_attn = CrossAttentionBlock(d_model)
        self.prime_prime_self_attn = TransformerEncoderBlock(d_model)

        self.sieve_module = ExplicitSieveModule(num_moduli=len(modulus_bases))
        self.attention_head = nn.Linear(d_model, 1)
        self.residual_head = nn.Linear(d_model, 1)

    def forward(self, N_features):
        # 1. Encode global number context
        number_context = self.number_encoder(N_features)

        # 2. Build sieve matrix z[i, j]
        sieve_matrix = compute_sieve_matrix(
            N_features.even_number,
            self.candidate_primes,
            self.modulus_bases
        )

        # 3. Explicit sieve logits
        sieve_logits = self.sieve_module(sieve_matrix)

        # 4. Build tokens
        prime_tokens = self.prime_token_builder(
            N_features,
            self.candidate_primes,
            sieve_matrix,
            number_context
        )

        modulus_tokens = self.modulus_token_builder(
            N_features,
            self.modulus_bases,
            sieve_matrix,
            number_context
        )

        # 5. Prime-Modulus Cross Attention
        prime_tokens = self.prime_mod_cross_attn(
            query=prime_tokens,
            key=modulus_tokens,
            value=modulus_tokens
        )

        # 6. Prime-Prime Self Attention
        prime_tokens = self.prime_prime_self_attn(prime_tokens)

        # 7. Attention and residual logits
        attention_logits = self.attention_head(prime_tokens).squeeze(-1)
        residual_logits = self.residual_head(prime_tokens).squeeze(-1)

        # 8. Final decomposed logits
        final_logits = sieve_logits + attention_logits + residual_logits

        return {
            "final_logits": final_logits,
            "sieve_logits": sieve_logits,
            "attention_logits": attention_logits,
            "residual_logits": residual_logits
        }
```

---

## 22. 关键设计原则

### 22.1 不让模型替代验证器

模型只负责候选排序，不负责数学证明。

### 22.2 不追求黑箱最高性能

如果更复杂模型提升性能但无法解释，不一定符合项目目标。

### 22.3 每个 token 必须有数论语义

不要把 Transformer 作为普通序列模型使用。Prime token、modulus token、sieve indicator 都必须有明确解释。

### 22.4 每个模块都要可 ablate

模型结构必须支持：

```text
remove sieve module
remove modulus tokens
remove specific attention head
remove residual head
```

### 22.5 先解释低阶，再寻找高阶

必须先建立 explicit sieve baseline。只有当 residual / attention 在 baseline 之上有稳定提升，才有资格讨论高阶 insight。

---

## 23. 总结

IGOT 的核心不是训练一个“会输出小素数”的神经网络，而是训练一个可以被逆向工程的数论探针。

推荐最终框架是：

```text
Explicit Sieve Module
+ Prime Token Transformer
+ Modulus Token Branch
+ Prime-Modulus Cross-Attention
+ Prime-Prime Self-Attention
+ Residual Insight Head
+ Deterministic Verifier
```

最终输出分解为：

```text
logit_i = sieve_logit_i + attention_logit_i + residual_logit_i
```

通过这种结构，我们可以系统性回答：

```text
模型是否学到了筛法？
是否学到了模数组合？
是否学到了候选素数之间的交互？
是否存在显式数论规则之外的 residual insight？
```

这才是该项目最有研究价值的部分。

---

## 24. 参考方向

本项目的方法论可以参考以下研究方向：

1. Mechanistic interpretability：通过逆向工程模型内部组件来理解模型学到的算法。
2. Grokking and modular arithmetic：模型在模运算任务中可能学到 Fourier-like representation。
3. Transformer circuits：attention head 可以形成可分析的功能 circuit。
4. Causal head ablation / gating：attention map 不能直接等同于解释，需要通过 ablation 和 causal intervention 验证。
5. Deterministic 64-bit primality testing：模型输出必须由确定性 verifier 进行最终数学验证。
