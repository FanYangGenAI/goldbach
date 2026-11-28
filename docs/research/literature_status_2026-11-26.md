# 文献与前提状态核查（2026-11-26，Claude 自主线）

> 目的：回答阶段报告第 6 节的部分文献问题，并记录 2026 年 9–11 月改变了"前提状态"的新进展。每条都附来源；未经同行评审的标注"预印本"。

## 1. 前提状态的变化

| 前提 | 状态 | 来源 |
|---|---|---|
| RH（ζ） | Alpöge 2026-11-20 发布 1700 万行 Lean 证明；Lean FRO 与 Mathlib Initiative 2026-11-21 联合声明认证"陈述与证明均正确" | leanprover-community.github.io/blog/posts/riemann-hypothesis/ |
| GRH（Dirichlet L） | Alpöge 同时宣称；上述认证声明 **只覆盖 RH**，不覆盖 GRH | 同上 |
| 零点自由半平面 Re s > 7/8（全部 Dirichlet L） | OpenAI 预印本（2026-09-30）；Liu 改进到 0.874957…；有 Lean 范围说明，未见完整认证 | arXiv:2610.12234、2610.15291 |

## 2. 直接受影响的哥德巴赫方向结果

| 结果 | 原条件 | 现在 |
|---|---|---|
| Chirre–Hagen：每个 (x, x + 123 log²x] 含哥德巴赫数 | RH | 随 RH 认证变为无条件（前提是该文无误） |
| 例外集 E(X) ≪ X^{1/2+ε} | GRH（Hardy–Littlewood 1924）；或固定零点自由半平面（Li–Liu，arXiv:2610.26756） | 仍条件性（GRH 未认证；7/8 半平面未认证） |
| 例外集无条件纪录 | — | E(X) ≪ X^{0.709}（赵根恒，arXiv:2511.05631）；Pintz–Althöfer 在零点自由半平面假设下 X^{7/12} |
| Linnik–哥德巴赫：两个素数 + K 个 2 的幂 | — | 无条件 K = 6，GRH 下 K = 5（Li–Liu，arXiv:2610.15784）；作者称无条件 K = 5 即将发表 |
| Shusterman 问题：N = a + b，λ(a) = λ(b) = −1 | GRH（Mangerel，arXiv:2412.17199） | Mangerel Remark 1：零点自由区 Re s > 1 − (log N)^{−c} 即可，故 7/8 半平面若成立也足够 |

## 3. 与本纲领直接相关的已知结果（审阅者问题的部分回答）

- **函数域二元哥德巴赫**：Sawin–Shusterman（Annals 196, 2022）在 q 为奇素数 p 的幂、q > 685090·p² 时证明了孪生素数的 HL 渐近，Remark 1.2 说证明同时给出函数域哥德巴赫。**质数域 q = p 不在范围内**。机制：特征 p 下 (r + s^p)′ = r′，Pellet 公式把 μ(r + s^p) 变成一个二次 Dirichlet 特征，再用几何（消失循环、Lefschetz）得到强于 Burgess 的短特征和界。整数中没有 Frobenius，整数版 Chowla 型猜想预测 λ 在多项式族上随机，所以这一机制没有已知的整数类比。
- **Bender–Pollack（arXiv:0912.1702）**：q 奇且 q > 8(n+6)^{2n²} 时每个 n 次首一多项式都有哥德巴赫表示。
- **F₂[t] 中"偶数"的正确类比**：被 t(t+1) 整除。（本轮穷举：4–16 次的"偶"多项式全部可表示；与 t(t+1) 互素者都不能表示，可严格证明：deg Q ≥ 2 的不可约 Q 满足 Q(0) = Q(1) = 1。）
- **Friedlander–Iwaniec 渐近筛（Annals 148, 1998）**：要求分布水平 D 满足 x^{2/3} < D < x，加双线性假设。用到 {N − p} 上即本纲领 majority 引理的 (A)。分布水平纪录：Lichtman 66/107 ≈ 0.617（三重可分解权重，arXiv:2309.08522）；Maynard 曾认为 x^{3/5} 是当时方法的极限。双线性假设经 Cauchy–Schwarz 展开后变为计数 m 使 N − mn₁、N − mn₂ 同为素数，属孪生素数型，与目标同难度。
- **Liouville 纤维和**（本纲领 O19/R13-2 的对象）在文献中称为 Corrádi–Kátai 猜想（1969）：Σ_{a+b=N} λ(a)λ(b) = o(N)。现状见 `research/M1_liouville_ladder/`。

## 4. 本轮新开的线

`gb/M1-liouville-ladder`：把 Shusterman 问题（条件性已证）与哥德巴赫用一个参数 z 连起来的"粗糙阶梯" G(z)，G(N^{1/3}) 正是哥德巴赫。

## 更新（2026-11-27 22:25）：Corrádi–Kátai / Shusterman 方向的两篇新文献

- **Meng（eScholarship，2026-09-24）**，"Decompositions of every integer N ≥ 11"，https://escholarship.org/content/qt7945n3rh/qt7945n3rh.pdf
  - 定理 A **无条件**成立：对一切 N ∉ {2, 3, 4, 5, 6, 9, 10} 和任一符号对 (s, t)，存在 a + b = N 使 λ(a) = s、λ(b) = t。
  - 文中说这是 Mangerel 定理 1.3（需要 GRH 和非光滑条件）的"无条件定性加强"。
  - 推论：N ≥ 11 时 |L_λ(N)| < N − 1，证明中还得到 ≤ N − 5。
  - 它**不给**表示个数的下界，所以不对应 Mangerel 的命题 1.4。
  - 因此第 21 行"Shusterman 问题：GRH（Mangerel）"已经过时，应改为"无条件（Meng 2026）"。
- **Krishnamoorthy（arXiv:2608.13266，2026-08-13）**，"On a conjecture of Corrádi and Kátai"，无条件。
  - 对任意 δ > 0，满足 |G₂(N)| > δN 的 N ≤ X 至多有 ≪_{A,δ} X/log^A X 个，故 lim inf |G₂(N)|/N = 0。
  - 完整的 o(N) 猜想（对一切 N）仍未解决。
- **对 M3 的影响**：
  - M3 的目标是：在 GRH 下，对一切素数 N 证明 |L_λ(N)| ≤ N − cN/((log N)²(log log N)^{2+ε})。
  - 它对应的比较对象是 Mangerel 命题 1.4（GRH）：|L_λ(N)| ≤ N − N·e^{−C(log log N)⁶}。这一点已用 arXiv HTML 版核对。
  - 上面两篇新文献都不覆盖这个"对一切 N 的定量节省"，所以 M3 的目标仍然是新的（若能完成）。但它的意义是对已有定理的定量改进，**不是**通向哥德巴赫的关键步骤。
