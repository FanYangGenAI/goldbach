# 三次函数域二阶矩 M₂：由一阶矩消去二次位移类和

**状态：完整证明稿，供交叉审阅。** 本稿不以数值重构的 M₂ 为前提。输入是 `0a25e29` 的五个一阶类型矩证明中的两项：已证明的 R_{I₃}，以及 `proof_R3.md` 引理 4、5 的 `M(3,1)=(q−epsilon)/3`。上游证明的两处笔误及补充核对见同次审阅文档。

沿用已审范围：q 为奇素数、q≠3，epsilon=1（q≡1 mod 3）或 −1（q≡2 mod 3）。下文所有“不可约二次”均允许非零常数倍。

## 1. 定义与目标

令 V 为次数小于 3 的多项式构成的加法群，

\[
L=\{\text{一次或二次不可约多项式的非零常数倍}\},\quad
P=\{H-t^3:H\text{ 首一三次不可约}\}.
\]

将首一三次的低三位系数与 V 等同，令

\[
\ell=|L|=\frac{q(q^2-1)}2,\quad n=|P|=\frac{q^3-q}3,
\quad r=L*P,
\]

\[
A_L(h)=\sum_G1_L(G)1_L(G+h),\quad
A_P(h)=\sum_H1_P(H)1_P(H+h).
\]

交换有限求和即得

\[
M_2=\sum_F r(F)^2=\sum_{h\in V}A_L(h)A_P(h),\qquad
\sum_{h\in V}A_P(h)=n^2. \tag{1}
\]

记

\[
S=\sum_{h\in\mathbb F_q^\times}A_P(h),\quad
T=\sum_{\deg h=1}A_P(h),\quad
R=R_{I_3}=\sum_{F\text{ 三次不可约}}r(F).
\]

目标为第十四轮的两支候选 M₂ 公式。本稿证明它们，无需求单个 `A_P(t²+e)`。

## 2. 第一个关键恒等式：不可约位移和就是 R

由 R 的定义，

\[
R=\sum_F1_P(F)\sum_{h\in L}1_P(F-h)
=\sum_{h\in L}A_P(h). \tag{2}
\]

这里用了自相关 `A_P(-h)=A_P(h)`。所以

\[
\boxed{\sum_{\substack{\deg h=2\\h\text{ 不可约}}}A_P(h)=R-T.} \tag{3}
\]

这就是计划中尚缺的二次平方类和所需的额外关系。R 已在一阶类型矩中出现，不需要增加一个独立的椭圆曲线计数问题。

## 3. 第二个关键恒等式：二重根位移和等于 T

先考虑 h=a(t-b)，a≠0，及 H、H+h 都不可约的有序对。不可约性保证 `c=H(b)≠0`。令

\[
K(u)=c^{-1}u^3H(b+u^{-1}).
\]

K 是首一三次不可约，且 `K(0)=1/c`：u=0 时非零，u!=0 时由 H 在 F_q 中无根知 K 也无根。另一个多项式变成

\[
K(u)+(a/c)u^2.
\]

对固定的 b，这是一个双射：反向由 K、d≠0 且 K、K+du² 不可约，取 `c=1/K(0)`、`a=dc`，并置

\[
H(t)=c(t-b)^3K((t-b)^{-1}).
\]

反向 H 首一且不可约。于是

\[
T=q\sum_{d\ne0}A_P(dt^2).
\]

而对 h=d(t-b)²，平移不变性同样给出

\[
\sum_{\substack{\deg h=2\\\operatorname{disc}h=0}}A_P(h)
=q\sum_{d\ne0}A_P(dt^2)=\boxed{T}. \tag{4}
\]

此处不需要证明一次位移两个平方类上的单个值相等。

## 4. 一次位移总和 T：范数、迹计数

令 K=F_{q³}，Tr、Nm 分别为到 F_q 的迹和范数。用 `m(a,d)` 表示首一三次不可约 `t³+a t²+c t+d` 在 c 上的个数。

固定 a,d 后，两个不可约多项式只在一次项不同的有序对有 `m(a,d)(m(a,d)−1)` 个。再由平移消去一次位移的常数项，

\[
T=q\left[\sum_{a,d}m(a,d)^2-n\right]. \tag{5}
\]

每个不可约多项式在 K 中有 3 个根，所以 `9 sum m²` 等于以下有序根对数 N：

\[
\alpha,\beta\in K\setminus F_q,\qquad
\operatorname{Tr}\alpha=\operatorname{Tr}\beta,\quad
\operatorname{Nm}\alpha=\operatorname{Nm}\beta.
\]

写 `beta=z alpha`，则 Nm z=1。

- z=1 的贡献为 q³−q。
- z≠1：方程 `Tr((1−z)alpha)=0` 是一个非零线性方程，解空间有 q² 个点，去掉 0 后有 q²−1 个。
- 其中 alpha 在 F_q^× 的无效解有 `(q−1)1_{Tr z=3}` 个；beta 在 F_q^× 的无效解有 `(q−1)1_{Tr z^{-1}=3}` 个。
- 这两类无效解不交：若 alpha、beta 都在 F_q^×，迹相等且 3≠0 就迫使 alpha=beta，因而 z=1，与当前条件矛盾。

范数为 1 的非零元素共 q²+q+1 个。另由已经证明的 R₃ 引理，

\[
\#\{z:\operatorname{Nm}z=1,\operatorname{Tr}z=3\}
=1+3m(3,1)=1+q-\epsilon. \tag{6}
\]

式 (6) 的说明：F_q 中只有 z=1 满足条件；其余元素的最小多项式为 `t³−3t²+c t−1`，置 t=−u 后对应 `u³+3u²+c u+1`，每个不可约式贡献 3 个根。

用 z↦z^{-1} 的双射将两类扣除量相加：

\[
\begin{aligned}
N&=(q^3-q)+(q^2+q)(q^2-1)-2(q-1)(q-\epsilon)\\
 &=q^4+2q^3-3q^2+2\epsilon q-2\epsilon.
\end{aligned}
\]

代回式 (5) 得

\[
\boxed{T=\frac{q(q-1)(q^3-3q+2\epsilon)}9.} \tag{7}
\]

这是 M₂ 所需的总和。它不以单个一次位移 A_P(at) 的候选为前提。

## 5. 非零常数位移总和 S：初等证明

令 `f_p(t)=t³+pt`，`u_p=q-|Im f_p|`。这是在 c 变化时 `t³+pt+c` 不可约的个数。消去二次项的平移是双射，故

\[
S=q\sum_{p\in F_q}u_p(u_p-1). \tag{8}
\]

首先，p=0 时由立方映射可得 `u_0=2(q−1)/3`（epsilon=1）或 0（epsilon=−1）。

若 p≠0，记 f_p 的纤维中有 j 个不同有理点的值有 N_j 个。有序碰撞 `f_p(x)=f_p(y),x!=y` 满足

\[
x^2+xy+y^2=-p.
\]

该二次曲线有 q−epsilon 个仿射点；用 `u=2x+y` 化为 `u²+3y²=-4p`，也可直接由二次特征和得到。减掉 x=y 的解数

\[
\nu=\#\{x:3x^2=-p\}\in\{0,2\}
\]

后，碰撞数为 q−epsilon−nu。

两个有理临界点对应两个不同临界值，且每个临界纤维有一个双根和一个不同单根，所以 N₂=nu。由碰撞计数与纤维大小计数，

\[
2N_2+6N_3=q-\epsilon-\nu,\qquad
u_p=N_2+2N_3=\frac{q-\epsilon}{3}.
\]

代入式 (8)，得到

\[
\boxed{S=\frac{q(q-1)(q^2-q-4-2\epsilon)}9.} \tag{9}
\]

这与 Bary-Soroker–Stix 的非零常数位移平均公式相同。本稿只需这个平均，不需其单个位移的椭圆曲线表达式。

## 6. L 的自相关：完全显式的权重

记

\[
B=\frac{q^3+q^2-5q-1}{4},\quad
c_I=\frac{q^2-2q-1}{2},\quad c_D=\frac{q^2-1}{4}.
\]

我们证明

\[
A_L(0)=\ell,\quad A_L(h)=\ell/2\ (h\text{ 非零常数}),\quad
A_L(h)=B\ (\deg h=1), \tag{10}
\]

且对二次位移

\[
\boxed{A_L(h)=B+1-c_I1_{h\text{ 不可约}}-c_D1_{\operatorname{disc}h=0}.} \tag{11}
\]

### 6.1 常数与一次位移

令 s=(q−1)/2 为非平方个数。非零常数位移下，一次式贡献 q(q−1)。对于二次式，判别式 D、D−4ah 均须为非平方；求和 a≠0 后等同于两个不同非平方的有序选择，另有 q 个一次系数，故贡献 q s(s−1)。总和为 ell/2。

对一次位移，利用 L 对单位倍数和平移的不变性，可取 h=t。一次式配对贡献 q(q−2)。二次式的判别式差为 2b+1；b 遍历 F_q 时，该差也遍历 F_q。对每个非零二次首项，选择两个非平方判别式有 s² 种，贡献 `(q−1)s²`。总和为 B。

### 6.2 二次位移

对 A_L 可以用单位倍数、平移将 h 化为 t²+e。G 的二次首项为 a。

- a=0 或 −1：一边一次、一边二次，总贡献为 (q−1)²。
- a≠0,−1：写
  \[
  D=b^2-4ac,\quad D'=b^2-4(a+1)(c+e),\quad k=4a(a+1)e.
  \]
  所需是 D、D' 都为非平方。

用 `1_{nonsquare}(D)=(1-eta(D)-1_{D=0})/2` 展开。固定 a 后，记

\[
K_a=\sum_{b,c}\eta(DD'),\quad
J_a=\sum_{D'=0}\eta(D)+\sum_{D=0}\eta(D'),\quad
Z_a=\#\{D=D'=0\}.
\]

配对数为 `(q²−2q+K_a+J_a+Z_a)/4`。令 `rho_a=eta(a(a+1))`、`delta=eta(-e)`，直接二次特征和给出

\[
Z_a=1+\delta\rho_a,\qquad K_a=q\delta,
\]

\[
J_a=\begin{cases}
-\eta(-a)-\eta(a+1),&e\ne0,\\
(q-1)[\eta(-a)+\eta(a+1)],&e=0.
\end{cases}
\]

例如 D=0 后 `D'=-(b²+k)/a`，D'=0 后 `D=(b²+k)/(a+1)`；这些式子给出 J_a。对固定 b，两个关于 c 的一次式的根相同时，乘积特征和为 `(q−1)rho_a`，否则为 `−rho_a`。共有 Z_a 个 b 属于前一种情形，因此

\[
K_a=\rho_a\bigl[(q-1)Z_a-(q-Z_a)\bigr]
=q\rho_a(Z_a-1)=q\delta.
\]

最后使用

\[
\sum_{a\ne0,-1}\rho_a=-1,\quad
\sum_{a\ne0,-1}\eta(-a)=\sum_{a\ne0,-1}\eta(a+1)=-1.
\]

加上跨次数的贡献，可得

\[
A_L(t^2+e)=
\begin{cases}
(q^3-5q+4)/4,&e=0,\\
[q^3-3q+4+\eta(-e)(q^2-2q-1)]/4,&e\ne0.
\end{cases}
\]

分别对应双根、两个不同有理根、不可约三种情况，正好是式 (11)。

## 7. 合成 M₂

由式 (1)、(3)、(4)，

\[
\begin{aligned}
\sum_{\deg h=2}A_P(h)&=n^2-n-S-T,\\
\sum_{\deg h=2,\ h\,\mathrm{irred}}A_P(h)&=R-T,\\
\sum_{\deg h=2,\ \operatorname{disc}h=0}A_P(h)&=T.
\end{aligned}
\]

代入自相关权重：

\[
\boxed{M_2=\ell n+\frac\ell2S+BT
+(B+1)(n^2-n-S-T)-c_I(R-T)-c_DT.} \tag{12}
\]

等价地，

\[
M_2=\frac{(q-1)^2(q+3)}4n^2
+\frac{(q-1)(q^2+3)}4n
-\frac{(q-1)(q-3)}4S
+\frac{(q-5)(q+1)}4T
-\frac{q^2-2q-1}{2}R. \tag{13}
\]

使用已经证明的一阶矩

\[
R=\begin{cases}
q(q-1)^2(q^3+q^2-q-4)/18,&\epsilon=1,\\
q(q-2)(q-1)(q+1)(q^2+q+1)/18,&\epsilon=-1,
\end{cases}
\]

以及式 (7)、(9)，代数化简得到

\[
\boxed{
M_2=
\begin{cases}
\dfrac{q(q-1)^2(q^6+2q^5-3q^3+9q-3)}{36},&q\equiv1\pmod3,\\[5pt]
\dfrac{q(q-1)(q+1)(q^6-2q^4-q^3+4q^2+3q+5)}{36},&q\equiv2\pmod3.
\end{cases}} \tag{14}
\]

这正是原 M₂ 候选，合成时没有将它代入任何中间步骤。∎

## 8. 证明依赖与复核范围

- 新的严格步骤：R 的位移和解释、互反双射、范数/迹线性族二阶计数、常数位移的值集平均、L 的二次特征和权重。
- 上游输入：已审阅的 R_{I₃} 一阶矩和 R₃ 引理中的 M(3,1)。二者均未使用 M₂。
- 已用符号代数核对两支合成，差为 0。
- `audit_m2_proof.py` 对 q=5,7,11,13,17,19,23,29,31,41 的全部低三位系数核对每一个 A_L 权重、各位移总和、范数/迹计数，以及直接 M₂ 与式 (12)，均精确相等。数值检查不替代上述证明。

在交叉审阅确认本稿和上游一阶矩后，第十四轮的有限 q 方差公式与两支展开即获得证明所需的最后一个算术输入；其背景仍须保持为原先截到次数 3 的定义。
