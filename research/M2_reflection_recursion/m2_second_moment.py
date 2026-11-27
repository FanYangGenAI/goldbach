"""M2-E2（探索性）：二阶矩/Chung–Erdős 路线是否可行。
对每个 n，X(n) = 从 qn 出发、长度 ≤ L、乘数受限（d ≤ Z 或 d = q·a, a ≤ Z）的成功反射路径条数（不同路径分别计数）。
Chung–Erdős：P(X ≥ 1) ≥ E[X]² / E[X²]。若对短路径族该下界 ≥ 3/4（且随 q 不变坏），引理 R 的密度版（ε = 1/4）就可以归结为显式和式的估计。"""
import sys, json
import numpy as np
import m2_e1 as M
def run(p, q, Z, L):
    CAP = q * (Z + 1) * p + p
    spf = M.spf_table(min(CAP, 60_000_000))
    def allowed_divs(t):
        fs = {}; x = t
        while x > 1:
            r = int(spf[x]); fs[r] = fs.get(r, 0) + 1; x //= r
        ds = [1]
        for r, e in fs.items(): ds = [d * r ** i for d in ds for i in range(e + 1)]
        return [d for d in ds if d >= 2 and (d <= Z or (d % q == 0 and d // q <= Z))]
    def count(t, depth):
        j = t // p
        if j == 0: return 1
        if depth == 0: return 0
        tot = 0
        for d in allowed_divs(t):
            if d <= j: continue
            v = d * p - t
            if v % p == 0 or v > CAP: continue
            tot += count(v, depth - 1)
        return tot
    X = np.array([count(q * n, L) for n in range(1, p)], dtype=np.float64)
    EX, EX2 = X.mean(), (X ** 2).mean()
    return dict(p=p, q=q, Z=Z, L=L, EX=round(EX, 3), EX2=round(EX2, 2), P_cover=round(float((X > 0).mean()), 4),
                chung_erdos_lower=round(EX ** 2 / EX2, 4) if EX2 > 0 else 0)
for q, zf, L in [(13, 3, 3), (13, 6, 3), (13, 6, 4), (23, 3, 3), (23, 6, 3), (23, 6, 4), (41, 3, 3), (41, 6, 3)]:
    print(json.dumps(run(1009, q, zf * q, L)), flush=True)
