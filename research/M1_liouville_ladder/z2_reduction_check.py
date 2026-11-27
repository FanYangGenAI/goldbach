"""检查 z=2 台阶（a、b 都为奇数）的约化引理：
对奇数 d | N（N 偶），P(N) ⊇ λ(d)·P(N/d)，其中 P(M) = 偶数 M 的奇数子纤维上出现的符号模式集合。
并列出 N ≤ 4000 中 (−,−) 不能由任何真因子约化得到、需要"直接"处理的 N（核心情形）的形状。"""
import numpy as np
M = 5000
lam = np.ones(M + 1, dtype=int)
for n in range(2, M + 1):
    m, c, p = n, 0, 2
    while p * p <= m:
        while m % p == 0: m //= p; c += 1
        p += 1
    if m > 1: c += 1
    lam[n] = (-1) ** c
def P(N):
    return {(int(lam[b]), int(lam[N - b])) for b in range(1, N, 2)}
cache = {N: P(N) for N in range(2, 4001, 2)}
bad, core = 0, []
for N in range(2, 4001, 2):
    derived = set()
    for d in range(3, N + 1, 2):
        if N % d == 0:
            s = int(lam[d]); derived |= {(s * x, s * y) for x, y in cache[N // d]}
    if not derived <= cache[N]: bad += 1
    if (-1, -1) in cache[N] and (-1, -1) not in derived and N >= 100:
        core.append(N)
def shape(N):
    k = 0
    while N % 2 == 0: N //= 2; k += 1
    return k, N
print("引理反例个数:", bad)
print("N∈[100,4000] 中 (−,−) 只能直接得到的 N 的个数:", len(core), " 例:", core[:15])
from collections import Counter
print("核心情形的 2-adic 指数分布:", Counter(shape(N)[0] for N in core).most_common(6))
print("所有 N∈[6,4000] 偶数都有 (−,−)?", all((-1, -1) in cache[N] for N in range(6, 4001, 2)))
