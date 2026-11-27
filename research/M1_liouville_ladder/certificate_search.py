"""M1-E3：寻找"刚性 ⇒ g(17)=+1"的 GF(2) 证书。
在 N=2^k 的奇数子纤维上，求奇数集合 S 使 Σ_{n∈S} [Ω_q(n)+Ω_q(N−n)]_q ≡ e_17（mod 2，对所有奇素数 q），且 |S| 为偶数（η 抵消）。
若存在，则任何刚性的完全积性 g 都有 g(17)=+1；而 λ(17)=−1，故 λ 在 2^k 不刚性。
然后看证书里的 n 的形状（n/N 的位置、因子结构），寻找与 k 无关的模式。"""
import sys, json, random
from rigidity_gf2 import odd_primes_upto, parity_vector
def certificate(k, target=17, seed=0):
    N = 1 << k
    primes = odd_primes_upto(N); idx = {p: i for i, p in enumerate(primes)}; E = len(primes)
    spf = list(range(N + 1))
    for i in range(2, int(N ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j: spf[j] = i
    ns = list(range(1, N // 2 + 1, 2))
    rows = [parity_vector(n, idx, spf) ^ parity_vector(N - n, idx, spf) ^ (1 << E) for n in ns]
    # 消元并记录组合（组合用一个大整数位集表示所选行）
    piv = {}
    for i, r in enumerate(rows):
        comb = 1 << i
        while r:
            h = r.bit_length() - 1
            if h in piv: r ^= piv[h][0]; comb ^= piv[h][1]
            else: piv[h] = (r, comb); break
    t = 1 << idx[target]          # 目标：e_17，e 分量为 0
    comb = 0
    while t:
        h = t.bit_length() - 1
        if h not in piv: return None
        t ^= piv[h][0]; comb ^= piv[h][1]
    S = [ns[i] for i in range(len(ns)) if comb >> i & 1]
    return N, S
for k in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    res = certificate(k)
    if res is None: print(k, "无证书"); continue
    N, S = res
    print(json.dumps(dict(k=k, N=N, size=len(S), smallest=S[:12], ratios=[round(n / N, 4) for n in S[:12]])), flush=True)
