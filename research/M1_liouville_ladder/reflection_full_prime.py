"""M1-E9（探索性）：完整纤维（允许偶数）时，一致的反射机制能推出多少 T(t)？
对素数 p，刚性解空间只有 {1, χ_p}，二者都满足所有 T(t)（t 的素因子 < p），所以 GF(2) 闭包是 100%。
问题：反射图（T(t) ⇔ T(dN − t)，d 为 t 的任一因子 ≥ 2 且 t/d < N）从 {t < N} 出发能覆盖多少？"""
import sys, json
from collections import deque
def run(N, cap_mult=40):
    CAP = cap_mult * N
    spf = list(range(CAP + 1))
    for i in range(2, int(CAP ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, CAP + 1, i):
                if spf[j] == j: spf[j] = i
    def divisors(t):
        fs = {}
        while t > 1:
            q = spf[t]; fs[q] = fs.get(q, 0) + 1; t //= q
        ds = [1]
        for q, e in fs.items(): ds = [d * q ** i for d in ds for i in range(e + 1)]
        return ds
    def pplus(t):
        m = 1
        while t > 1: m = max(m, spf[t]); t //= spf[t]
        return m
    imp = bytearray(CAP + 1); dq = deque()
    for t in range(1, N): imp[t] = 1; dq.append(t)
    while dq:
        u = dq.popleft()
        for d in divisors(u):
            if d < 2 or d * N <= u: continue
            v = d * N - u
            if v <= CAP and not imp[v]: imp[v] = 1; dq.append(v)
    tot = ok = 0
    for t in range(N + 1, 3 * N):
        if t % N == 0 or max(pplus(t), pplus(t % N)) >= N: continue
        tot += 1; ok += imp[t]
    return dict(N=N, t_considered=tot, reflection_closure_frac=ok / tot)
for N in [int(a) for a in sys.argv[1:]]:
    print(json.dumps(run(N)), flush=True)
