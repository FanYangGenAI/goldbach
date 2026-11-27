"""M1-E4：异类刚性解的等变缺陷在哪里。
取 2^k 奇数子纤维刚性的 GF(2) 解空间中的随机元素 g（完全积性 ±1），计算
  E_d = {n 奇 < N : g(dn mod N) ≠ g(d) g(n)}，d = 3, 5,
并统计 E_d 中元素 n 与 dn mod N 的最大素因子 P⁺。预期（"因子递推"图景）：缺陷只出现在牵涉大素数（> N/5 量级）的 n 上，
比例约 log 5 / log N。"""
import sys, json, random
from rigidity_gf2 import odd_primes_upto, parity_vector
from rigidity_gf2_nullspace import nullspace
def run(k, trials=5, seed=1):
    rnd = random.Random(seed)
    N = 1 << k
    primes = odd_primes_upto(N); idx = {p: i for i, p in enumerate(primes)}; E = len(primes)
    spf = list(range(N + 1))
    for i in range(2, int(N ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j: spf[j] = i
    rows = [parity_vector(n, idx, spf) ^ parity_vector(N - n, idx, spf) ^ (1 << E) for n in range(1, N // 2 + 1, 2)]
    basis = nullspace(rows, E + 1)
    def Pplus(n):
        m = 1
        while n > 1: p = spf[n]; m = max(m, p); n //= p
        return m
    out = []
    for t in range(trials):
        x = 0
        for b in basis:
            if rnd.random() < 0.5: x ^= b
        g = {}
        def gv(n):
            v = parity_vector(n, idx, spf) & x
            return -1 if bin(v).count("1") % 2 else 1
        # 先验证刚性
        eta = gv(1) * gv(N - 1)
        assert all(gv(n) * gv(N - n) == eta for n in range(1, N, 2))
        res = {}
        for d in (3, 5):
            Ed = [n for n in range(1, N, 2) if gv((d * n) % N) != gv(d) * gv(n)]
            big = [n for n in Ed if max(Pplus(n), Pplus((d * n) % N)) > N // 7]
            res[d] = dict(size=len(Ed), frac=len(Ed) / (N // 2),
                          frac_involving_prime_gt_N_over_7=(len(big) / len(Ed)) if Ed else None)
        out.append(res)
    return dict(k=k, dim=len(basis), trials=out)
for k in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    print(json.dumps(run(k)), flush=True)
