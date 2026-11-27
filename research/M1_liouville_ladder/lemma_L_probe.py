"""M1-E5（探索性）：目标引理 L 的数值探针。
从 2^k 的异类刚性解 g 出发，随机翻转 t 个素数上的值（小素数或大素数），得到"近刚性"函数，测
  |E|（少数符号对数）与 |E_3|、|E_5|，看 |E_d| − |E_d(g_0)| 是否 ≲ C·|E|。"""
import sys, json, random
from rigidity_gf2 import odd_primes_upto, parity_vector
from rigidity_gf2_nullspace import nullspace
def run(k, seed=3):
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
    x0 = 0
    for b in basis:
        if rnd.random() < 0.5: x0 ^= b
    pv = [0] * N
    for n in range(1, N, 2): pv[n] = parity_vector(n, idx, spf)
    def stats(x):
        g = [0] * N
        for n in range(1, N, 2): g[n] = -1 if bin(pv[n] & x).count("1") % 2 else 1
        prod = [g[n] * g[N - n] for n in range(1, N, 2)]
        plus = sum(1 for v in prod if v == 1); Esz = min(plus, len(prod) - plus)
        out = {"E": Esz}
        for d in (3, 5):
            out[f"E{d}"] = sum(1 for n in range(1, N, 2) if g[(d * n) % N] != g[d] * g[n])
        return out
    base = stats(x0 & ((1 << E) - 1))
    rows_out = [dict(flip="none", **base)]
    small = [p for p in primes if p < 60]
    for t, pool, name in [(1, small, "1 small"), (2, small, "2 small"), (1, primes[len(primes)//2:], "1 large"), (5, primes[len(primes)//2:], "5 large"), (20, primes, "20 any")]:
        for rep in range(3):
            x = x0 & ((1 << E) - 1)
            for p in rnd.sample(pool, t): x ^= 1 << idx[p]
            rows_out.append(dict(flip=name, **stats(x)))
    return dict(k=k, N=N, half_fibre=N // 2, results=rows_out)
for k in [int(a) for a in sys.argv[1:]]:
    r = run(k); print(json.dumps(r, ensure_ascii=False), flush=True)
