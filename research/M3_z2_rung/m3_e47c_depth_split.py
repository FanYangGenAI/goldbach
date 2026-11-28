"""M3-E47c（探索性，非预注册）：Λ* 按深度拆分，直接测命题 D′ 的量 Σ_{t>=3} E[log indeg*(σ_t)]。"""
import sys, json, math, random
import gmpy2
from gmpy2 import mpz
from m3_e47_large_N import Forest, thue_cf
rr = random.Random(20261128)
N = gmpy2.next_prime(mpz(rr.getrandbits(1024)) | (mpz(1) << 1023))
for spec in sys.argv[1].split(","):
    p, S = map(int, spec.split(":"))
    rng = random.Random(p); F = Forest(N, p); lo = (N + p - 1) // p; by = {}
    for _ in range(S):
        n = mpz(rng.randrange(lo, N)); a, x = p, n; path = [(a, x)]
        while a > 1 and a * x >= N:
            v = thue_cf(a, x, N); a, x = v, (a * x) % N; path.append((a, x))
        for t in range(1, len(path)):
            a, x = path[t]
            if a < 2: continue
            d = sum(1 for pp in F.parents(a, x) if F.is_sn(*pp))
            by[t] = by.get(t, 0.0) + math.log(d)
    lp = math.log(p)
    deep = sum(v for t, v in by.items() if t >= 3) / S
    print(json.dumps(dict(p=p, S=S, per_depth={t: round(v / S, 4) for t, v in sorted(by.items())},
                          shallow_t12=round((by.get(1, 0) + by.get(2, 0)) / S, 4), deep_t3plus=round(deep, 4),
                          deep_over_lnp=round(deep / lp, 4), deep_over_lnlnp=round(deep / math.log(lp), 4))), flush=True)
