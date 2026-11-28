"""M3-E49g（探索性）：对随机相邻互素序列平均，按尾数长度 ℓ 统计 E[(P(q|M(M+1)) − 2/q)·a_t] 与 E[P·q/2]
（q 为素数，j = j(q) ≤ t − 2，q ≥ 3）。引理 T″ 的平均版本预言前者有界。"""
import random, json
from m3_e49_coprime import gen_seq, strings, prime_factors
rng = random.Random(2468); by = {}
for it in range(700):
    t = rng.randint(3, 9); a = gen_seq(rng, t); rel = strings(a, True)
    if not rel: continue
    R = len(rel)
    for q in sorted(set().union(*[prime_factors(x) for x in a])):
        if q < 3: continue
        j = max(k for k in range(t + 1) if a[k] % q == 0)
        if j > t - 2: continue
        pr = sum(1 for M, _, _ in rel if M % q == 0 or (M + 1) % q == 0) / R
        ell = t - j - 2
        b = by.setdefault(ell, [0, 0.0, 0.0, 0.0]); b[0] += 1; b[1] += (pr - 2 / q) * a[t]; b[2] += pr * q / 2; b[3] = max(b[3], (pr - 2 / q) * a[t])
print(json.dumps({k: dict(n=c, mean_excess_times_at=round(s1 / c, 3), mean_ratio=round(s2 / c, 3), max_excess_times_at=round(m, 2)) for k, (c, s1, s2, m) in sorted(by.items())}))
