"""M3-E49f（探索性）：实际脊柱（1024 位 N）上，斜率 A = −(a_t a_{t−2})^{−1} mod a_{t−1} 的分布（A/a_{t−1} 分 20 格），
只统计 gcd 条件满足（a_t a_{t−2} 与 a_{t−1} 互素）的三元组；并统计 A ≡ ±1 (mod a_{t−1}) 的频率。"""
import sys, json, math, random
import gmpy2
from gmpy2 import mpz
from m3_e47_large_N import thue_cf
rr = random.Random(20261128)
N = gmpy2.next_prime(mpz(rr.getrandbits(1024)) | (mpz(1) << 1023))
for p in map(int, sys.argv[1].split(",")):
    rng = random.Random(p); lo = (N + p - 1) // p; S = int(sys.argv[2]); bins = [0] * 20; tot = 0; pm1 = 0; small_m = 0; exp_pm1 = 0.0
    for _ in range(S):
        n = mpz(rng.randrange(lo, N)); a, x = p, n; seq = [p]
        while a > 1 and a * x >= N:
            v = thue_cf(a, x, N); a, x = v, (a * x) % N
            if a >= 2: seq.append(a)
        for t in range(2, len(seq)):
            m = seq[t - 1]; u = seq[t] * seq[t - 2]
            if m < 5 or math.gcd(u, m) != 1: small_m += (m < 5); continue
            A = (-pow(u, -1, m)) % m; tot += 1
            bins[min(19, int(20 * A / m))] += 1
            pm1 += (A in (1, m - 1)); phi = sum(1 for k in range(1, m) if math.gcd(k, m) == 1); exp_pm1 += 2 / phi
    mean = tot / 20
    print(json.dumps(dict(p=p, triples=tot, max_bin_over_mean=round(max(bins) / mean, 3), min_bin_over_mean=round(min(bins) / mean, 3),
                          frac_A_pm1=round(pm1 / tot, 4), uniform_units_expectation=round(exp_pm1 / tot, 4), bins=bins)), flush=True)
