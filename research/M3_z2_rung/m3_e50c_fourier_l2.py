"""核对引理 F 在 ℓ = 2 时的情形（截断较粗，只作量级核对）。内层对 μ₁ 的求和用恒等式
直接数值求和（μ₁ 截断到 ±W）。"""
import math, random, json
import numpy as np
from m3_e49_coprime import strings, prime_factors, is_prime
def sinc(x):
    x = np.asarray(x, dtype=float); out = np.ones_like(x); nz = x != 0
    out[nz] = np.sin(2 * np.pi * x[nz]) / (2 * np.pi * x[nz]); return out
def F_l2(a, q, K=60, W=20000):
    t = len(a) - 1; at, a1, a2 = a[t], a[t - 1], a[t - 2]
    Sj = a2 * a1 * at
    mu1 = np.arange(-W, W + 1); s1 = sinc(mu1 / a2)
    total = 2 / q
    for k in range(-K, K + 1):
        gk = 2 / q if k == 0 else math.sin(2 * math.pi * k / q) / (math.pi * k)
        if abs(gk) < 1e-15: continue
        for mu2 in range(-4 * a1, 4 * a1 + 1):
            if k == 0 and mu2 == 0: continue
            x0 = -k * Sj / q - mu2 * at * a2
            d = x0 - mu1 * a1
            inner = float(np.sum(s1 * sinc(d)))
            total += gk * inner * float(sinc(np.array([mu2 / a1]))[0])
    return total
rng = random.Random(5); rows = []
cands = []
for it in range(4000):
    t = rng.randint(4, 5)
    p = rng.choice([x for x in range(31, 200) if is_prime(x)]); a = [p]; ok = True
    for i in range(t):
        c = [b for b in range(max(3, a[-1] // 3), a[-1]) if math.gcd(b, a[-1]) == 1]
        if not c: ok = False; break
        a.append(rng.choice(c))
    if not ok or a[-2] > 30: continue
    for q in prime_factors(a[t - 4]):
        if q > 2 and not any(a[i] % q == 0 for i in range(t - 3, t + 1)):
            cands.append((a, q))
    if len(cands) >= 8: break
for a, q in cands[:8]:
    pure = strings(a, False); R = len(pure)
    ex = sum(1 for M, _, _ in pure if M % q == 0 or (M + 1) % q == 0) / R
    rows.append((a, q, round(ex, 4), round(F_l2(a, q), 4)))
print(json.dumps(dict(cases=len(rows), max_abs_diff=round(max(abs(r[2] - r[3]) for r in rows), 4))))
for r in rows: print(r)
