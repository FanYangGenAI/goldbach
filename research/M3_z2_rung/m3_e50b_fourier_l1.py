"""核对 ℓ = 1 的 Fourier 公式（纯 Bohr，相邻互素，q | a_{t-3}，gcd(q, a_{t-2}a_{t-1}a_t) = 1）：
P(q | M(M+1)) = 2/q + Σ_{k≠0} Σ_{μ∈Z} sinc(a_t(k a_{t-1} + qμ)/q)·sinc(μ/a_{t-1})·sin(2πk/q)/(πk)，
sinc(x) = sin(2πx)/(2πx)。消元推导见 RESULTS 第 65 节。"""
import math, random, json
import numpy as np
from m3_e49_coprime import gen_seq, strings, prime_factors
def sinc(x):
    x = np.asarray(x, dtype=float); out = np.ones_like(x); nz = x != 0
    out[nz] = np.sin(2 * np.pi * x[nz]) / (2 * np.pi * x[nz]); return out
def fourier_l1(q, a1, at, K=600):
    s = 2 / q
    W = int(K * a1 / q) + 2000
    mu = np.arange(-W, W + 1)
    sm = sinc(mu / a1)
    for k in range(1, K + 1):
        inner = np.sum(sinc(at * (k * a1 + q * mu) / q) * sm)
        s += 2 * inner * math.sin(2 * math.pi * k / q) / (math.pi * k)   # k 与 -k 对称
    return s
rng = random.Random(77); rows = []; maxerr = 0
while len(rows) < 25:
    t = rng.randint(4, 6); a = gen_seq(rng, t)
    for q in prime_factors(a[t - 3]):
        if any(a[i] % q == 0 for i in range(t - 2, t + 1)): continue
        pure = strings(a, False); R = len(pure)
        ex = sum(1 for M, _, _ in pure if M % q == 0 or (M + 1) % q == 0) / R
        fo = fourier_l1(q, a[t - 1], a[t])
        maxerr = max(maxerr, abs(ex - fo)); rows.append((a, q, round(ex, 5), round(fo, 5)))
print(json.dumps(dict(cases=len(rows), max_abs_diff=round(maxerr, 5))))
for r in rows[:8]: print(r)
