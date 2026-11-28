"""核对 ℓ = 0 的 Fourier 公式（纯 Bohr 系统，相邻互素，q | a_{t-2}，gcd(q, a_{t-1} a_t) = 1）：
P(q | M 或 q | M+1) = 2/q + Σ_{k≠0} [q·sin(2π a_t k/q)/(2π a_t k)]·[sin(2πk/q)/(πk)]。
注意 Fourier 公式对应开条件 ‖(P/q)θ‖ < 1/q，按区间测度计算，与"区间端点"计数一致。"""
import math, random, json
from m3_e49_coprime import gen_seq, strings, prime_factors
def fourier(q, at, K=200000):
    s = 2 / q
    for k in range(1, K + 1):
        x = 2 * math.pi * at * k / q
        s += 2 * (q * math.sin(x) / x) * (math.sin(2 * math.pi * k / q) / (math.pi * k))
    return s
rng = random.Random(31); rows = []; maxerr = 0
for it in range(200):
    t = rng.randint(3, 6); a = gen_seq(rng, t)
    for q0 in prime_factors(a[t - 2]):
        if a[t - 1] % q0 == 0 or a[t] % q0 == 0: continue
        q = q0
        pure = strings(a, False); R = len(pure)
        P = 1
        for x in a: P *= x
        # 按测度：θ ∈ (M/P,(M+1)/P)，事件 ‖(P/q)θ‖ < 1/q ⟺ 区间 (M,M+1) 的某个端点 ≡ 0 (mod q)
        ex = sum(1 for M, _, _ in pure if M % q == 0 or (M + 1) % q == 0) / R
        fo = fourier(q, a[t])
        maxerr = max(maxerr, abs(ex - fo)); rows.append((a, q, round(ex, 5), round(fo, 5)))
print(json.dumps(dict(cases=len(rows), max_abs_diff=round(maxerr, 6))))
for r in rows[:6]: print(r)
