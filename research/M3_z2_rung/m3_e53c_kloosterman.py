"""探索性核对（非预注册）：引理 W 用到的两个标准界的显式常数。
(1) Weil–Estermann：|S(α, β; Q)| ≤ τ(Q)·gcd(α, β, Q)^{1/2}·Q^{1/2}；
(2) 不完全和：区间 I 长度 ≤ Q 时 |Σ_{b∈I, (b,Q)=1} e(α b̄/Q)| ≤ τ(Q)√(gcd(α,Q)·Q)·(2 + log Q)。"""
import random, json, math, cmath
from sympy import divisor_count
rng = random.Random(531)
def e(x): return cmath.exp(2j * math.pi * x)
w1 = w2 = 0; r1 = r2 = 0.0; n1 = n2 = 0
for _ in range(400):
    Q = rng.randint(2, 1500); tau = int(divisor_count(Q))
    units = [b for b in range(1, Q) if math.gcd(b, Q) == 1] if Q > 1 else [0]
    inv = {b: pow(b, -1, Q) for b in units}
    for _ in range(4):
        al = rng.randrange(1, Q) if Q > 1 else 0; be = rng.randrange(Q)
        S = abs(sum(e((al * inv[b] + be * b) / Q) for b in units))
        B = tau * math.sqrt(math.gcd(math.gcd(al, be), Q) * Q); n1 += 1
        r1 = max(r1, S / B); w1 += S > B * (1 + 1e-9)
        L = rng.randint(1, Q); s0 = rng.randrange(Q)
        T = abs(sum(e(al * inv[b % Q] / Q) for b in range(s0, s0 + L) if math.gcd(b, Q) == 1))
        B2 = tau * math.sqrt(math.gcd(al, Q) * Q) * (2 + math.log(Q)); n2 += 1
        r2 = max(r2, T / B2); w2 += T > B2 * (1 + 1e-9)
print(json.dumps(dict(complete_checks=n1, complete_viol=w1, complete_max_ratio=round(r1, 4),
                      incomplete_checks=n2, incomplete_viol=w2, incomplete_max_ratio=round(r2, 4))))
