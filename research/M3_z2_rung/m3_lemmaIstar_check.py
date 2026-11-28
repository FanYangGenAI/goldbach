"""引理 I* 的独立核对（有理数区间精确计算，与数字自动机无关）：
相邻互素时，四个条件（深度 4）的测度恰为乘积；不互素时记录比值。"""
from fractions import Fraction as Fr
import random, json, math
from m3_lemmaI_check import intervals, inter, meas
rng = random.Random(5); co = [0, 0]; nonco = []
primes = [p for p in range(23, 70) if all(p % d for d in range(2, p))]
for _ in range(300):
    p = rng.choice(primes); a = [p]
    for _ in range(4):
        if a[-1] <= 2: break
        a.append(rng.randrange(2, a[-1]))
    if len(a) < 5: continue
    S = [(Fr(0), Fr(1))]; Pm = 1; prod = Fr(1)
    for i in range(4):
        S = inter(S, intervals(a[i + 1] * Pm, Fr(1, a[i]))); prod *= Fr(2, a[i]); Pm *= a[i]
    coprime = all(math.gcd(a[i], a[i + 1]) == 1 for i in range(4))
    if coprime:
        co[0] += 1; co[1] += (meas(S) != prod)
    else:
        nonco.append(float(meas(S) / prod))
print(json.dumps(dict(coprime_cases=co[0], coprime_not_exact=co[1], noncoprime_cases=len(nonco),
                      noncoprime_ratio_min=round(min(nonco), 3), noncoprime_ratio_max=round(max(nonco), 3),
                      noncoprime_exact_fraction=round(sum(1 for x in nonco if abs(x - 1) < 1e-12) / len(nonco), 3))))
