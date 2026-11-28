"""引理 I 的精确核对：μ{θ : ‖aθ‖ < 1/p, ‖bpθ‖ < 1/a} = 4/(pa)（b < a < p，p 素数），用有理数区间精确计算。
另外核对三个条件时的偏差（非独立的例子）。"""
from fractions import Fraction as Fr
import random, json
def intervals(q, r):  # {θ ∈ [0,1): ‖qθ‖ < r}，r <= 1/2
    out = []
    for k in range(q + 1):
        lo, hi = Fr(k, q) - r / q, Fr(k, q) + r / q
        lo, hi = max(lo, Fr(0)), min(hi, Fr(1))
        if lo < hi: out.append((lo, hi))
    return out
def inter(A, B):
    i = j = 0; out = []
    while i < len(A) and j < len(B):
        lo, hi = max(A[i][0], B[j][0]), min(A[i][1], B[j][1])
        if lo < hi: out.append((lo, hi))
        if A[i][1] < B[j][1]: i += 1
        else: j += 1
    return out
meas = lambda S: sum(h - l for l, h in S)
rng = random.Random(1); bad = 0; tested = 0; maxdev3 = 0
primes = [p for p in range(11, 90) if all(p % d for d in range(2, p))]
for _ in range(400):
    p = rng.choice(primes); a = rng.randrange(3, p); b = rng.randrange(2, a)
    S = inter(intervals(a, Fr(1, p)), intervals(b * p, Fr(1, a)))
    tested += 1; bad += (meas(S) != Fr(4, p * a))
    c = rng.randrange(2, b) if b > 2 else None
    if c:
        S3 = inter(S, intervals(p * a * c, Fr(1, b)))
        dev = meas(S3) / (Fr(8, p * a * b)); maxdev3 = max(maxdev3, float(dev))
print(json.dumps(dict(pairs_tested=tested, pairs_not_exact=bad, max_ratio_triple_to_product=round(maxdev3, 3))))
