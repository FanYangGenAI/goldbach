"""M3-E49b（探索性，非预注册）：互素情形下 P(r^e | M(M+1)) 与 2(1/r^e + 1/a_t) 之比的最大值，
按 ℓ = t - j - 2 分组（j 为 r | a_j 的最大下标）；含素数幂 r^e | P。"""
import random, json, math
from m3_e49_coprime import gen_seq, strings, prime_factors
rng = random.Random(777); by = {}; worst = []
for it in range(500):
    t = rng.randint(3, 9); a = gen_seq(rng, t); rel = strings(a, True)
    if not rel: continue
    R = len(rel); P = 1
    for x in a: P *= x
    for r in sorted(set().union(*[prime_factors(x) for x in a])):
        j = max(k for k in range(t + 1) if a[k] % r == 0)
        if j >= t - 1: continue   # j = t-1 系统性，j = t 为末位本身
        v = 0; x = P
        while x % r == 0: x //= r; v += 1
        for e in range(1, v + 1):
            q = r ** e
            pr = sum(1 for M, _, _ in rel if M % q == 0 or (M + 1) % q == 0) / R
            ratio = pr / (2 * (1 / q + 1 / a[t]))
            ell = t - j - 2
            b = by.setdefault(ell, [0, 0.0]); b[0] += 1; b[1] = max(b[1], ratio)
            if ratio > 1.5: worst.append(dict(a=a, r=r, e=e, j=j, P=round(pr, 4), ratio=round(ratio, 3)))
print(json.dumps({k: (c, round(m, 3)) for k, (c, m) in sorted(by.items())}))
for w in sorted(worst, key=lambda w: -w['ratio'])[:8]: print(json.dumps(w))
