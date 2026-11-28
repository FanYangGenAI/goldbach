"""探索性核对（非预注册）：G3 的关键恒等式。
设 q 为素数、k ≥ 1，i₀ = 满足 v_q(a_i⋯a_t) ≥ k 的最大 i（i₀ ≤ t − 1）。若 q ∤ a_{i₀+1}，则 M ≡ X_{i₀} (mod q^k)，
其中 X_j = M^{(j+2)} − ε_j T_{j+1}（与 a_j、a_{j+1} 无关）。"""
import random, json
from m3_e49_coprime import gen_seq, strings, prime_factors
def prod(xs):
    r = 1
    for x in xs: r *= x
    return r
def vq(n, q):
    v = 0
    while n % q == 0: n //= q; v += 1
    return v
rng = random.Random(5354); checks = viol = cases = 0; skipped = 0
for _ in range(400):
    t = rng.randint(3, 8); a = gen_seq(rng, t)
    T = [prod(a[k + 1:]) for k in range(t + 1)]
    Mt = lambda c, i: sum(c[kk] * T[kk] for kk in range(i, t + 1)) if i <= t else 0
    S = None
    qs = set()
    for x in a: qs |= prime_factors(x)
    for q in qs:
        vtot = sum(vq(x, q) for x in a)
        for k in range(2, vtot + 1):
            i0 = max(i for i in range(t + 1) if sum(vq(a[ii], q) for ii in range(i, t + 1)) >= k)
            if i0 > t - 1: continue
            if a[i0 + 1] % q == 0: skipped += 1; continue
            cases += 1
            if S is None: S = strings(a, True)
            for (M, eps, c) in S:
                X = Mt(c, i0 + 2) - eps[i0] * T[i0 + 1]
                checks += 1
                if (M - X) % (q ** k): viol += 1
print(json.dumps(dict(cases_q_k=cases, skipped_q_divides_next=skipped, checks=checks, viol=viol)))
