"""探索性核对（非预注册）：G3 情形 B1 的条件约化。
j = i₀(q, k)，e = v_q(S) ∈ [1, k − 1]（S = a_{j+4}⋯a_t），且 q ∤ a_{j+2}, a_{j+3}。断言：
 q^k | M  ⟺  q^e | I  且  b(K − ε_j m) ≡ −(I/q^e)(S/q^e)^{−1} (mod q^{k−e})，其中 I = −ε_{j+2}S + M^{(j+4)}；
并且 q^e | I ⟺ q^e | M。"""
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
rng = random.Random(5355); cases = checks = v1 = v2 = 0; tries = 0
while cases < 300 and tries < 200000:
    tries += 1
    t = rng.randint(5, 9); a = gen_seq(rng, t)
    T = [prod(a[k + 1:]) for k in range(t + 1)]
    Mt = lambda c, i: sum(c[kk] * T[kk] for kk in range(i, t + 1)) if i <= t else 0
    qs = set()
    for x in a: qs |= {q for q in prime_factors(x) if q <= 7}
    S_all = None
    for q in qs:
        vtot = sum(vq(x, q) for x in a)
        for k in range(2, vtot + 1):
            j = max(i for i in range(t + 1) if sum(vq(a[ii], q) for ii in range(i, t + 1)) >= k)
            if j > t - 4: continue
            S0 = prod(a[j + 4:]); e = vq(S0, q); kp = k - e
            if e < 1 or a[j + 2] % q == 0 or a[j + 3] % q == 0: continue
            cases += 1
            if S_all is None: S_all = strings(a, True)
            m = a[j + 3]; b = a[j + 2]; S = S0; assert 1 <= kp <= vq(a[j], q)
            for (M, eps, c) in S_all:
                K = (a[j + 3] * c[j + 2] + c[j + 3] + eps[j + 2]) // b
                I = -eps[j + 2] * S + Mt(c, j + 4)
                lhs = (M % q ** k == 0)
                if I % q ** e: rhs = False
                else:
                    r = (-(I // q ** e) * pow(S // q ** e, -1, q ** kp)) % q ** kp
                    rhs = ((b * (K - eps[j] * m) - r) % q ** kp == 0)
                checks += 1; v1 += (lhs != rhs)
                v2 += ((I % q ** e == 0) != (M % q ** e == 0))
print(json.dumps(dict(cases_q_k=cases, checks=checks, viol_reduction=v1, viol_cond_equiv=v2)))
