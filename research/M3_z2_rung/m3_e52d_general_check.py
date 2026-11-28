"""核对第 70 节的一般 ℓ 恒等式：q | a_j，u = a_{j+2}S_{j+2}，K = K_{j+2}，
 (a) M ≡ u(K − ε_j a_{j+3}) + I (mod q)，I = −ε_{j+2}S_{j+2} + M^{(j+4)}；
 (b) uK + I ≡ M^{(j+5)} − ε_{j+3}S_{j+3} (mod a_{j+3})。
M^{(i)} = Σ_{k≥i} c_k T_k，T_k = a_{k+1}⋯a_t，S_i = a_{i+2}⋯a_t。"""
import random, json
from m3_e49_coprime import gen_seq, strings, prime_factors
def prod(xs):
    r = 1
    for x in xs: r *= x
    return r
rng = random.Random(70); tot = ba = bb = 0; seqs = 0
while seqs < 120:
    ell = rng.randint(3, 5); t = rng.randint(ell + 2, ell + 3); a = gen_seq(rng, t); j = t - 2 - ell
    qs = [q for q in prime_factors(a[j]) if all(a[i] % q for i in range(j + 1, t + 1))]
    if not qs: continue
    seqs += 1
    T = [prod(a[k + 1:]) for k in range(t + 1)]
    S = lambda i: prod(a[i + 2:])
    Mtail = lambda c, i: sum(c[k] * T[k] for k in range(i, t + 1)) if i <= t else 0
    for q in qs:
        u = a[j + 2] * S(j + 2)
        for M, eps, c in strings(a, True):
            num = a[j + 3] * c[j + 2] + c[j + 3] + eps[j + 2]; assert num % a[j + 2] == 0
            K = num // a[j + 2]
            I = -eps[j + 2] * S(j + 2) + Mtail(c, j + 4)
            tot += 1
            if (M - (u * (K - eps[j] * a[j + 3]) + I)) % q: ba += 1
            rhs = Mtail(c, j + 5) - (eps[j + 3] * S(j + 3) if j + 3 < t else 0)
            if (u * K + I - rhs) % a[j + 3]: bb += 1
print(json.dumps(dict(sequences=seqs, strings=tot, viol_a=ba, viol_b=bb)))
