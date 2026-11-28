"""M3-E51：引理 F 离散版本的精确核对（预注册：DESIGN.md 第 35 节）。
离散公式（RESULTS 第 67 节）：
  P(q | M(M+1)) = Σ_{s∈Z/q} Σ_{r_l∈Z/R_l, l=j+2..t-1} Ĝ(s)·F̂_j(ρ_j)·Π_l F̂_l(r_l) / [(2/a_j)·Π_l (2/a_l)]，
其中 F̂_i(r) = (1/R_i)Σ_{x=-S_i}^{S_i-1} e(-rx/R_i)，Ĝ(s) = (1 + e(s/q))/q，
λ ≡ -Σ_l r_l w_l (mod S_j)，ρ_j = -s(a_j/q)S_j + a_j λ (mod R_j)，w_l = a_{l+1}·a_{j+2}⋯a_{l-1}。"""
import cmath, math, random, json, itertools
from m3_e49_coprime import strings, prime_factors, is_prime

def prod(xs):
    r = 1
    for x in xs: r *= x
    return r

def Fhat(r, S, R):
    r %= R
    if r == 0: return 2 * S / R
    z = cmath.exp(-2j * math.pi * r / R)
    return (cmath.exp(2j * math.pi * r * S / R) * (1 - z ** (2 * S)) / (1 - z)) / R

def formula(a, q, j):
    t = len(a) - 1
    S = lambda i: prod(a[i + 2:t + 1])
    R = lambda i: a[i] * S(i)
    Sj, Rj = S(j), R(j)
    ls = list(range(j + 2, t))
    w = {l: a[l + 1] * prod(a[j + 2:l]) for l in ls}
    norm = (2 / a[j]) * prod([2 / a[l] for l in ls])
    total = 0j
    ranges = [range(R(l)) for l in ls]
    pre = {l: [Fhat(r, S(l), R(l)) for r in range(R(l))] for l in ls}
    for s in range(q):
        G = (1 + cmath.exp(2j * math.pi * s / q)) / q
        for rs in itertools.product(*ranges):
            lam = (-sum(r * w[l] for r, l in zip(rs, ls))) % Sj
            rho = (-s * (a[j] // q) * Sj + a[j] * lam) % Rj
            term = G * Fhat(rho, Sj, Rj)
            for r, l in zip(rs, ls): term *= pre[l][r]
            total += term
    return total / norm

def gen(rng, t, amax):
    p = rng.choice([x for x in range(11, amax + 1) if is_prime(x)]); a = [p]
    for _ in range(t):
        c = [b for b in range(max(3, a[-1] // 3), a[-1]) if math.gcd(b, a[-1]) == 1]
        if not c: return None
        a.append(rng.choice(c))
    return a

rng = random.Random(51); rows = []; maxdiff = 0.0; maxim = 0.0
targets = {0: 40, 1: 25, 2: 12}
got = {0: 0, 1: 0, 2: 0}
tries = 0
while any(got[l] < targets[l] for l in targets) and tries < 20000:
    tries += 1
    ell = rng.choice([l for l in targets if got[l] < targets[l]])
    t = rng.randint(ell + 2, ell + 4)
    a = gen(rng, t, 1000 if ell < 2 else 60)
    if not a: continue
    j = t - 2 - ell
    qs = [q for q in prime_factors(a[j]) if all(a[i] % q for i in range(j + 1, t + 1))]
    if not qs: continue
    q = rng.choice(qs)
    pure = strings(a, False); Rn = len(pure)
    ex = sum(1 for M, _, _ in pure if M % q == 0 or (M + 1) % q == 0) / Rn
    fo = formula(a, q, j)
    d = abs(ex - fo.real); maxdiff = max(maxdiff, d); maxim = max(maxim, abs(fo.imag))
    rows.append(dict(ell=ell, a=a, q=q, exact=round(ex, 6), formula=round(fo.real, 6))); got[ell] += 1
print(json.dumps(dict(cases=len(rows), by_ell=got, max_abs_diff=maxdiff, max_abs_imag=maxim)))
for r in rows[:3] + [x for x in rows if x['ell'] == 1][:2] + [x for x in rows if x['ell'] == 2][:3]: print(json.dumps(r))
