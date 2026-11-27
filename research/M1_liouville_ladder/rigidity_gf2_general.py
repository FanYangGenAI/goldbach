"""M1-E2 推广：任意偶数 N 的奇数子纤维刚性 g(n)g(N−n)=η（n 奇，n<N），g 完全积性 ±1。
求 GF(2) 解空间，限制到与 N 互素的小素数 q < B 后，检查是否落在"模 N 的实特征"张成的空间里。
候选张成：χ_{−4}、χ_8（导子整除 8），以及 N 的每个奇素因子 p 的 Legendre 符号。"""
import sys, json
from rigidity_gf2_nullspace import nullspace
def primes_upto(n):
    s = bytearray([1]) * (n + 1); s[0] = s[1] = 0
    for i in range(2, int(n ** .5) + 1):
        if s[i]: s[i*i::i] = bytearray(len(s[i*i::i]))
    return [p for p in range(2, n + 1) if s[p]]
def factor(n):
    f, d = [], 2
    while d * d <= n:
        if n % d == 0:
            f.append(d)
            while n % d == 0: n //= d
        d += 1
    if n > 1: f.append(n)
    return f
def run(N, B=150):
    ps = [p for p in primes_upto(N) if p > 2]
    idx = {p: i for i, p in enumerate(ps)}; E = len(ps)
    spf = list(range(N + 1))
    for i in range(2, int(N ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j: spf[j] = i
    def vec(n):
        v = 0
        while n > 1:
            q = spf[n]; c = 0
            while n % q == 0: n //= q; c ^= 1
            if c and q != 2: v ^= 1 << idx[q]
        return v
    rows = [vec(n) ^ vec(N - n) ^ (1 << E) for n in range(1, N // 2 + 1, 2) if True]
    basis = nullspace(rows, E + 1)
    fac = factor(N)
    small = [q for q in ps if q < B and N % q != 0]
    gens = []
    gens.append([((q % 4) == 3) * 1 for q in small])          # χ_{−4}
    gens.append([(q % 8 in (3, 5)) * 1 for q in small])        # χ_8
    for p in fac:
        if p > 2: gens.append([(pow(q, (p - 1) // 2, p) == p - 1) * 1 for q in small])
    def span(vs):
        s = {tuple([0] * len(small))}
        for v in vs: s |= {tuple(a ^ b for a, b in zip(v, t)) for t in s}
        return s
    char_span = span([tuple(g) for g in gens])
    restricted = span([tuple((x >> idx[q]) & 1 for q in small) for x in basis])
    lam_small = tuple([1] * len(small))
    return dict(N=N, factors=fac, dim=len(basis), restricted_dim=len(restricted).bit_length() - 1,
                char_span_dim=len(char_span).bit_length() - 1, restricted_subset_of_char_span=restricted <= char_span,
                lambda_pattern_in_restricted=lam_small in restricted)
Ns = [int(a) for a in sys.argv[1:]]
for N in Ns: print(json.dumps(run(N)), flush=True)
