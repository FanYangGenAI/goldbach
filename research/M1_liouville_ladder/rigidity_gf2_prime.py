"""对照：素数 p 的完整纤维 g(m)g(p−m)=η（1≤m<p）对完全积性 g 的 GF(2) 解空间。
Mangerel（arXiv:2404.12117）对 λ 证明了：刚性 ⇒ λ = χ_p（Legendre 符号）于 [1,p)。
这里检查：对一般完全积性 ±1 函数，解空间是否只有 {g≡1, g=χ_p}（各自配合合适的 η）。"""
import sys, json
def primes_upto(n):
    s = bytearray([1]) * (n + 1); s[0] = s[1] = 0
    for i in range(2, int(n ** .5) + 1):
        if s[i]: s[i*i::i] = bytearray(len(s[i*i::i]))
    return [p for p in range(2, n + 1) if s[p]]
def run(p):
    ps = [q for q in primes_upto(p - 1)]
    idx = {q: i for i, q in enumerate(ps)}; E = len(ps)
    spf = list(range(p + 1))
    for i in range(2, int(p ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, p + 1, i):
                if spf[j] == j: spf[j] = i
    def vec(n):
        v = 0
        while n > 1:
            q = spf[n]; c = 0
            while n % q == 0: n //= q; c ^= 1
            if c: v ^= 1 << idx[q]
        return v
    piv = {}
    for m in range(1, (p - 1) // 2 + 1):
        r = vec(m) ^ vec(p - m) ^ (1 << E)
        while r:
            h = r.bit_length() - 1
            if h in piv: r ^= piv[h]
            else: piv[h] = r; break
    ok = lambda x: all(bin(r & x).count("1") % 2 == 0 for r in piv.values())
    leg = sum(1 << idx[q] for q in ps if pow(q, (p - 1) // 2, p) == p - 1)
    return dict(p=p, n_vars=E + 1, rank=len(piv), dim=E + 1 - len(piv),
                trivial=[e for e in (0, 1) if ok(e << E)], legendre=[e for e in (0, 1) if ok(leg | (e << E))])
for p in [int(a) for a in sys.argv[1:]]:
    print(json.dumps(run(p)), flush=True)
