"""M3-E8（预注册，DESIGN 第 7 节）：完整纤维上 ρ(g) = |𝓔_q(g)|/|𝓔(g)|。"""
import sys, json, random, time
def primes_upto(n):
    s = bytearray([1]) * (n + 1); s[0] = s[1] = 0
    for i in range(2, int(n ** .5) + 1):
        if s[i]: s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(n + 1) if s[i]]
def run(N, q, seed):
    rng = random.Random(seed); P = primes_upto(N); L = q * N + 1
    spf = list(range(L + 1))
    for i in range(2, int(L ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, L + 1, i):
                if spf[j] == j: spf[j] = i
    def fac(n):
        f = []
        while n > 1:
            p = spf[n]; c = 0
            while n % p == 0: n //= p; c ^= 1
            if c: f.append(p)
        return f
    rowf = [(fac(n), fac(N - n)) for n in range(1, (N + 1) // 2)]
    tgtf = [(fac(q * n), fac((q * n) % N)) for n in range(1, N)]
    # χ_N（Legendre 符号）在素数上的值
    chi = {p: (1 if pow(p, (N - 1) // 2, N) == 1 else -1) for p in P if p != N}
    def ev(g, fs):
        s = 1
        for p in fs: s *= g.get(p, 1)
        return s
    def counts(g):
        vals = [ev(g, a) * ev(g, b) for a, b in rowf]
        eta = 1 if sum(vals) >= 0 else -1
        E = sum(1 for v in vals if v != eta)
        Eq = sum(1 for a, b in tgtf if ev(g, a) != ev(g, b))
        return E, Eq
    def rho(g):
        E, Eq = counts(g); return (Eq / E if E else (0.0 if Eq == 0 else float("inf"))), E, Eq
    out = dict(N=N, q=q)
    lam = {p: -1 for p in P}; r, E, Eq = rho(lam); out["lambda"] = dict(rho=round(r, 3), E=E, Eq=Eq)
    rs = [rho({p: rng.choice((-1, 1)) for p in P})[0] for _ in range(100)]; out["random_max"] = round(max(rs), 3)
    for k in (1, 3, 10, 30):
        mx = 0.0
        for _ in range(30):
            g = dict(chi)
            for p in rng.sample(P[:-1] if P[-1] == N else P, k): g[p] = -g.get(p, 1)
            mx = max(mx, rho(g)[0])
        out[f"nearchar_k{k}_max"] = round(mx, 3)
    best = 0.0; cand = [p for p in P if p != N]
    for _ in range(10):
        g = dict(chi); p0 = rng.choice(cand); g[p0] = -g[p0]; cur = rho(g)[0]
        for _ in range(200):
            p = rng.choice(cand); g[p] = -g[p]; new = rho(g)[0]
            if new >= cur and new != float("inf"): cur = new
            else: g[p] = -g[p]
        best = max(best, cur)
    out["localsearch_max"] = round(best, 3)
    return out
if __name__ == "__main__":
    for N in (1009, 4093):
        for q in (3, 31, 67):
            t0 = time.time(); r = run(N, q, N + q); r["sec"] = round(time.time() - t0, 1); print(json.dumps(r), flush=True)
