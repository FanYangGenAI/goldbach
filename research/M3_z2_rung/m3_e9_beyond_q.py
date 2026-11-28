"""M3-E9（预注册，DESIGN 第 8 节）：从"翻转 q"出发做局部搜索，看 ρ 能否明显超过 q。"""
import sys, json, random, time
from m3_e8_ratio import primes_upto
def run(N, q, seed):
    rng = random.Random(seed); P = [p for p in primes_upto(N) if p != N]; L = q * N + 1
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
    rowf = [fac(n) + fac(N - n) for n in range(1, (N + 1) // 2)]
    tgtf = [(fac(q * n), fac((q * n) % N)) for n in range(1, N)]
    chi = {p: (1 if pow(p, (N - 1) // 2, N) == 1 else -1) for p in P}
    def ev(g, fs):
        s = 1
        for p in fs: s *= g[p]
        return s
    def rho(g):
        vals = [ev(g, f) for f in rowf]; eta = 1 if sum(vals) >= 0 else -1
        E = sum(1 for v in vals if v != eta)
        Eq = sum(1 for a, b in tgtf if ev(g, a) != ev(g, b))
        return Eq / E if E else 0.0
    g0 = dict(chi); g0[q] = -g0[q]; r0 = rho(g0)
    best = r0
    for _ in range(10):
        g = dict(g0); cur = r0
        for _ in range(600):
            p = rng.choice(P); g[p] = -g[p]; new = rho(g)
            if new >= cur: cur = new
            else: g[p] = -g[p]
        best = max(best, cur)
    return dict(N=N, q=q, rho_flip_q=round(r0, 3), localsearch_from_flip_q_max=round(best, 3), q_plus_5=q + 5)
if __name__ == "__main__":
    for N in (1009, 4093):
        for q in (31, 67):
            t0 = time.time(); r = run(N, q, N * q); r["sec"] = round(time.time() - t0, 1); print(json.dumps(r), flush=True)
