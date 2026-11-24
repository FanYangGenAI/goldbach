"""O6 (preregistered): parameter-free refined local model for chi_{N,z} that adds the local factors of the
primes p > z dividing N (missing from the O1 base model).  For p | N, p > z, a z-rough semiprime s ~ t is
divisible by p with probability w_p(t) = [1/(p log(t/p))] * 1{t/p > z} / rho2(t); primes essentially never.
With residues mod p uniform on the units otherwise, the pair factor relative to PP is
    PS: 1 - w(N-t),  SP: 1 - w(t),  SS: (p-1) w(t) w(N-t) + (1-w(t))(1-w(N-t)).
X_pred = S(N) * int rho_X(t) rho_Y(N-t) prod_p gamma_p^{XY}(t) dt.  See DESIGN.md.  Usage: python o6_chi_refined.py [--smoke]"""
import json, math, os, sys, time
import numba as nb
import numpy as np
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv
PLAN = [(10**6, 100), (10**7, 100)] if SMOKE else [(10**8, 300), (10**9, 300)]
SEED = 2026112413
C2 = 0.6601618158468695739278121100145
t0 = time.time()


@nb.njit(cache=True)
def sieve(n):
    spf = np.zeros(n + 1, dtype=np.int32)
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            j = np.int64(i) * i
            while j <= n:
                if spf[j] == 0:
                    spf[j] = i
                j += i
    om = np.zeros(n + 1, dtype=np.int8)
    for m in range(2, n + 1):
        om[m] = om[m // spf[m]] + 1
    return spf, om


@nb.njit(cache=True)
def icbrt(n):
    z = int(round(n ** (1.0 / 3.0)))
    while z * z * z > n: z -= 1
    while (z + 1) * (z + 1) * (z + 1) <= n: z += 1
    return z


@nb.njit(parallel=True, cache=True)
def counts(Ns, spf, om):
    out = np.zeros((len(Ns), 4), dtype=np.int64)
    for k in nb.prange(len(Ns)):
        N = Ns[k]; z = icbrt(N)
        pp = 0; ps = 0; sp = 0; ss = 0
        for n in range(z + 1, N - z):
            if spf[n] <= z: continue
            m = N - n
            if spf[m] <= z: continue
            a = om[n]; b = om[m]
            if a == 1 and b == 1: pp += 1
            elif a == 1 and b == 2: ps += 1
            elif a == 2 and b == 1: sp += 1
            elif a == 2 and b == 2: ss += 1
        out[k, 0] = pp; out[k, 1] = ps; out[k, 2] = sp; out[k, 3] = ss
    return out


def odd_prime_factors(N):
    ps = []; m = N
    while m % 2 == 0: m //= 2
    p = 3
    while p * p <= m:
        if m % p == 0:
            ps.append(p)
            while m % p == 0: m //= p
        p += 2
    if m > 1: ps.append(m)
    return ps


def chi_from(pp, ps, sp, ss):
    A = pp + ps + sp + ss; B = -pp + ss + (ps - sp); C = pp - ps - sp + ss
    return C / A - (B / A) ** 2


def predict(N, z, big):
    lz = math.log(z)
    def r1(t): return 1 / math.log(t)
    def r2(t):
        L = math.log(t)
        return 0.0 if L <= 2 * lz else math.log((L - lz) / lz) / L
    def w(p, t):
        if t / p <= z: return 0.0
        d = r2(t)
        return 0.0 if d <= 0 else min(1.0, 1 / (p * math.log(t / p)) / d)
    def g(kind, t):
        v = 1.0
        for p in big:
            a, b = w(p, t), w(p, N - t)
            if kind == 1: v *= 1 - b
            elif kind == 2: v *= 1 - a
            elif kind == 3: v *= (p - 1) * a * b + (1 - a) * (1 - b)
        return v
    funcs = [(r1, r1, 0), (r1, r2, 1), (r2, r1, 2), (r2, r2, 3)]
    lo, hi = z + 1, N - z - 1
    pts = sorted({min(max(x, lo), hi) for p in big for x in (p * z, N - p * z)} | {N / 2})
    res_base, res_ref = [], []
    for f, h, kind in funcs:
        edges = [lo] + [x for x in pts if lo < x < hi] + [hi]
        Ib = Ir = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            Ib += integrate.quad(lambda t: f(t) * h(N - t), a, b, limit=400)[0]
            Ir += integrate.quad(lambda t: f(t) * h(N - t) * g(kind, t), a, b, limit=400)[0]
        res_base.append(Ib); res_ref.append(Ir)
    return chi_from(*res_base), chi_from(*res_ref)


def centre(v, g):
    v = np.asarray(v, float).copy()
    for gg in (True, False):
        if (g == gg).any(): v[g == gg] -= v[g == gg].mean()
    return v


NMAX = int(1.1 * max(X for X, _ in PLAN)) + 10
spf, om = sieve(NMAX)
print(f"sieve to {NMAX:,}: {time.time()-t0:.0f}s", flush=True)
# exclude N used by O1 (seed 2026112411, 20+20 per scale) and O4 (seed 2026112412) samples
used = set()
r_ = np.random.default_rng(2026112411)
for X in [10**6, 10**7, 10**8, 10**9]:
    g3, g0 = [], []
    while len(g3) < 20 or len(g0) < 20:
        N = int(2 * r_.integers(X // 2, int(1.1 * X) // 2))
        if N in g3 or N in g0: continue
        if N % 3 == 0 and len(g3) < 20: g3.append(N)
        elif N % 3 != 0 and len(g0) < 20: g0.append(N)
    used |= set(g3 + g0)
r_ = np.random.default_rng(2026112412)
for X, K in [(10**7, 1000), (10**8, 400), (10**9, 300)]:
    s = set()
    while len(s) < K: s.add(int(2 * r_.integers(X // 2, int(1.1 * X) // 2)))
    used |= s
rng = np.random.default_rng(SEED)
rows = []; summ = {}
for X, K in PLAN:
    Ns = set()
    while len(Ns) < K:
        N = int(2 * rng.integers(X // 2, int(1.1 * X) // 2))
        if N not in used: Ns.add(N)
    Ns = np.array(sorted(Ns), dtype=np.int64)
    Cn = counts(Ns, spf, om)
    obs, pb, pr = [], [], []
    for i, N in enumerate(Ns):
        N = int(N); z = int(icbrt(N))
        big = [p for p in odd_prime_factors(N) if p > z]
        cb, cr = predict(N, z, big)
        co = chi_from(*[int(x) for x in Cn[i]])
        obs.append(co); pb.append(cb); pr.append(cr)
        rows.append(dict(X=X, N=N, z=z, big=big, counts=[int(x) for x in Cn[i]], chi_obs=co, chi_base=cb, chi_ref=cr))
    obs, pb, pr = map(np.array, (obs, pb, pr))
    g = Ns % 3 == 0
    co, cp = centre(obs, g), centre(pr, g)
    co2, cp2 = obs - obs.mean(), pr - pr.mean()
    summ[str(X)] = dict(rho=float(np.corrcoef(co, cp)[0, 1]), sd_ratio=float(cp.std(ddof=1) / co.std(ddof=1)),
                        R2_pred=float(1 - np.sum((co - cp) ** 2) / np.sum(co**2)),
                        rho_plain=float(np.corrcoef(co2, cp2)[0, 1]), sd_ratio_plain=float(cp2.std(ddof=1) / co2.std(ddof=1)),
                        R2_plain=float(1 - np.sum((co2 - cp2) ** 2) / np.sum(co2**2)),
                        mean_obs=float(obs.mean()), mean_base=float(pb.mean()), mean_ref=float(pr.mean()),
                        sd_base_over_obs=float(pb.std() / obs.std()), secs=time.time() - t0)
    print(X, json.dumps({k: round(v, 5) for k, v in summ[str(X)].items()}), flush=True)
    json.dump(dict(summary=summ, rows=rows, smoke=SMOKE), open(os.path.join(OUT, "o6_smoke.json" if SMOKE else "o6_summary.json"), "w"), indent=1, allow_nan=False)
key = summ[str(max(X for X, _ in PLAN))]
verdict = "support" if (key["rho"] > 0.5 and 0.5 <= key["sd_ratio"] <= 2 and key["R2_pred"] > 0.25) else ("reject" if (key["rho"] < 0.2 or key["sd_ratio"] < 0.1) else "partial")
print("verdict:", verdict, flush=True)
json.dump(dict(summary=summ, verdict=verdict, rows=rows, smoke=SMOKE), open(os.path.join(OUT, "o6_smoke.json" if SMOKE else "o6_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
