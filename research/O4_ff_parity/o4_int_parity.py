"""O4 integer companion (preregistered): same statistics as o4_ff_parity.py for chi_{N,z} in the integers
(z = floor(N^{1/3}), definitions as in O1).  Larger samples than O1; computes the random-labelling dispersion D
and the R^2 of chi on local features of N.  See DESIGN.md.  Usage: python o4_int_parity.py [--smoke]"""
import json, math, os, sys, time
import numba as nb
import numpy as np
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv
PLAN = [(10**5, 200), (10**6, 200)] if SMOKE else [(10**7, 1000), (10**8, 400), (10**9, 300)]
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


def pair_integral(N, f, g, z):
    a, b = z + 1, N - z - 1
    I1, _ = integrate.quad(lambda t: f(t) * g(N - t), a, N / 2, limit=400)
    I2, _ = integrate.quad(lambda t: f(t) * g(N - t), N / 2, b, limit=400)
    return I1 + I2


def null_var_coef(p):
    c = np.array([1, -1, -1, 1.0]); b = np.array([-1, 1, -1, 1.0])
    cb, bb = (p * c).sum(), (p * b).sum()
    S = np.array([[(p * c * c).sum() - cb * cb, (p * c * b).sum() - cb * bb], [(p * c * b).sum() - cb * bb, (p * b * b).sum() - bb * bb]])
    g = np.array([1.0, -2 * bb])
    return float(g @ S @ g)


def r2(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return float(1 - res.var() / y.var())


NMAX = int(1.1 * max(X for X, _ in PLAN)) + 10
spf, om = sieve(NMAX)
print(f"sieve to {NMAX:,}: {time.time()-t0:.0f}s", flush=True)
rng = np.random.default_rng(2026112412)
rows = []
for X, K in PLAN:
    Ns = set()
    while len(Ns) < K:
        Ns.add(int(2 * rng.integers(X // 2, int(1.1 * X) // 2)))
    Ns = np.array(sorted(Ns), dtype=np.int64)
    Cn = counts(Ns, spf, om)
    chi = []; vnull = []; A = []; feats = []
    for i, N in enumerate(Ns):
        N = int(N); z = int(icbrt(N)); lz = math.log(z)
        def rho1(t): return 1 / math.log(t)
        def rho2(t):
            L = math.log(t)
            return 0.0 if L <= 2 * lz else math.log((L - lz) / lz) / L
        pc = np.array([pair_integral(N, rho1, rho1, z), pair_integral(N, rho1, rho2, z), pair_integral(N, rho2, rho1, z), pair_integral(N, rho2, rho2, z)])
        pc = pc / pc.sum()
        pp, ps, sp, ss = [int(x) for x in Cn[i]]
        a = pp + ps + sp + ss; b = -pp + ss + (ps - sp); c = pp - ps - sp + ss
        chi.append(c / a - (b / a) ** 2); A.append(a); vnull.append(null_var_coef(pc))
        ops = odd_prime_factors(N)
        logS = math.log(2 * C2) + sum(math.log((p - 1) / (p - 2)) for p in ops)
        Nodd = N
        while Nodd % 2 == 0: Nodd //= 2
        feats.append([logS, sum(1 for p in ops if p <= z), sum(1 for p in ops if p > z), float(len(ops) == 1 and ops[0] == Nodd and ops[0] > z)])
    chi = np.array(chi); A = np.array(A, float); vnull = np.array(vnull); F_ = np.array(feats)
    z_ = (chi - chi.mean()) / np.sqrt(vnull / A)
    row = dict(X=X, K=int(K), chi_mean=float(chi.mean()), chi_sd=float(chi.std()), mean_A=float(A.mean()), D=float(z_.std()),
               corr_chi_logS=float(np.corrcoef(chi, F_[:, 0])[0, 1]), R2_local=r2(chi, F_.T), R2_local_z=r2(z_, F_.T), R2_each=[r2(chi, [F_[:, j]]) for j in range(4)],
               secs=time.time() - t0)
    rows.append(row)
    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    json.dump(dict(rows=rows, smoke=SMOKE), open(os.path.join(OUT, "o4_int_smoke.json" if SMOKE else "o4_int_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
