"""O1: HL-type local-model prediction of the signed parity two-point quantity chi_{N,z} (round-12 review eq. 3-5),
z = floor(N^{1/3}).  See DESIGN.md.  Usage: python o1_chi_local.py [--smoke]"""
import json, math, os, sys, time
import numba as nb
import numpy as np
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv
SCALES = [10**5, 10**6] if SMOKE else [10**6, 10**7, 10**8, 10**9]
PER = 10 if SMOKE else 20          # per mod-3 group
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
    out = np.zeros((len(Ns), 4), dtype=np.int64)   # PP, PS, SP, SS (ordered; S = Omega 2, both sides z-rough)
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


def oddprimes(N):
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


def sing(N):
    v = 2 * C2
    for p in oddprimes(N): v *= (p - 1) / (p - 2)
    return v


def rho1(t): return 1 / math.log(t)


def rho2_factory(z):
    lz = math.log(z)
    def rho2(t):
        L = math.log(t)
        if L <= 2 * lz: return 0.0
        return math.log((L - lz) / lz) / L
    return rho2


def pair_integral(N, f, g, z):
    a, b = z + 1, N - z - 1
    I1, _ = integrate.quad(lambda t: f(t) * g(N - t), a, N / 2, limit=400)
    I2, _ = integrate.quad(lambda t: f(t) * g(N - t), N / 2, b, limit=400)
    return I1 + I2


def chi_from(pp, ps, sp, ss):
    A = pp + ps + sp + ss; B = -pp + ss + (ps - sp); C = pp - ps - sp + ss
    return dict(A=A, B=B, C=C, chi=C / A - (B / A) ** 2, mu=B / A)


NMAX = int(1.1 * max(SCALES)) + 10
spf, om = sieve(NMAX)
print(f"sieve to {NMAX:,}: {time.time()-t0:.0f}s", flush=True)
rng = np.random.default_rng(2026112411)
rows = []
for X in SCALES:
    g3, g0 = [], []
    while len(g3) < PER or len(g0) < PER:
        N = int(2 * rng.integers(X // 2, int(1.1 * X) // 2))
        if N in g3 or N in g0: continue
        if N % 3 == 0 and len(g3) < PER: g3.append(N)
        elif N % 3 != 0 and len(g0) < PER: g0.append(N)
    Ns = np.array(g3 + g0, dtype=np.int64)
    Cn = counts(Ns, spf, om)
    for i, N in enumerate(Ns):
        N = int(N); z = int(round(N ** (1 / 3)))
        while z ** 3 > N: z -= 1
        while (z + 1) ** 3 <= N: z += 1
        S = sing(N); r2 = rho2_factory(z)
        PPp = S * pair_integral(N, rho1, rho1, z)
        PSp = S * pair_integral(N, rho1, r2, z)
        SSp = S * pair_integral(N, r2, r2, z)
        obs = chi_from(*[int(x) for x in Cn[i]]); pred = chi_from(PPp, PSp, PSp, SSp)
        rows.append(dict(X=X, N=N, mod3=N % 3, z=z, counts=[int(x) for x in Cn[i]], pred_counts=[PPp, PSp, PSp, SSp],
                         chi_obs=obs["chi"], chi_pred=pred["chi"], mu_obs=obs["mu"], mu_pred=pred["mu"], A_obs=obs["A"], A_pred=pred["A"]))
    print(X, f"{time.time()-t0:.0f}s", flush=True)


def centre(v, g):
    v = np.asarray(v, float).copy()
    for gg in (True, False):
        v[g == gg] -= v[g == gg].mean()
    return v


summ = {}
for X in SCALES:
    rs = [r for r in rows if r["X"] == X]
    g = np.array([r["mod3"] == 0 for r in rs])
    co = centre([r["chi_obs"] for r in rs], g); cp = centre([r["chi_pred"] for r in rs], g)
    rho = float(np.corrcoef(co, cp)[0, 1]); sdr = float(cp.std(ddof=1) / co.std(ddof=1)); R2 = float(1 - np.sum((co - cp) ** 2) / np.sum(co**2))
    summ[str(X)] = dict(rho=rho, sd_ratio=sdr, R2_pred=R2,
                        mean_chi_obs=float(np.mean([r["chi_obs"] for r in rs])), mean_chi_pred=float(np.mean([r["chi_pred"] for r in rs])),
                        mean_mu_obs=float(np.mean([r["mu_obs"] for r in rs])), mean_mu_pred=float(np.mean([r["mu_pred"] for r in rs])),
                        mean_A_ratio=float(np.mean([r["A_obs"] / r["A_pred"] for r in rs])),
                        mean_PP_ratio=float(np.mean([r["counts"][0] / r["pred_counts"][0] for r in rs])))
    print(X, json.dumps({k: round(v, 5) for k, v in summ[str(X)].items()}), flush=True)
key = summ[str(max(SCALES))]
verdict = "support" if (key["rho"] > 0.5 and 0.5 <= key["sd_ratio"] <= 2 and key["R2_pred"] > 0.25) else ("reject" if (key["rho"] < 0.2 or key["sd_ratio"] < 0.1) else "partial")
print("verdict:", verdict)
json.dump(dict(summary=summ, verdict=verdict, rows=rows, smoke=SMOKE), open(os.path.join(OUT, "o1_smoke.json" if SMOKE else "o1_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
