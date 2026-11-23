"""Q1b: Euler-product (two-variable Selberg-Delange-type) model for the sector table T_jk(N)
and the zero cluster of H_N(z). No fitted parameters. See DESIGN.md."""
import json, math, os, time
import numba as nb
import numpy as np
import mpmath as mp
from scipy.special import rgamma

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
KMAX = 40
M = 128          # FFT points per dimension on |z| = |w| = 1
PMAX = 10**6
t0 = time.time()

# reuse the Omega sieve / sector tables from q1q3 (same seeds -> same N samples)
src = open(os.path.join(HERE, "q1q3_sectors_zeros.py"), encoding="utf-8").read().split("NMAX = 10**8")[0]
src = src.replace("cache=True", "cache=False")
g = {"__name__": "q1b", "__file__": os.path.join(HERE, "q1q3_sectors_zeros.py")}
exec(compile(src, "q1q3_sectors_zeros.py", "exec"), g)

ang = np.exp(2j * np.pi * np.arange(M) / M)
Z = ang[:, None] * np.ones((1, M)); W = np.ones((M, 1)) * ang[None, :]

# primes up to PMAX
s = np.ones(PMAX + 1, bool); s[:2] = False
for i in range(2, int(PMAX**0.5) + 1):
    if s[i]:
        s[i * i :: i] = False
primes = np.nonzero(s)[0]


def J(p, e, z, w):
    Zp = (1 - 1 / p) / (1 - z / p); Wp = (1 - 1 / p) / (1 - w / p)
    val = 1 - 2 / p + (z * Zp + w * Wp) / p
    for _ in range(e):
        val = z * w * val / p + (1 - 1 / p)
    return val


# base Euler product over all p <= PMAX with e = 0
logbase = np.zeros((M, M), complex)
for p in primes:
    logbase += np.log(J(p, 0, Z, W)) + (Z + W - 2) * math.log(1 - 1 / p)
print(f"base Euler product {time.time()-t0:.1f}s", flush=True)


def model_coeffs(N):
    logA = logbase.copy()
    n = N; p = 2
    fac = {}
    while p * p <= n:
        while n % p == 0:
            fac[p] = fac.get(p, 0) + 1; n //= p
        p += 1
    if n > 1:
        fac[n] = fac.get(n, 0) + 1
    for p, e in fac.items():
        if p <= PMAX:
            logA -= np.log(J(p, 0, Z, W))
        else:
            logA -= (Z + W - 2) * math.log(1 - 1 / p)  # was not in base; add full factor below
        logA += np.log(J(p, e, Z, W)) + (0 if p <= PMAX else 0)
        if p > PMAX:
            logA += (Z + W - 2) * math.log(1 - 1 / p)
    L = math.log(N)
    G = N * np.exp((Z + W - 2) * math.log(L) + logA) * rgamma(Z) * rgamma(W)
    C = np.fft.fft2(G) / (M * M)  # coefficient of z^j w^k at [j, k] (radius 1)
    return C.real


def corr(T):
    T = np.asarray(T, float)
    row = T.sum(1); col = T.sum(0); tot = T.sum()
    Ind = np.outer(row, col) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(Ind > 0, T / Ind, np.nan)


NMAX = 10**8
om = g["omega_sieve"](int(1.1 * NMAX) + 10)
rng = np.random.default_rng(11)
res = {}
for N0 in [10**6, 10**7, 10**8]:
    Ns = (2 * rng.integers(N0 // 2, int(1.1 * N0) // 2, size=30)).astype(np.int64)
    T = g["sector_tables"](Ns, om)
    Cobs = []; Cmod = []; zc = []
    for i, N in enumerate(Ns):
        Tm = model_coeffs(int(N))[:KMAX, :KMAX]
        Tm = np.maximum(Tm, 0)
        Cobs.append(corr(T[i])[:9, :9]); Cmod.append(corr(Tm)[:9, :9])
        if i < 5:
            deg = 0
            h_obs = np.zeros(2 * KMAX); h_mod = np.zeros(2 * KMAX)
            for j in range(KMAX):
                for k in range(KMAX):
                    h_obs[j + k] += T[i][j, k]; h_mod[j + k] += Tm[j, k]
            top = int(np.nonzero(h_obs)[0].max())
            def near_m1(h):
                mp.mp.dps = 40
                c = [mp.mpf(float(x)) for x in h[2 : top + 1]]
                r = np.array([complex(x) for x in mp.polyroots(list(reversed(c)), maxsteps=400, extraprec=400)])
                idx = np.argsort(np.abs(r + 1))[:3]
                return r[idx]
            ro = near_m1(h_obs); rm = near_m1(h_mod)
            zc.append(dict(N=int(N), obs=[[x.real, x.imag] for x in ro], model=[[x.real, x.imag] for x in rm],
                           centroid_obs=[ro.mean().real, ro.mean().imag], centroid_model=[rm.mean().real, rm.mean().imag],
                           dist=float(abs(ro.mean() - rm.mean()))))
    Cobs = np.nanmean(np.array(Cobs), 0); Cmod = np.nanmean(np.array(Cmod), 0)
    err = np.log(Cmod[1:9, 1:9]) - np.log(Cobs[1:9, 1:9])
    res[N0] = dict(logerr=np.round(err, 3).tolist(), max_abs_logerr_jk_le4=float(np.nanmax(np.abs(err[:4, :4]))),
                   C_obs=np.round(Cobs[1:9, 1:9], 3).tolist(), C_model=np.round(Cmod[1:9, 1:9], 3).tolist(),
                   zero_cluster=zc, median_centroid_dist=float(np.median([z["dist"] for z in zc])))
    print(N0, "max|logerr| j,k<=4:", res[N0]["max_abs_logerr_jk_le4"], "median centroid dist:", res[N0]["median_centroid_dist"], f"{time.time()-t0:.1f}s", flush=True)
res["P2"] = bool(res[10**8]["max_abs_logerr_jk_le4"] < 0.10)
res["P3"] = bool(res[10**8]["median_centroid_dist"] < 0.15)
print("P2:", res["P2"], "P3:", res["P3"])
json.dump(res, open(os.path.join(OUT, "q1b_summary.json"), "w"), indent=1, default=float)
