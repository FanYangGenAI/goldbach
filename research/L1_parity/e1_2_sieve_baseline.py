"""E1.2 parity probe with the full sieve-information baseline (see DESIGN.md).

For even N, z = floor(N^(1/3)); among primes p <= N/2 with m = N - p z-rough
(no prime factor <= z), m is prime or a product of two primes > z.
f(N) = (# m prime) / (# m rough). Compare with the local baseline f_loc.
"""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

WINDOWS = [10**6, 10**7, 10**8]
W = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
t0 = time.time()


@nb.njit(cache=True)
def lpf_sieve(n):
    lpf = np.zeros(n + 1, dtype=np.int32)
    for i in range(2, n + 1):
        if lpf[i] == 0:
            lpf[i] = i
            j = np.int64(i) * i
            while j <= n:
                if lpf[j] == 0:
                    lpf[j] = i
                j += i
    return lpf


@nb.njit(cache=True)
def icbrt(n):
    z = int(round(n ** (1.0 / 3.0)))
    while z * z * z > n:
        z -= 1
    while (z + 1) * (z + 1) * (z + 1) <= n:
        z += 1
    return z


@nb.njit(parallel=True, cache=True)
def parity_counts(Ns, lpf, odd_primes):
    K = len(Ns)
    R = np.zeros(K, np.int64); Pc = np.zeros(K, np.int64)
    for k in nb.prange(K):
        N = Ns[k]; z = icbrt(N)
        r = 0; pc = 0
        for p in odd_primes:
            if p > N // 2:
                break
            m = N - p
            f = lpf[m]
            if f > z:
                r += 1
                if f == m:
                    pc += 1
        R[k] = r; Pc[k] = pc
    return R, Pc


@nb.njit(parallel=True, cache=True)
def baseline(Ns, lpf):
    K = len(Ns)
    out = np.zeros(K)
    for k in nb.prange(K):
        N = Ns[k]; z = icbrt(N)
        wr = 0.0; wp = 0.0
        m = N // 2 + 1 if (N // 2) % 2 == 0 else N // 2
        while m <= N - 3:
            f = lpf[m]
            if f > z:
                w = 1.0 / np.log(N - m)
                wr += w
                if f == m:
                    wp += w
            m += 2
        out[k] = wp / wr
    return out




@nb.njit(parallel=True, cache=True)
def sieve_baseline(Ns, lpf):
    """prime share of m among m in [N/2, N) with m AND N-m z-rough, weight 1/log(N-m)."""
    K = len(Ns)
    out = np.zeros(K)
    for k in nb.prange(K):
        N = Ns[k]; z = icbrt(N)
        wr = 0.0; wp = 0.0
        m = N // 2 + 1 if (N // 2) % 2 == 0 else N // 2
        while m <= N - 3:
            f = lpf[m]
            if f > z and lpf[N - m] > z:
                w = 1.0 / np.log(N - m)
                wr += w
                if f == m:
                    wp += w
            m += 2
        out[k] = wp / wr
    return out


NMAX = int(1.1 * max(WINDOWS)) + 2 * W + 10
lpf = lpf_sieve(NMAX)
odd_primes = np.nonzero(lpf == np.arange(NMAX + 1, dtype=np.int32))[0]
odd_primes = odd_primes[odd_primes > 2].astype(np.int64)
print(f"lpf sieve to {NMAX:,}: {time.time()-t0:.1f}s")
rng = np.random.default_rng(12)


def sing_log(N):
    s = 0.0; n = N
    while n % 2 == 0:
        n //= 2
    p = 3
    while p * p <= n:
        if n % p == 0:
            s += np.log((p - 1) / (p - 2))
            while n % p == 0:
                n //= p
        p += 2
    if n > 1:
        s += np.log((n - 1) / (n - 2))
    return s


def analyse(Ns):
    R, Pc = parity_counts(Ns, lpf, odd_primes)
    f = Pc / R
    fs = sieve_baseline(Ns, lpf)
    sd_bin = np.sqrt(fs * (1 - fs) / R)
    zsc = (f - fs) / sd_bin
    cls = {}
    for q in [3, 5, 7]:
        cls[q] = {}
        for c in range(q):
            m = (Ns % q) == c
            cls[q][c] = float(zsc[m].mean() / (zsc.std() / np.sqrt(m.sum())))
    lS = np.array([sing_log(int(n)) for n in Ns])
    A = np.vstack([np.ones_like(lS), lS]).T
    coef, *_ = np.linalg.lstsq(A, zsc, rcond=None)
    resid = zsc - A @ coef
    t_logS = float(coef[1] / np.sqrt(resid.var() / ((lS - lS.mean()) ** 2).sum()))
    rel = float((f.mean() - fs.mean()) / fs.mean())
    maxt = max(abs(v) for q in cls for v in cls[q].values())
    return dict(mean_f=float(f.mean()), mean_fsieve=float(fs.mean()), rel_dev_mean=rel,
                z_mean=float(zsc.mean()), z_sd=float(zsc.std()), z_min=float(zsc.min()),
                f_min_over_fsieve=float((f / fs).min()), max_abs_class_t=maxt, class_t=cls,
                t_logS=t_logS, corr_fsieve_vs_logS=float(np.corrcoef(fs, lS)[0, 1])), (Ns, f, fs, zsc)


results = {}
arrays = {}
for N0 in WINDOWS:
    cons = np.arange(N0, N0 + 2 * W, 2, dtype=np.int64)
    rnd = np.sort(2 * rng.integers(N0 // 2, int(1.1 * N0) // 2, size=W)).astype(np.int64)
    rc, ac = analyse(cons)
    rr, ar = analyse(rnd)
    results[N0] = dict(consecutive=rc, random=rr)
    arrays[N0] = (ac, ar)
    print(N0, "cons", {k: v for k, v in rc.items() if k != "class_t"})
    print(N0, "rand", {k: v for k, v in rr.items() if k != "class_t"}, f"{time.time()-t0:.1f}s")

all_t = [abs(v) for N0 in WINDOWS for q in results[N0]["random"]["class_t"] for v in results[N0]["random"]["class_t"][q].values()]
results["P2prime"] = bool(max(all_t) < 3.5 and all(abs(results[N0]["random"]["t_logS"]) < 3 for N0 in WINDOWS))
results["max_abs_t_random"] = float(max(all_t))
print("P2':", results["P2prime"], "max |t| (random):", max(all_t))
with open(os.path.join(OUT, "e1_2_summary.json"), "w") as fh:
    json.dump(results, fh, indent=2, default=float)

fig, axs = plt.subplots(2, 3, figsize=(14, 7))
for j, N0 in enumerate(WINDOWS):
    (Ns, f, fs, zsc), (Nr, fr, fsr, zr) = arrays[N0]
    axs[0, j].plot(Ns, f, ".", ms=2, label="f(N)")
    axs[0, j].plot(Ns, fs, ".", ms=2, label="f_sieve(N)")
    axs[0, j].set_title(f"N ≈ {N0:.0e} (consecutive)")
    axs[1, j].hist(zr, bins=50, density=True, alpha=0.7, label=f"random N, sd={zr.std():.2f}")
    xx = np.linspace(-4, 4, 200); axs[1, j].plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=0.8)
    axs[1, j].set_xlabel("z = (f − f_sieve)/sd_binom"); axs[1, j].legend(fontsize=8)
axs[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e1_2_sieve_baseline.png"), dpi=140); plt.close(fig)
print(f"done {time.time()-t0:.1f}s")
