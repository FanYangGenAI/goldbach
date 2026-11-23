"""E1.1 parity probe (see DESIGN.md).

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
N_BASE = 12  # N per window where the (expensive) baseline is computed exactly
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


NMAX = max(WINDOWS) + 2 * W + 10
lpf = lpf_sieve(NMAX)
odd_primes = np.nonzero(lpf == np.arange(NMAX + 1, dtype=np.int32))[0]
odd_primes = odd_primes[odd_primes > 2].astype(np.int64)
print(f"lpf sieve to {NMAX:,}: {time.time()-t0:.1f}s")


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


results = {}
for N0 in WINDOWS:
    Ns = np.arange(N0, N0 + 2 * W, 2, dtype=np.int64)
    R, Pc = parity_counts(Ns, lpf, odd_primes)
    f = Pc / R
    bN = Ns[np.linspace(0, len(Ns) - 1, N_BASE).astype(int)]
    fb = baseline(bN, lpf)
    floc = np.interp(Ns, bN, fb)
    sd_bin = np.sqrt(floc * (1 - floc) / R)
    zsc = (f - floc) / sd_bin
    # P2: class dependence and regression on log S(N)
    cls = {}
    for q in [3, 5, 7]:
        tq = {}
        for c in range(0, q):
            m = (Ns % q) == c
            tq[c] = dict(mean_z=float(zsc[m].mean()), t=float(zsc[m].mean() / (zsc.std() / np.sqrt(m.sum()))), n=int(m.sum()))
        cls[q] = tq
    lS = np.array([sing_log(int(n)) for n in Ns])
    A = np.vstack([np.ones_like(lS), lS]).T
    coef, res, *_ = np.linalg.lstsq(A, zsc, rcond=None)
    resid = zsc - A @ coef
    se = np.sqrt(resid.var() / ((lS - lS.mean()) ** 2).sum())
    t_logS = float(coef[1] / se)
    # also the raw (non-normalized) dependence of f on S: correlation
    corr_f_logS = float(np.corrcoef(f, lS)[0, 1])
    rel_dev = float((f.mean() - floc.mean()) / floc.mean())
    imin = int(np.argmin(zsc))
    results[N0] = dict(
        z=int(round(N0 ** (1 / 3))), mean_f=float(f.mean()), mean_floc=float(floc.mean()),
        rel_dev_mean=rel_dev, z_mean=float(zsc.mean()), z_sd=float(zsc.std()),
        z_min=float(zsc.min()), N_at_zmin=int(Ns[imin]), f_min=float(f.min()),
        f_min_over_floc=float((f / floc).min()), mean_R=float(R.mean()),
        class_t=cls, t_logS=t_logS, corr_f_logS=corr_f_logS,
        P1=abs(rel_dev) < 0.005,
        P2=all(abs(v["t"]) < 3 for q in cls for v in cls[q].values()) and abs(t_logS) < 3,
        P3=0.3 <= float(zsc.std()) <= 1.5,
        P4=float(zsc.min()) > -6,
    )
    np.savez(os.path.join(OUT, f"e1_1_N{N0}.npz"), Ns=Ns, R=R, Pc=Pc, floc=floc, lS=lS)
    print(N0, {k: v for k, v in results[N0].items() if k != "class_t"}, f"{time.time()-t0:.1f}s")

results["f_star_asymptotic"] = 1 / (1 + np.log(2))
with open(os.path.join(OUT, "e1_1_summary.json"), "w") as fh:
    json.dump(results, fh, indent=2, default=float)

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for ax, N0 in zip(axs, WINDOWS):
    d = np.load(os.path.join(OUT, f"e1_1_N{N0}.npz"))
    f = d["Pc"] / d["R"]
    ax.plot(d["Ns"], f, ".", ms=2, label="f(N) actual")
    ax.plot(d["Ns"], d["floc"], "r-", lw=1, label="f_loc baseline")
    ax.axhline(1 / (1 + np.log(2)), color="k", ls=":", lw=0.8, label="f* = 1/(1+log2)")
    ax.set_title(f"N ≈ {N0:.0e}, z = N^(1/3)"); ax.set_xlabel("N")
axs[0].set_ylabel("prime share among rough N−p"); axs[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e1_1_prime_share.png"), dpi=140); plt.close(fig)
print(f"done {time.time()-t0:.1f}s")
