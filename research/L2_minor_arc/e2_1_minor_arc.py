"""E2.1: exact major/minor arc decomposition of R(N) = sum_{p+q=N} log p log q.
See DESIGN.md.  Run: python e2_1_minor_arc.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from math import gcd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "goldbach_explore"))
LOG2XS = [int(a) for a in sys.argv[1:]] or [20, 22, 24]
QS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
t0 = time.time()


def prime_sieve(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False; s[4::2] = False
    for i in range(3, int(n**0.5) + 1, 2):
        if s[i]:
            s[i * i :: 2 * i] = False
    return s


@nb.njit(cache=True)
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


@nb.njit(cache=True)
def major_mask(L, X, Q):
    H = L // 2 + 1
    m = np.zeros(H, dtype=np.bool_)
    for q in range(1, Q + 1):
        w = Q * L / (q * X)
        for a in range(0, q // 2 + 1):
            if _gcd(a, q) != 1:
                continue
            c = L * a / q
            lo = int(np.floor(c - w)); hi = int(np.ceil(c + w))
            if lo < 0:
                lo = 0
            if hi > H - 1:
                hi = H - 1
            for k in range(lo, hi + 1):
                m[k] = True
    return m


summary = {}
for LOG2X in LOG2XS:
    X = 1 << LOG2X
    L = 1 << (LOG2X + 2)
    isp = prime_sieve(X)
    a = np.zeros(X + 1); a[isp] = np.log(np.nonzero(isp)[0])
    A = np.fft.rfft(a, L)
    A2 = A * A
    absS2 = np.abs(A) ** 2
    wk = np.full(len(A), 2.0); wk[0] = 1; wk[-1] = 1  # half-spectrum weights for Parseval sums
    R = np.fft.irfft(A2, L)
    Nv = np.arange(X // 2, X + 1, 2)
    Rv = R[Nv]
    rows = []
    for Q in QS:
        if Q * Q > X:  # keep arcs essentially disjoint (Q <= sqrt X)
            break
        mk = major_mask(L, X, Q)
        Rmaj = np.fft.irfft(A2 * mk, L)[Nv]
        Rmin = Rv - Rmaj
        T = float((absS2 * wk * (~mk)).sum() / L)  # int over minor arcs of |S|^2
        meas = float((wk * mk).sum() / L)
        ratio = np.abs(Rmin) / Rmaj
        z = (Rmin - Rmin.mean()) / Rmin.std()
        n = len(z)
        need_over_obs = (Rmaj / T) / (np.abs(Rmin) / T + 1e-300)
        rows.append(dict(
            Q=Q, major_measure=meas, T_minor_L2=T, mean_Rmaj=float(Rmaj.mean()), mean_R=float(Rv.mean()),
            rms_Rmin=float(Rmin.std()), mean_Rmin=float(Rmin.mean()),
            max_ratio=float(ratio.max()), q99_ratio=float(np.quantile(ratio, 0.99)), median_ratio=float(np.median(ratio)),
            min_Rmaj=float(Rmaj.min()), T_over_meanRmaj=float(T / Rmaj.mean()),
            median_need_over_obs=float(np.median(need_over_obs)), min_need_over_obs=float(need_over_obs.min()),
            excess_kurtosis=float((z**4).mean() - 3), skew=float((z**3).mean()),
            maxabs_z=float(np.abs(z).max()), maxz_over_sqrt2lnn=float(np.abs(z).max() / np.sqrt(2 * np.log(n))),
        ))
        print(LOG2X, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, f"{time.time()-t0:.1f}s")
        if LOG2X == max(LOG2XS) and Q in (16, 256):
            np.savez(os.path.join(OUT, f"e2_1_X{LOG2X}_Q{Q}.npz"), N=Nv, R=Rv, Rmaj=Rmaj)
    summary[LOG2X] = rows
    del A, A2, absS2, R

# preregistered checks
X = max(LOG2XS); rowsX = summary[X]
logX2 = np.log(2.0**X) ** 2
Q0 = next((r["Q"] for r in rowsX if all(rr["max_ratio"] < 0.5 for rr in rowsX if rr["Q"] >= r["Q"])), None)
r256 = next(r for r in rowsX if r["Q"] == 256)
scal = []
for lx in LOG2XS:
    target = np.log(2.0**lx) ** 2
    rr = min(summary[lx], key=lambda r: abs(np.log(r["Q"] / target)))
    scal.append((2.0**lx, rr["Q"], rr["rms_Rmin"] / rr["mean_Rmaj"]))
beta = -np.polyfit(np.log([s[0] for s in scal]), np.log([s[2] for s in scal]), 1)[0] if len(scal) > 1 else None
checks = dict(
    Q0=Q0, logX_sq=float(logX2), P1=bool(Q0 is not None and Q0 <= logX2),
    P2=bool(abs(r256["excess_kurtosis"]) < 0.5 and 0.7 <= r256["maxz_over_sqrt2lnn"] <= 1.3),
    P3=bool(r256["median_need_over_obs"] >= 100), scaling=scal, beta=beta, P4=bool(beta is not None and beta > 0.3),
)
summary["checks"] = checks
print(checks)
with open(os.path.join(OUT, "e2_1_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2, default=float)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for lx in LOG2XS:
    rs = summary[lx]
    axs[0].loglog([r["Q"] for r in rs], [r["max_ratio"] for r in rs], "o-", label=f"max |R_min|/R_maj, X=2^{lx}")
    axs[0].loglog([r["Q"] for r in rs], [r["rms_Rmin"] / r["mean_Rmaj"] for r in rs], "s--", ms=3, label=f"rms R_min / mean R_maj, X=2^{lx}")
    axs[1].loglog([r["Q"] for r in rs], [r["T_over_meanRmaj"] for r in rs], "o-", label=f"trivial bound T/R_maj, X=2^{lx}")
axs[0].axhline(1, color="k", lw=0.6); axs[0].set_xlabel("Q"); axs[0].legend(fontsize=7)
axs[0].set_title("minor-arc contribution relative to main term")
axs[1].axhline(1, color="k", lw=0.6); axs[1].set_xlabel("Q"); axs[1].legend(fontsize=7)
axs[1].set_title(r"trivial minor-arc bound $\int_m|S|^2$ vs main term")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e2_1_minor_vs_Q.png"), dpi=140); plt.close(fig)
print(f"done {time.time()-t0:.1f}s")
