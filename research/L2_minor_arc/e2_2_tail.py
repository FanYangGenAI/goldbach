"""E2.2: is the minor-arc part = singular-series tail (q > Q) + random-phase noise?
See DESIGN.md.  Run: python e2_2_tail.py
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
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "out"); FIG = os.path.join(HERE, "figures")
LOG2XS = [int(a) for a in sys.argv[1:]] or [20, 22, 24]
Q = 256
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
            lo = max(int(np.floor(c - w)), 0); hi = min(int(np.ceil(c + w)), H - 1)
            for k in range(lo, hi + 1):
                m[k] = True
    return m


def mu_phi(K):
    mu = np.ones(K + 1, dtype=np.int64); phi = np.arange(K + 1, dtype=np.float64)
    isp = prime_sieve(K)
    for p in np.nonzero(isp)[0]:
        mu[p::p] *= -1; mu[p * p :: p * p] = 0
        phi[p::p] *= (1 - 1 / p)
    return mu, phi


@nb.njit(parallel=True, cache=True)
def sing_partial(Ns, qs, mu, phi, Q):
    """returns S_K(N) - S_Q(N) = sum_{Q < q <= K, q squarefree} mu(q/d) phi(d) / phi(q)^2."""
    out = np.zeros(len(Ns))
    for i in nb.prange(len(Ns)):
        N = Ns[i]; s = 0.0
        for q in qs:
            if q <= Q:
                continue
            d = _gcd(q, N)
            s += mu[q // d] * phi[d] / (phi[q] * phi[q])
        out[i] = s
    return out


summary = {}
for LOG2X in LOG2XS:
    X = 1 << LOG2X; L = 1 << (LOG2X + 2); K = int(2 * np.sqrt(X))
    isp = prime_sieve(X)
    a = np.zeros(X + 1); a[isp] = np.log(np.nonzero(isp)[0])
    A = np.fft.rfft(a, L); A2 = A * A
    Nv = np.arange(X // 2, X + 1, 2)
    R = np.fft.irfft(A2, L)[Nv]
    mk = major_mask(L, X, Q)
    Rmaj = np.fft.irfft(A2 * mk, L)[Nv]
    Rmin = R - Rmaj
    del A, A2
    mu, phi = mu_phi(K)
    qs = np.array([q for q in range(1, K + 1) if mu[q] != 0], dtype=np.int64)
    tail = sing_partial(Nv.astype(np.int64), qs, mu, phi, Q) * Nv
    a_fit, b_fit = np.polyfit(tail, Rmin, 1)
    resid = Rmin - (a_fit * tail + b_fit)
    R2 = 1 - resid.var() / Rmin.var()
    z = (resid - resid.mean()) / resid.std(); n = len(z)
    zr = (Rmin - Rmin.mean()) / Rmin.std()
    summary[LOG2X] = dict(
        X=X, K=K, Q=Q, a_fit=float(a_fit), R2=float(R2),
        rms_Rmin_over_Rmaj=float(Rmin.std() / Rmaj.mean()), rms_resid_over_Rmaj=float(resid.std() / Rmaj.mean()),
        kurt_before=float((zr**4).mean() - 3), kurt_after=float((z**4).mean() - 3),
        maxz_before=float(np.abs(zr).max() / np.sqrt(2 * np.log(n))), maxz_after=float(np.abs(z).max() / np.sqrt(2 * np.log(n))),
        max_resid_over_Rmaj=float((np.abs(resid) / Rmaj).max()),
        N_at_max_resid=int(Nv[np.argmax(np.abs(z))]),
    )
    print(LOG2X, summary[LOG2X], f"{time.time()-t0:.1f}s")
    if LOG2X == max(LOG2XS):
        sub = np.random.default_rng(0).choice(n, 200000, replace=False)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        axs[0].scatter(tail[sub] / Nv[sub], Rmin[sub] / Nv[sub], s=0.3, alpha=0.3)
        xx = np.linspace(tail.min() / Nv.max(), (tail / Nv).max(), 10); axs[0].plot(xx, xx, "r--", lw=0.8, label="slope 1")
        axs[0].set_xlabel(r"singular-series tail $\mathfrak{S}_K(N)-\mathfrak{S}_Q(N)$"); axs[0].set_ylabel(r"$R_{minor}(N)/N$")
        axs[0].set_title(f"minor arcs vs arithmetic tail (Q={Q}, X=2^{LOG2X}), R²={R2:.3f}"); axs[0].legend()
        axs[1].hist(zr, bins=300, density=True, alpha=0.5, label=f"R_minor, kurt={summary[LOG2X]['kurt_before']:.1f}", log=True)
        axs[1].hist(z, bins=300, density=True, alpha=0.5, label=f"after removing tail, kurt={summary[LOG2X]['kurt_after']:.2f}", log=True)
        xx = np.linspace(-8, 8, 300); axs[1].plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=0.8)
        axs[1].set_xlim(-10, 30); axs[1].set_ylim(1e-7, 1); axs[1].legend(fontsize=8); axs[1].set_xlabel("standardized")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "e2_2_tail.png"), dpi=140); plt.close(fig)

xs = np.array([2.0**k for k in LOG2XS]); rr = np.array([summary[k]["rms_resid_over_Rmaj"] for k in LOG2XS])
beta = float(-np.polyfit(np.log(xs), np.log(rr), 1)[0]) if len(xs) > 1 else None
top = summary[max(LOG2XS)]
summary["checks"] = dict(
    P1=bool(top["R2"] >= 0.5 and 0.7 <= top["a_fit"] <= 1.3),
    P2=bool(abs(top["kurt_after"]) < 0.5 and 0.7 <= top["maxz_after"] <= 1.3),
    beta=beta, P3=bool(beta is not None and beta > 0.3),
)
print(summary["checks"])
with open(os.path.join(OUT, "e2_2_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2, default=float)
print(f"done {time.time()-t0:.1f}s")
