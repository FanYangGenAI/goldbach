"""Experiment 2: sum/difference duality (Goldbach vs prime pairs with gap h),
the Fourier (circle-method) picture, and an affine (a*p + b) universality check.

Run:  python research/goldbach_explore/exp2_duality.py [log2X]
"""
from __future__ import annotations

import sys
import time
from math import gcd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from common import (C2, FIG_DIR, autoconv_counts, autocorr_counts, li2_goldbach_interp,
                    odd_part_singular_factor, prime_sieve, save_json)

LOG2X = int(sys.argv[1]) if len(sys.argv) > 1 else 24
X = 1 << LOG2X
H = X // 4
t0 = time.time()
is_p = prime_sieve(X)
primes = np.nonzero(is_p)[0]
sing = odd_part_singular_factor(X, primes)

# ---- Goldbach side (N <= X) and difference side (h <= H, pairs inside [2, X]) ----
n = np.arange(10000, H + 1, 2)
r = autoconv_counts(is_p)[: X + 1][n].astype(float)
d = autocorr_counts(is_p, H)[n].astype(float)
HLg = 2 * C2 * sing[n] * li2_goldbach_interp(n)

# difference integral int_2^{X-h} dt/(log t log(t+h)) on a grid in h, interpolated
hg = np.linspace(n.min(), n.max(), 3000)
Ig = np.array([integrate.quad(lambda t, h=h: 1 / (np.log(t) * np.log(t + h)), 2, X - h, limit=400)[0] for h in hg])
HLd = 2 * C2 * sing[n] * np.interp(n, hg, Ig)
EG, ED = r / HLg - 1, d / HLd - 1
print(f"sum/diff counts ready {time.time()-t0:.1f}s")

# correlations: shared fluctuation beyond the singular series?
corr_GD = float(np.corrcoef(EG, ED)[0, 1])
corr_raw = float(np.corrcoef(r / (2 * C2 * li2_goldbach_interp(n)), d / (2 * C2 * np.interp(n, hg, Ig)))[0, 1])
# local (detrended) residual correlation: subtract moving average
def detrend(x, w=501):
    k = np.ones(w) / w
    return x - np.convolve(x, k, mode="same")
corr_GD_local = float(np.corrcoef(detrend(EG)[1000:-1000], detrend(ED)[1000:-1000])[0, 1])

Sn = sing[n]
corr_EG_logS = float(np.corrcoef(EG, np.log(Sn))[0, 1])
corr_ED_logS = float(np.corrcoef(ED, np.log(Sn))[0, 1])
# partial: residual correlation within fixed n mod 30 classes after removing per-class mean
cls = n % 30
EGc, EDc = EG.copy(), ED.copy()
for c in np.unique(cls):
    m = cls == c
    EGc[m] -= EG[m].mean(); EDc[m] -= ED[m].mean()
corr_GD_within_class = float(np.corrcoef(detrend(EGc), detrend(EDc))[0, 1])

# lag correlation of Goldbach residual: corr(E(N), E(N+h))
top = n > H // 2
EGt = detrend(EG[top]); EDt = detrend(ED[top])
lags = np.arange(1, 211)  # in steps of 2 => h = 2*lag
lagc_G = np.array([np.corrcoef(EGt[:-l], EGt[l:])[0, 1] for l in lags])
lagc_D = np.array([np.corrcoef(EDt[:-l], EDt[l:])[0, 1] for l in lags])
# predictor of lag-correlation: singular factor of the shift 2*lag
lag_sing = np.array([sing[2 * l] if 2 * l <= X else 1 for l in lags])
corr_lag_vs_sing = float(np.corrcoef(lagc_G, lag_sing)[0, 1])

# ---- Fourier picture: |S(alpha)|^2 on a grid, major-arc peaks at a/q ----
Xs = 1 << 20
a = is_p[: Xs + 1].astype(float)
L = 1 << 22
Sa = np.fft.fft(a, L)  # S(-k/L) ; |S|^2 symmetric
P2 = np.abs(Sa[: L // 2 + 1]) ** 2
alpha = np.arange(L // 2 + 1) / L
pi_xs = a.sum()
peaks = []
for q in range(1, 13):
    for aa in range(0, q):
        if gcd(aa, q) != 1 or aa / q > 0.5:
            continue
        k = int(round(aa / q * L))
        w = slice(max(k - 3, 0), k + 4)
        mu = {1: 1, 2: -1, 3: -1, 5: -1, 6: 1, 7: -1, 10: 1, 11: -1}.get(q, 0)
        phi = sum(1 for t in range(1, q + 1) if gcd(t, q) == 1)
        peaks.append(dict(a=aa, q=q, observed=float(P2[w].max() / pi_xs ** 2), predicted=float((mu / phi) ** 2)))

# ---- affine universality: #{p <= x : a p + b prime} vs Bateman-Horn ----
xa = 2_000_000
big = prime_sieve(6 * xa + 5000)
pa = np.nonzero(big[: xa + 1])[0]
small_l = np.nonzero(prime_sieve(20000))[0]


def bh_const(a, b):
    c = 1.0
    for l in small_l:
        if a % l == 0:
            w = 1 if b % l else l
        else:
            w = 1 if b % l == 0 else 2
        if w >= l:
            return 0.0
        c *= (1 - w / l) / (1 - 1 / l) ** 2
    return c


aff = {}
for A in [1, 2, 3, 4, 6]:
    bs = [b for b in range(1, 4001) if gcd(A, b) == 1 and (A + b) % 2 == 1][:600]  # parity-admissible: A*p+b odd
    lgI = {}
    ratios = []
    for b in bs:
        cnt = int(big[A * pa + b].sum())
        C = bh_const(A, b)
        I = integrate.quad(lambda t: 1 / (np.log(t) * np.log(A * t + b)), 2, xa, limit=200)[0]
        ratios.append(cnt / (C * I))
    ratios = np.array(ratios)
    aff[A] = dict(n_b=len(bs), mean_ratio=float(ratios.mean()), std_ratio=float(ratios.std()),
                  min_ratio=float(ratios.min()), max_ratio=float(ratios.max()))
    print(A, aff[A])

summary = dict(
    X=X, H=H,
    goldbach=dict(mean_E=float(EG.mean()), std_E=float(EG.std())),
    difference=dict(mean_E=float(ED.mean()), std_E=float(ED.std())),
    corr_raw_ratios_sum_vs_diff=corr_raw, corr_residuals_sum_vs_diff=corr_GD,
    corr_residuals_sum_vs_diff_detrended=corr_GD_local,
    corr_EG_logS=corr_EG_logS, corr_ED_logS=corr_ED_logS, corr_residuals_within_mod30_class=corr_GD_within_class,
    lag_corr_goldbach_first=[float(x) for x in lagc_G[:20]],
    corr_lagcorr_vs_singular_factor_of_shift=corr_lag_vs_sing,
    major_arc_peaks=peaks, affine_bateman_horn=aff, elapsed_s=time.time() - t0,
)
print(save_json("exp2_summary.json", summary))

# ---- figures ----
rng = np.random.default_rng(1)
sub = rng.choice(len(n), 40000, replace=False)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
axs[0].scatter(r[sub] / (2 * C2 * li2_goldbach_interp(n[sub])), d[sub] / (2 * C2 * np.interp(n[sub], hg, Ig)), s=0.5, alpha=0.4)
axs[0].plot([1, 3.5], [1, 3.5], "k--", lw=0.7)
axs[0].set_xlabel("Goldbach  r(n) / (2C2 Li2)"); axs[0].set_ylabel("gap-h pairs  d(n) / (2C2 Li)")
axs[0].set_title(f"same singular series (corr={corr_raw:.4f})")
axs[1].scatter(EG[sub], ED[sub], s=0.5, alpha=0.4)
axs[1].set_xlabel("E_sum(n) = r/HL - 1"); axs[1].set_ylabel("E_diff(n) = d/HL - 1")
axs[1].set_title(f"residuals after S(n): corr={corr_GD:.3f} (detrended {corr_GD_local:.3f})")
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/exp2_sum_vs_diff.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(2 * lags, lagc_G, ".-", ms=3, lw=0.6, label="Goldbach residual: corr(E(N), E(N+h))")
ax.plot(2 * lags, lagc_D, ".-", ms=3, lw=0.6, label="gap residual: corr(E_d(h0), E_d(h0+h))")
ax.set_xlabel("shift h"); ax.legend(); ax.axhline(0, color="k", lw=0.5)
ax.set_title("residual correlations peak at shifts divisible by 3·5·7… (twin-type singular series of h)")
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/exp2_lag_corr.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(alpha, P2 / pi_xs ** 2 + 1e-12, lw=0.3)
for pk in peaks:
    if pk["predicted"] > 0:
        ax.plot(pk["a"] / pk["q"], pk["predicted"], "rx", ms=6)
ax.set_ylim(1e-8, 2); ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$|S(\alpha)|^2/\pi(X)^2$")
ax.set_title(r"Exponential sum over primes: major arcs at a/q, height $\approx(\mu(q)/\varphi(q))^2$ (red x)")
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/exp2_exponential_sum.png", dpi=140); plt.close(fig)
print(f"done {time.time()-t0:.1f}s")
