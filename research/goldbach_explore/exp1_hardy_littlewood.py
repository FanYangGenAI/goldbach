"""Experiment 1: r(N) vs the Hardy-Littlewood prediction; normalized fluctuations.

Outputs figures/exp1_*.png and out/exp1_summary.json.
Run:  python research/goldbach_explore/exp1_hardy_littlewood.py [log2X]
"""
from __future__ import annotations

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (C2, FIG_DIR, autoconv_counts, li2_goldbach_interp,
                    odd_part_singular_factor, prime_sieve, save_json)

LOG2X = int(sys.argv[1]) if len(sys.argv) > 1 else 25
X = 1 << LOG2X
rng = np.random.default_rng(20261122)
t0 = time.time()

is_p = prime_sieve(X)
primes = np.nonzero(is_p)[0]
r_all = autoconv_counts(is_p)[: X + 1]
print(f"sieve+FFT to X={X:,}: {time.time()-t0:.1f}s, pi(X)={len(primes):,}")

N = np.arange(1000, X + 1, 2)
r = r_all[N].astype(np.float64)
S = 2 * C2 * odd_part_singular_factor(X, primes)[N]
L2 = li2_goldbach_interp(N)
HL = S * L2
E = r / HL - 1.0
print(f"HL ready: {time.time()-t0:.1f}s")

# ---------- Granville-style random models: sieve by primes <= z, then random ----------
from numpy.fft import irfft, rfft

Lf = 1 << int(np.ceil(np.log2(2 * (X + 1))))
n_idx = np.arange(3, X + 1)


def granville_model(z):
    small = primes[primes <= z]
    coprime = np.ones(X + 1, dtype=bool)
    coprime[:2] = False
    for p in small:
        coprime[p::p] = False
    dens = np.prod([1 - 1 / p for p in small])
    prob = np.zeros(X + 1)
    prob[3:] = np.where(coprime[3:], np.minimum(1.0, 1.0 / (np.log(n_idx) * dens)), 0.0)
    prob[small] = 1.0
    mis = rng.random(X + 1) < prob
    rm = autoconv_counts(mis)[: X + 1][N].astype(np.float64)
    P = rfft(prob, Lf)
    erm = irfft(P * P, Lf)[: X + 1][N]
    return rm, erm


Z_LIST = [7, 50, 500]
models = {z: granville_model(z) for z in Z_LIST}
Z_MODEL = 50
r_model, Er_model = models[Z_MODEL]
E_model = r_model / Er_model - 1.0
print(f"random models ready: {time.time()-t0:.1f}s")

# ---------- log-binned statistics ----------
edges = np.unique(np.round(np.logspace(np.log10(2000), np.log10(X), 40)).astype(np.int64))
rows = []
for a, b in zip(edges[:-1], edges[1:]):
    m = (N >= a) & (N < b)
    if m.sum() < 50:
        continue
    dev = r[m] - HL[m]
    devm = r_model[m] - Er_model[m]
    rows.append(dict(
        N_mid=float(np.sqrt(a * b)), n=int(m.sum()),
        mean_E=float(E[m].mean()), std_E=float(E[m].std()),
        std_dev=float(dev.std()), mean_HL=float(HL[m].mean()),
        fano=float(dev.var() / HL[m].mean()),
        mean_E_model=float(E_model[m].mean()), std_E_model=float(E_model[m].std()),
        fano_model=float(devm.var() / Er_model[m].mean()),
        min_E=float(E[m].min()), max_E=float(E[m].max()),
        **{f"fano_model_z{z}": float(((models[z][0] - models[z][1])[m]).var() / models[z][1][m].mean()) for z in Z_LIST},
    ))
Nmid = np.array([x["N_mid"] for x in rows])
std_dev = np.array([x["std_dev"] for x in rows])
sel = Nmid > 1e5
beta, logc = np.polyfit(np.log(Nmid[sel]), np.log(std_dev[sel]), 1)
hl_sel = np.array([x["mean_HL"] for x in rows])[sel]
beta_hl, _ = np.polyfit(np.log(Nmid[sel]), np.log(hl_sel), 1)

# ---------- residual vs N mod q (top decade) ----------
top = N > X // 10
mod_table = {}
for q in [3, 5, 7, 11, 13, 30, 210]:
    cls = {}
    for c in range(q):
        mm = top & (N % q == c)
        if mm.sum():
            cls[int(c)] = float(E[mm].mean())
    mod_table[q] = cls

# ---------- spectra of the raw and corrected sequences ----------
W = 1 << min(20, int(np.log2(len(N))))
seg = slice(len(N) - W, len(N))
raw = r[seg] / L2[seg]  # no singular series
raw = raw - raw.mean()
cor = E[seg] - E[seg].mean()
mod = E_model[seg] - E_model[seg].mean()
freqs = np.fft.rfftfreq(W, d=1.0)  # cycles per step of 2 in N
spec_raw = np.abs(np.fft.rfft(raw)) ** 2 / W
spec_cor = np.abs(np.fft.rfft(cor)) ** 2 / W
spec_mod = np.abs(np.fft.rfft(mod)) ** 2 / W


def top_peaks(spec, k=8):
    idx = np.argsort(spec[1:])[::-1][:k] + 1
    return [dict(freq=float(freqs[i]), power_over_median=float(spec[i] / np.median(spec[1:]))) for i in idx]


peaks_raw = top_peaks(spec_raw)
peaks_cor = top_peaks(spec_cor)
# lag autocorrelation of E (step of 2 in N)
ac = {int(l): float(np.corrcoef(cor[:-l], cor[l:])[0, 1]) for l in [1, 2, 3, 5, 15, 105]}
ac_model = {int(l): float(np.corrcoef(mod[:-l], mod[l:])[0, 1]) for l in [1, 2, 3, 5, 15, 105]}


# ---------- prime-race prediction of class means ("spectral lines" in the residual) ----------
from math import gcd

race_pred = {}
pX = primes[(primes > 7) & (primes <= X)]
for q in [3, 5, 7]:
    units = [a for a in range(q) if gcd(a, q) == 1]
    cnt = {a: int((pX % q == a).sum()) for a in units}
    mean_c = np.mean(list(cnt.values()))
    frac = {a: cnt[a] / mean_c for a in units}
    pred = {}
    for c in range(q):
        terms = [frac[a] * frac[(c - a) % q] for a in units if gcd((c - a) % q, q) == 1]
        pred[int(c)] = float(np.mean(terms) - 1.0) if terms else None
    obs = {c: mod_table[q][c] - np.mean(list(mod_table[q].values())) for c in mod_table[q]}
    race_pred[q] = dict(class_counts=cnt, predicted_rel=pred, observed_E_centered=obs)

# worst normalized N (closest to "counterexample" in the ratio sense)
big = N > 1e5
worst = np.argsort(E[big])[:10]
worst_list = [dict(N=int(N[big][i]), r=int(r[big][i]), HL=float(HL[big][i]), ratio=float(1 + E[big][i])) for i in worst]

summary = dict(
    X=X, pi_X=int(len(primes)), model_z=Z_MODEL,
    std_dev_powerlaw_beta=float(beta), HL_powerlaw_beta=float(beta_hl),
    bins=rows, mean_E_by_class_top_decade=mod_table,
    spectrum_peaks_raw=peaks_raw, spectrum_peaks_corrected=peaks_cor,
    spectrum_corrected_max_over_median=float(spec_cor[1:].max() / np.median(spec_cor[1:])),
    spectrum_model_max_over_median=float(spec_mod[1:].max() / np.median(spec_mod[1:])),
    autocorr_E=ac, prime_race_class_prediction=race_pred, autocorr_E_model=ac_model, worst_ratio_N_gt_1e5=worst_list,
    elapsed_s=time.time() - t0,
)
print(save_json("exp1_summary.json", summary))

# ---------- figures ----------
sub = rng.choice(len(N), size=min(len(N), 60000), replace=False)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(N[sub], r[sub], s=0.3, alpha=0.4, label="r(N) (ordered)", color="#4a6fa5")
o = np.argsort(N[sub])
ax.plot(N[sub][o], (2 * C2 * L2[sub][o]), color="#c0392b", lw=1, label=r"$2C_2\,\mathrm{Li}_2(N)$ (lower band)")
ax.set_xlabel("N"); ax.set_ylabel("count"); ax.set_title("Goldbach comet")
ax.legend(markerscale=10); fig.tight_layout(); fig.savefig(f"{FIG_DIR}/exp1_comet.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(N[sub], 1 + E[sub], s=0.3, alpha=0.4, color="#4a6fa5")
ax.axhline(1, color="k", lw=0.7)
ax.set_xscale("log"); ax.set_xlabel("N"); ax.set_ylabel("r(N) / HL(N)")
ax.set_title("Normalized comet: the singular series collapses all bands")
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/exp1_ratio.png", dpi=140); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
axs[0].loglog(Nmid, std_dev, "o-", label="std(r - HL) real")
axs[0].loglog(Nmid, [x["mean_HL"] ** 0.5 for x in rows], "--", label=r"$\sqrt{HL}$ (Poisson scale)")
axs[0].loglog(Nmid, [np.sqrt(x["fano_model"] * x["mean_HL"]) for x in rows], ":", label="random-model scale")
axs[0].set_xlabel("N"); axs[0].legend(); axs[0].set_title(f"fluctuation size, slope ≈ {beta:.3f}")
axs[1].semilogx(Nmid, [x["fano"] for x in rows], "o-", label="real primes")
for z in Z_LIST:
    axs[1].semilogx(Nmid, [x[f"fano_model_z{z}"] for x in rows], "s-", ms=3, label=f"random model, sieved by p≤{z}")
axs[1].set_xlabel("N"); axs[1].set_ylabel("Var(r - HL) / HL"); axs[1].legend()
axs[1].set_title("Fano factor of Goldbach counts")
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/exp1_fluctuations.png", dpi=140); plt.close(fig)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for a, sp, t in zip(axs, [spec_raw, spec_cor, spec_mod], ["r/Li2 (no singular series)", "r/HL - 1 (corrected)", "random model residual"]):
    a.semilogy(freqs[1:], sp[1:], lw=0.3)
    a.set_title(t, fontsize=10)
axs[-1].set_xlabel("frequency (cycles per step ΔN=2)")
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/exp1_spectrum.png", dpi=140); plt.close(fig)

seg_z = (r - HL)[N > X // 2] / np.sqrt(HL[N > X // 2])
segm_z = (r_model - Er_model)[N > X // 2] / np.sqrt(Er_model[N > X // 2])
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(seg_z, bins=200, density=True, alpha=0.6, label=f"real, sd={seg_z.std():.3f}")
ax.hist(segm_z, bins=200, density=True, alpha=0.6, label=f"model, sd={segm_z.std():.3f}")
ax.set_xlabel(r"$(r-HL)/\sqrt{HL}$, N in [X/2, X]"); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/exp1_zscore_hist.png", dpi=140); plt.close(fig)
summary2 = dict(z_sd_real=float(seg_z.std()), z_sd_model=float(segm_z.std()),
                z_skew_real=float(((seg_z - seg_z.mean()) ** 3).mean() / seg_z.std() ** 3),
                z_kurt_real=float(((seg_z - seg_z.mean()) ** 4).mean() / seg_z.std() ** 4 - 3))
save_json("exp1_zscores.json", summary2)
print(summary2, f"done {time.time()-t0:.1f}s")
