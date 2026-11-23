"""E3.4: Fujii's explicit formula for Goldbach averages vs zeta zeros; local version
R_bar(x) ~ x + 2(psi(x) - x); share of pointwise variance explained by zeta zeros. See DESIGN.md."""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
LOG2X = 26
X = 1 << LOG2X
K = 20000
C2 = 0.6601618158468695739278121100145


def zero_im(n):
    import mpmath
    mpmath.mp.dps = 20
    return float(mpmath.zetazero(n).imag)


def main():
    t0 = time.time()
    zpath = os.path.join(OUT, f"zeta_zeros_{K}.npy")
    if os.path.exists(zpath):
        gam = np.load(zpath)
    else:
        with Pool(24) as pool:
            gam = np.array(pool.map(zero_im, range(1, K + 1), chunksize=50))
        np.save(zpath, gam)
    print(f"{K} zeros, gamma_K={gam[-1]:.2f}, {time.time()-t0:.1f}s", flush=True)

    # von Mangoldt
    s = np.ones(X + 1, dtype=bool); s[:2] = False; s[4::2] = False
    for i in range(3, int(X**0.5) + 1, 2):
        if s[i]:
            s[i * i :: 2 * i] = False
    pr = np.nonzero(s)[0]
    lam = np.zeros(X + 1)
    for p in pr:
        lp = np.log(p); q = p
        while q <= X:
            lam[q] = lp
            if q > X // p:
                break
            q *= p
    psi = np.cumsum(lam)
    A = np.fft.rfft(lam, 2 * X)
    R = np.fft.irfft(A * A, 2 * X)[: X + 1]
    del A
    cumR = np.cumsum(R)
    print(f"R_Lambda ready {time.time()-t0:.1f}s", flush=True)

    rho = 0.5 + 1j * gam

    def zero_sum(xs):
        xs = np.asarray(xs, float)
        out = np.zeros(len(xs))
        for i0 in range(0, len(xs), 50):
            xx = xs[i0 : i0 + 50, None]
            out[i0 : i0 + 50] = -2 * 2 * np.real(np.exp((rho + 1) * np.log(xx)) / (rho * (rho + 1))).sum(axis=1)
        return out

    # ---- P1 ----
    xs = np.unique(np.round(np.exp(np.linspace(np.log(2**14), np.log(X - 1), 400))).astype(np.int64))
    G = cumR[xs] - xs.astype(float) ** 2 / 2
    Z = zero_sum(xs)
    corr_raw = float(np.corrcoef(G, Z)[0, 1]); res_raw = float(np.sqrt(((G - Z) ** 2).mean()) / np.sqrt((G**2).mean()))
    g = G / xs**1.5; z = Z / xs**1.5
    corr_norm = float(np.corrcoef(g, z)[0, 1]); res_norm = float(np.sqrt(((g - z) ** 2).mean()) / np.sqrt((g**2).mean()))
    resid_over_xlogx = (G - Z) / (xs * np.log(xs))
    # ---- P2 ----
    Hw = 1 << 14
    starts = np.unique(np.round(np.exp(np.linspace(np.log(2**20), np.log(X - Hw - 1), 2000))).astype(np.int64))
    Rbar = (cumR[starts + Hw] - cumR[starts]) / Hw
    xbar = starts + (Hw + 1) / 2.0
    xi = np.round(xbar).astype(np.int64)
    ypsi = 2 * (psi[xi] - xbar)
    yR = Rbar - xbar
    slope, icpt = np.polyfit(ypsi, yR, 1)
    R2 = 1 - np.var(yR - (slope * ypsi + icpt)) / np.var(yR)
    # normalized by sqrt(x) to avoid domination by large x
    slope_n, icpt_n = np.polyfit(ypsi / np.sqrt(xbar), yR / np.sqrt(xbar), 1)
    R2_n = 1 - np.var(yR / np.sqrt(xbar) - (slope_n * ypsi / np.sqrt(xbar) + icpt_n)) / np.var(yR / np.sqrt(xbar))
    # ---- P3 ----
    shares = {}
    for j in [24, 26]:
        lo = (1 << j) - (1 << 21) if j == LOG2X else (1 << j)
        N = np.arange(lo, lo + (1 << 21), 2)
        Sg = np.full(len(N), 2 * C2)
        for p in pr[1:]:
            if p > N[-1]:
                break
            st = (-lo) % p
            if st % 2 == 1:
                st += p
            Sg[st // 2 :: p] *= (p - 1) / (p - 2)
        F = R[N] - N * Sg
        zeta_part = 2 * (psi[N] - N)
        zp = (zeta_part - zeta_part.mean()) * Sg
        shares[j] = dict(var_F=float(F.var()), var_zeta_part=float(zp.var()), share=float(zp.var() / F.var()),
                         rms_F_over_sqrtN=float(F.std() / np.sqrt(N.mean())), rms_psi_err_over_sqrtN=float((psi[N] - N).std() / np.sqrt(N.mean())))
    checks = dict(P1=bool(corr_raw > 0.95 and res_raw < 0.3), P2=bool(0.9 <= slope <= 1.1 and R2 > 0.5),
                  P3=all(v["share"] < 0.05 for v in shares.values()))
    summary = dict(K=K, gamma_K=float(gam[-1]), P1=dict(corr_raw=corr_raw, resid_rms_raw=res_raw, corr_norm=corr_norm, resid_rms_norm=res_norm,
                   max_abs_resid_over_xlogx=float(np.abs(resid_over_xlogx).max()), median_abs_resid_over_xlogx=float(np.median(np.abs(resid_over_xlogx)))),
                   P2=dict(H=Hw, slope=float(slope), intercept=float(icpt), R2=float(R2), slope_norm=float(slope_n), R2_norm=float(R2_n)),
                   P3=shares, checks=checks, elapsed_s=time.time() - t0)
    print(json.dumps(summary, indent=1), flush=True)
    json.dump(summary, open(os.path.join(OUT, "e3_4_summary.json"), "w"), indent=2)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    axs[0].semilogx(xs, g, lw=1, label=r"$G(x)/x^{3/2}$ (data)")
    axs[0].semilogx(xs, z, lw=1, ls="--", label=f"zero sum, K={K}")
    axs[0].set_title("Fujii: cumulative Goldbach count vs zeta zeros"); axs[0].legend(); axs[0].set_xlabel("x")
    sel = slice(None)
    axs[1].plot(ypsi / np.sqrt(xbar), yR / np.sqrt(xbar), ".", ms=2)
    lim = np.abs(ypsi / np.sqrt(xbar)).max(); axs[1].plot([-lim, lim], [-lim, lim], "r--", lw=0.8)
    axs[1].set_xlabel(r"$2(\psi(x)-x)/\sqrt{x}$"); axs[1].set_ylabel(r"$(\bar R(x)-x)/\sqrt{x}$, H=$2^{14}$")
    axs[1].set_title(f"local Goldbach average tracks prime error: slope {slope:.3f}, R²={R2:.3f}")
    axs[2].semilogx(xs, resid_over_xlogx, lw=1); axs[2].set_title(r"Fujii residual / $(x\log x)$"); axs[2].set_xlabel("x")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "e3_4_explicit_formula.png"), dpi=140)
    print(f"done {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
