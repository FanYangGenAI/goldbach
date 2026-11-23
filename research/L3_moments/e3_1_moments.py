"""E3.1: variance and moment ladder of F(N) = R(N) - N*S(N) in dyadic windows. See DESIGN.md."""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
LOG2X = int(sys.argv[1]) if len(sys.argv) > 1 else 26
X = 1 << LOG2X
C2 = 0.6601618158468695739278121100145
t0 = time.time()

s = np.ones(X + 1, dtype=bool); s[:2] = False; s[4::2] = False
for i in range(3, int(X**0.5) + 1, 2):
    if s[i]:
        s[i * i :: 2 * i] = False
pr = np.nonzero(s)[0]
a = np.zeros(X + 1); a[pr] = np.log(pr)
L = 2 * X
A = np.fft.rfft(a, L); del a
R = np.fft.irfft(A * A, L)[: X + 1]; del A
print(f"FFT done {time.time()-t0:.1f}s")
S = np.full(X + 1, 2 * C2)
for p in pr[1:]:
    S[p::p] *= (p - 1) / (p - 2)
print(f"singular series {time.time()-t0:.1f}s")

rows = []
for j in range(12, LOG2X):
    N = np.arange(1 << j, 1 << (j + 1), 2)
    F = R[N] - N * S[N]
    Z = F / np.sqrt(S[N])
    zc = Z - Z.mean()
    m2 = (zc**2).mean(); m4 = (zc**4).mean(); m6 = (zc**6).mean()
    sig = np.sqrt(m2)
    K = float((N * S[N] / (sig * np.sqrt(S[N]))).min())
    B4 = float(((F / (N * S[N])) ** 4).sum())
    B2 = float(((F / (N * S[N])) ** 2).sum())
    zs = zc / sig
    rows.append(dict(j=j, x=float(1 << j), n=len(N), mean_F_over_NS=float((F / (N * S[N])).mean()),
                     sigma2=float(m2), m4_ratio=float(m4 / m2**2), m6_ratio=float(m6 / m2**3),
                     maxz_over_sqrt2lnn=float(np.abs(zs).max() / np.sqrt(2 * np.log(len(N)))),
                     K_sigmas_to_counterexample=K, B4=B4, B2=B2,
                     max_abs_F_over_NS=float(np.abs(F / (N * S[N])).max())))
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rows[-1].items()})

lx = np.log([r["x"] for r in rows]); ls2 = np.log([r["sigma2"] for r in rows])
slopes = np.diff(ls2) / np.diff(lx)
for r, sl in zip(rows[1:], slopes):
    r["local_slope_sigma2"] = float(sl)
sel = [r for r in rows if r["j"] >= 16]
P1 = all(1.0 <= r.get("local_slope_sigma2", 0) <= 1.25 for r in sel)
P2 = all(2.7 <= r["m4_ratio"] <= 3.3 and 13.5 <= r["m6_ratio"] <= 16.5 for r in rows if r["j"] >= 18)
Ks = [r["K_sigmas_to_counterexample"] for r in rows if r["j"] >= 14]
P3 = bool(all(b > a_ for a_, b in zip(Ks[:-1], Ks[1:])) and rows[-1]["K_sigmas_to_counterexample"] >= 100)
B4s = [r["B4"] for r in rows]
P4 = bool(rows[-1]["B4"] < 1e-4 and B4s[-1] < B4s[-5])
fit = np.polyfit(lx[4:], ls2[4:], 1)
checks = dict(P1=P1, P2=P2, P3=P3, P4=P4, global_slope_sigma2_j16plus=float(fit[0]),
              B4_slope_per_log_x=float(np.polyfit(lx[4:], np.log(B4s[4:]), 1)[0]))
print(checks)
json.dump(dict(rows=rows, checks=checks), open(os.path.join(OUT, "e3_1_summary.json"), "w"), indent=2)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
xs = [r["x"] for r in rows]
axs[0].loglog(xs, [r["sigma2"] for r in rows], "o-", label=r"$\sigma^2=\mathrm{Var}(F/\sqrt{S})$")
axs[0].loglog(xs, [x * np.log(x) ** 2 for x in xs], "--", label=r"$x\log^2x$")
axs[0].loglog(xs, [x * np.log(x) for x in xs], ":", label=r"$x\log x$")
axs[0].set_xlabel("window start x"); axs[0].legend(); axs[0].set_title("variance of the Goldbach error")
axs[1].semilogx(xs, [r["m4_ratio"] for r in rows], "o-", label="m4/m2² (Gaussian 3)")
axs[1].semilogx(xs, [r["m6_ratio"] / 5 for r in rows], "s-", label="m6/m2³ ÷5 (Gaussian 3)")
axs[1].axhline(3, color="k", lw=0.6); axs[1].legend(); axs[1].set_title("moment ratios"); axs[1].set_xlabel("x")
axs[2].loglog(xs, [r["K_sigmas_to_counterexample"] for r in rows], "o-", label="K: σ's to a counterexample")
axs[2].loglog(xs, [r["B4"] for r in rows], "s-", label=r"$B_4=\sum (F/NS)^4$")
axs[2].loglog(xs, [r["B2"] for r in rows], "^-", label=r"$B_2=\sum (F/NS)^2$")
axs[2].axhline(1, color="k", lw=0.6); axs[2].legend(fontsize=8); axs[2].set_xlabel("x")
axs[2].set_title("moment ladder: B_p < 1 ⇒ no counterexample in window")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e3_1_moments.png"), dpi=140); plt.close(fig)
print(f"done {time.time()-t0:.1f}s")
