"""Diagnostic (exploratory, not preregistered): is the residual heavy tail after E2.3
just heteroscedasticity (noise variance growing with the singular series S(N))?"""
import os, sys, json
import numpy as np
sys.argv = [sys.argv[0]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2_3_tail_full.py"), encoding="utf-8").read()
ns = {"__file__": os.path.abspath(__file__), "__name__": "diag"}
exec(compile(src.split("summary = {}")[0], "e2_3_tail_full.py", "exec"), ns)
g = ns
out = {}
for LOG2X in [20, 22, 24]:
    X = 1 << LOG2X; L = 1 << (LOG2X + 2); Q = g["Q"]
    isp = g["prime_sieve"](X); a = np.zeros(X + 1); a[isp] = np.log(np.nonzero(isp)[0])
    A = np.fft.rfft(a, L); A2 = A * A; Nv = np.arange(X // 2, X + 1, 2)
    R = np.fft.irfft(A2, L)[Nv]; mk = g["major_mask"](L, X, Q); Rmaj = np.fft.irfft(A2 * mk, L)[Nv]; Rmin = R - Rmaj
    C2 = 0.6601618158468695739278121100145
    fullS = np.full(X + 1, 2 * C2)
    for p in np.nonzero(isp)[0][1:]:
        fullS[p::p] *= (p - 1) / (p - 2)
    mu, phi = g["mu_phi"](Q); qs = np.array([q for q in range(1, Q + 1) if mu[q] != 0], dtype=np.int64)
    SQ = g["sing_partial"](Nv.astype(np.int64), qs, mu, phi, 0)
    tail = (fullS[Nv] - SQ) * Nv
    af, bf = np.polyfit(tail, Rmin, 1); res = Rmin - (af * tail + bf)
    S = fullS[Nv]
    rows = {}
    qb = np.quantile(S, [0, .5, .8, .95, .99, 1])
    rows["var_ratio_by_S_bin"] = [(float(lo), float(hi), float(res[(S >= lo) & (S <= hi)].var() / res.var()), float(S[(S >= lo) & (S <= hi)].mean())) for lo, hi in zip(qb[:-1], qb[1:])]
    # fit var ~ S^gamma
    cen = np.digitize(S, np.quantile(S, np.linspace(0, 1, 21)[1:-1]))
    ms = np.array([S[cen == c].mean() for c in range(20)]); vs = np.array([res[cen == c].var() for c in range(20)])
    rows["gamma_var_vs_S"] = float(np.polyfit(np.log(ms), np.log(vs), 1)[0])
    for name, w in [("raw", np.ones_like(S)), ("div_S^(gamma/2)", S ** (rows["gamma_var_vs_S"] / 2))]:
        z = res / w; z = (z - z.mean()) / z.std(); n = len(z)
        rows[name] = dict(kurt=float((z**4).mean() - 3), maxz=float(np.abs(z).max() / np.sqrt(2 * np.log(n))))
    out[LOG2X] = rows
    print(LOG2X, json.dumps(rows))
json.dump(out, open(os.path.join(g["OUT"], "e2_3_diag_hetero.json"), "w"), indent=2)
