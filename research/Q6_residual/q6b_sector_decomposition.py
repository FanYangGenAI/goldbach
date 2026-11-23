"""Q6b (exploratory): decompose the centred absolute residual D_N(z)=E_N(z)/z^2 on |z|=0.5 into Taylor
coefficients e_m = [z^m]E_N (total-Omega sectors m = 2,3,...), via the 16-point FFT (Parseval)."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "out", "q6_summary.json")))
out = {}
for src, X in sorted({(r["src"], r["X"]) for r in d["rows"]}):
    rows = [r for r in d["rows"] if r["src"] == src and r["X"] == X]
    D = np.array([[complex(*x) for x in r["D_over_N"]] for r in rows])          # D/N on 16 points
    coef = np.fft.fft(D, axis=1) / 16                                              # coef[:, k] = e_{k+2}/N * 0.5^k
    k = np.arange(16)
    e = coef / 0.5 ** k                                                            # e_{m}/N, m = k+2
    ec = e - e.mean(0)
    var_share = (np.abs(ec * 0.5 ** k) ** 2).mean(0); var_share = var_share / var_share.sum()
    # correlation of top-mode amplitude with e_2 (Goldbach coefficient deviation)
    Dc = D - D.mean(0)
    C = (Dc.T @ Dc.conj()) / (len(rows) - 1); w, V = np.linalg.eigh(C)
    amp = Dc @ V[:, -1].conj()
    cors = [float(abs(np.corrcoef(np.vstack([amp.real, ec[:, j].real]))[0, 1])) for j in range(6)]
    out[f"{src} {X}"] = dict(n=len(rows), var_share_m2_to_m9=[float(x) for x in var_share[:8]],
                             corr_topmode_with_e_m=cors, mean_e2_over_N=float(e[:, 0].real.mean()), sd_e2_over_N=float(e[:, 0].real.std(ddof=1)))
    print(src, X, "var share m=2..7:", np.round(var_share[:6], 3), " |corr(top mode, e_m)| m=2..7:", np.round(cors, 2))
json.dump(out, open(os.path.join(HERE, "out", "q6b_summary.json"), "w"), indent=1, allow_nan=False)
