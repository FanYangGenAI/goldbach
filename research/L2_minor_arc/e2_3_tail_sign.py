"""Diagnostic: sign structure of the arithmetic tail S(N) - S_Q(N) (relative to S(N))."""
import os, sys, json
import numpy as np
sys.argv = [sys.argv[0]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2_3_tail_full.py"), encoding="utf-8").read()
g = {"__file__": os.path.abspath(__file__), "__name__": "diag"}
exec(compile(src.split("summary = {}")[0], "e2_3_tail_full.py", "exec"), g)
X = 1 << 24; Nv = np.arange(X // 2, X + 1, 2)
isp = g["prime_sieve"](X)
C2 = 0.6601618158468695739278121100145
fullS = np.full(X + 1, 2 * C2)
for p in np.nonzero(isp)[0][1:]:
    fullS[p::p] *= (p - 1) / (p - 2)
res = {}
for Q in [16, 64, 256, 1024]:
    mu, phi = g["mu_phi"](Q); qs = np.array([q for q in range(1, Q + 1) if mu[q] != 0], dtype=np.int64)
    SQ = g["sing_partial"](Nv.astype(np.int64), qs, mu, phi, 0)
    rel = (fullS[Nv] - SQ) / fullS[Nv]
    res[Q] = dict(min_rel=float(rel.min()), max_rel=float(rel.max()), frac_negative=float((rel < 0).mean()),
                  q01=float(np.quantile(rel, 0.01)), q99=float(np.quantile(rel, 0.99)), Q_times_min=float(Q * rel.min()))
    print(Q, res[Q])
json.dump(res, open(os.path.join(g["OUT"], "e2_3_tail_sign.json"), "w"), indent=2)
