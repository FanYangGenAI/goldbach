"""O11 post-hoc (exploratory): is the residual chi_obs - chi_ref still correlated with local features of N?"""
import json, math, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "out", "o11_summary.json")))
C2 = 0.6601618158468695739278121100145
def opf(N):
    ps = []; m = N
    while m % 2 == 0: m //= 2
    p = 3
    while p * p <= m:
        if m % p == 0:
            ps.append(p)
            while m % p == 0: m //= p
        p += 2
    if m > 1: ps.append(m)
    return ps
def r2(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X)); b, *_ = np.linalg.lstsq(X, y, rcond=None); return float(1 - (y - X @ b).var() / y.var())
out = {}
for X in sorted({r["X"] for r in D["rows"]}):
    rs = [r for r in D["rows"] if r["X"] == X]
    y0 = np.array([r["chi_obs"] for r in rs]); y1 = y0 - np.array([r["chi_ref"] for r in rs])
    F = []
    for r in rs:
        ps = opf(r["N"]); z = r["z"]
        F.append([sum(1 for p in ps if p <= z), len(r["big"]), math.log(2 * C2) + sum(math.log((p - 1) / (p - 2)) for p in ps),
                  sum(1 / p for p in r["big"]) * z])
    F = np.array(F).T
    out[str(X)] = dict(R2_before=r2(y0, F), R2_after=r2(y1, F), R2_after_each=[r2(y1, [f]) for f in F], chance=F.shape[0] / len(rs))
    print(X, out[str(X)])
json.dump(out, open(os.path.join(HERE, "out", "resid_locality.json"), "w"), indent=1)
