"""V2: (a) recompute E3.3 ratios with the missing -N*S term (from frozen JSON, no rerun);
(b) exact small-N identity R^2 = diag + antidiag - overlap + offdiag, checked by brute force."""
import json, math, os, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
raw = subprocess.check_output(["git", "show", "b0e8c56:research/L3c_4tuples/out/e3_3_rows.json"], cwd=HERE)
rows = json.loads(raw)["rows"]
fix = {}
for r in rows:
    # stored sigma2_HL = mean((d+o)/S); the design formula subtracts N*S^2, i.e. N*S after dividing by S
    corrected = np.mean([(d + o) / S - N * S for (N, S, d, o, m) in r["samples"]])
    fix[r["j"]] = dict(reported_ratio=r["ratio"], corrected_ratio=float(corrected / r["sigma2_obs"]))
print("L3c corrected:", fix)

# (b) brute-force identity for small N
X = 1 << 13
s = np.ones(X + 1, bool); s[:2] = False; s[4::2] = False
for i in range(3, int(X**0.5) + 1, 2):
    if s[i]: s[i*i::2*i] = False
th = np.where(s, np.log(np.maximum(np.arange(X + 1), 1)), 0.0)
errs = []
for N in [1000, 2048, 4096, 6000, 8190, 8192]:
    g = th[: N + 1] * th[N::-1]
    R = g.sum()
    C0 = (g * g).sum()
    overlap = g[N // 2] ** 2 if N % 2 == 0 else 0.0
    D = 2 * C0 - overlap
    # brute-force off-diagonal: pairs (a,a') with a' != a and a' != N-a
    G = np.outer(g, g)
    idx = np.arange(N + 1)
    mask = np.ones_like(G, bool); mask[idx, idx] = False; mask[idx, N - idx] = False
    off = G[mask].sum()
    errs.append(abs(D + off - R * R) / (R * R))
print("identity max rel err:", max(errs))
json.dump(dict(l3c_corrected=fix, identity_max_rel_err=float(max(errs)), V2_identity=bool(max(errs) < 1e-10)),
          open(os.path.join(HERE, "out", "v2_summary.json"), "w"), indent=2)
