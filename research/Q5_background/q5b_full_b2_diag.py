"""Q5b (exploratory diagnosis): drift of r(N)/b2 with the untruncated background coefficient
b2_full = S(N) int_2^{N-2} dt/(log t log(N-t)); tests whether the residual T2 drift is the sigma0 truncation."""
import json, math, os
import numpy as np
from scipy import integrate
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "out", "q5_summary.json")))
res = {}
for scale in ["1000000", "10000000", "100000000"]:
    sel = [r for r in d["rows"] if r["scale"] == scale]
    rel = []
    for r in sel:
        N = r["N"]; S = r["old"] * math.log(N) ** 2 / N
        I, _ = integrate.quad(lambda t: 1 / (math.log(t) * math.log(N - t)), 2, N / 2, limit=400)
        rel.append(r["r"] / (S * 2 * I) - 1)
    rel = np.array(rel)
    trunc = np.array([r["rel_b2"] for r in sel])
    res[scale] = dict(mean_full=float(rel.mean()), sem_full=float(rel.std() / math.sqrt(len(rel))), mean_trunc=float(trunc.mean()))
    print(scale, res[scale])
json.dump(res, open(os.path.join(HERE, "out", "q5b_summary.json"), "w"), indent=1)
