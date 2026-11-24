"""O6 post-hoc (exploratory): random-labelling dispersion D of chi before and after subtracting the refined
local prediction (null variance from the observed class proportions of each N)."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "out", "o6_summary.json")))
def nullv(p):
    c = np.array([1, -1, -1, 1.0]); b = np.array([-1, 1, -1, 1.0])
    cb, bb = (p * c).sum(), (p * b).sum()
    S = np.array([[(p*c*c).sum()-cb*cb, (p*c*b).sum()-cb*bb], [(p*c*b).sum()-cb*bb, (p*b*b).sum()-bb*bb]])
    g = np.array([1.0, -2 * bb]); return float(g @ S @ g)
out = {}
for X in sorted({r["X"] for r in D["rows"]}):
    rs = [r for r in D["rows"] if r["X"] == X]
    A = np.array([sum(r["counts"]) for r in rs], float)
    v = np.array([nullv(np.array(r["counts"], float) / sum(r["counts"])) for r in rs])
    s = np.sqrt(v / A)
    obs = np.array([r["chi_obs"] for r in rs]); ref = np.array([r["chi_ref"] for r in rs])
    d0 = obs - obs.mean(); d1 = (obs - ref) - (obs - ref).mean()
    out[str(X)] = dict(D_before=float((d0 / s).std()), D_after=float((d1 / s).std()))
    print(X, out[str(X)])
json.dump(out, open(os.path.join(HERE, "out", "resid_dispersion.json"), "w"), indent=1)
