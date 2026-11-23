"""V3 fit: window-integral model sigma2(window) = (1/H) int_x^{x+H} c t (log t)^k dt, nonlinear LSQ in log space."""
import json, os
import numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "out", "v3_gpu.json")))
rows = d["fixed"]
gl_x, gl_w = np.polynomial.legendre.leggauss(16)

def model(params, x, H):
    lc, k = params
    t = x[:, None] + (gl_x[None, :] + 1) / 2 * H[:, None]
    return np.log(np.exp(lc) * (t * np.log(t) ** k * gl_w[None, :] / 2).sum(1))

def fit(sel):
    x = np.array([r["x"] for r in sel], float); H = np.array([r["H"] for r in sel], float)
    y = np.log([r["sigma2"] for r in sel])
    res = least_squares(lambda p: model(p, x, H) - y, x0=[-2.0, 2.5])
    J = res.jac; s2 = (res.fun ** 2).sum() / max(len(y) - 2, 1)
    cov = np.linalg.inv(J.T @ J) * s2
    return dict(k=float(res.x[1]), c=float(np.exp(res.x[0])), k_se=float(np.sqrt(cov[1, 1])), rms_log_resid=float(np.sqrt((res.fun**2).mean())), js=[r["j"] for r in sel])

out = dict(all=fit(rows), first_half=fit([r for r in rows if r["j"] <= 26]), second_half=fit([r for r in rows if r["j"] >= 26]),
           excl_j20=fit([r for r in rows if r["j"] >= 21]))
# alternative model: sigma2 = x log^2 x (a + b log x)
x = np.array([r["x"] + r["H"] / 2 for r in rows]); L = np.log(x); v = np.array([r["sigma2"] for r in rows])
A = np.vstack([np.ones_like(L), L]).T; coef, res, *_ = np.linalg.lstsq(A, v / (x * L**2), rcond=None)
out["alt_linear_in_logx"] = dict(a=float(coef[0]), b=float(coef[1]), rms_rel=float(np.sqrt(((A @ coef) / (v / (x * L**2)) - 1).__pow__(2).mean())))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "out", "v3_fit.json"), "w"), indent=2)
