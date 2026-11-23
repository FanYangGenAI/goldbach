"""Q2: phase map of the partition-function error E_N = G_N - G_model on |z|=|w|=rho. See DESIGN.md."""
import json, math, os, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "q1b_euler_model.py"), encoding="utf-8").read().split(chr(10) + "NMAX = 10**8" + chr(10))[0]
g = {"__name__": "q2", "__file__": os.path.join(HERE, "q1b_euler_model.py")}
exec(compile(src, "q1b_euler_model.py", "exec"), g)
M = g["M"]; KMAX = g["KMAX"]; primes = g["primes"]; J = g["J"]; ang = g["ang"]
from scipy.special import rgamma
t0 = time.time()

BASE = {}

def model_grid(N, rho):
    Z = rho * ang[:, None] * np.ones((1, M)); W = rho * np.ones((M, 1)) * ang[None, :]
    if rho not in BASE:
        b = np.zeros((M, M), complex)
        for p in primes:
            b += np.log(J(p, 0, Z, W)) + (Z + W - 2) * math.log(1 - 1 / p)
        BASE[rho] = b
    logA = BASE[rho].copy()
    n = N; p = 2; fac = {}
    while p * p <= n:
        while n % p == 0:
            fac[p] = fac.get(p, 0) + 1; n //= p
        p += 1
    if n > 1: fac[n] = fac.get(n, 0) + 1
    for p, e in fac.items():
        if p <= g["PMAX"]: logA -= np.log(J(p, 0, Z, W))
        logA += np.log(J(p, e, Z, W))
    L = math.log(N)
    return N * np.exp((Z + W - 2) * math.log(L) + logA) * rgamma(Z) * rgamma(W)

om = g["g"]["omega_sieve"](int(1.1 * 10**8) + 10)
rng = np.random.default_rng(11)
res = {}
RHOS = [0.25, 0.5, 0.75, 1.0, 1.5]
for N0 in [10**6, 10**8]:
    Ns = (2 * rng.integers(N0 // 2, int(1.1 * N0) // 2, size=30)).astype(np.int64)[:5]
    T = g["g"]["sector_tables"](Ns, om)
    out = []
    for i, N in enumerate(Ns):
        N = int(N); c11 = g["model_coeffs"](N)[1, 1]
        rows = []
        for rho in RHOS:
            Gm = model_grid(N, rho)
            V = (rho * ang)[:, None] ** np.arange(KMAX)[None, :]
            Go = V @ T[i].astype(float) @ V.T
            E = Go - Gm
            rel = np.abs(E) / np.abs(Gm)
            a, b = np.unravel_index(np.argmax(rel), rel.shape)
            th = np.angle(ang)
            rows.append(dict(rho=rho, budget=float(np.abs(E).max() / (rho**2 * c11)), max_rel=float(rel.max()),
                             argz_at_max=float(th[a]), argw_at_max=float(th[b]),
                             rel_at_pos_real=float(rel[0, 0]), rel_at_neg_real=float(rel[M // 2, M // 2])))
        out.append(dict(N=N, rows=rows))
        print(N, [(r["rho"], round(r["budget"], 4), round(r["max_rel"], 4), round(r["argz_at_max"], 2), round(r["argw_at_max"], 2), round(r["rel_at_pos_real"], 5), round(r["rel_at_neg_real"], 4)) for r in rows], f"{time.time()-t0:.0f}s", flush=True)
    res[N0] = out
p4 = [abs(r["argz_at_max"]) > 2 * math.pi / 3 and abs(r["argw_at_max"]) > 2 * math.pi / 3 for o in res[10**8] for r in o["rows"] if r["rho"] in (0.5, 1.0)]
res["P4"] = bool(all(p4)); res["P4_fraction"] = float(np.mean(p4))
print("P4:", res["P4"], res["P4_fraction"])
json.dump(res, open(os.path.join(HERE, "out", "q2_summary.json"), "w"), indent=1)
