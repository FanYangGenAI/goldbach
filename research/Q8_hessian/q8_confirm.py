"""Q8: preregistered fresh-sample confirmation of the second-order local Mellin predictor (round-10 review eq. 11).
Uses Q7's frozen code (commit 289f063) for the background, b3 extraction and the linear cross term c3."""
import json, math, os, subprocess, time
import numba as nb
import numpy as np
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
Q7C = "289f063affe39c611e577f7c844a1abf3830bb57"
t0 = time.time()
src = subprocess.check_output(["git", "show", f"{Q7C}:research/Q7_crossterm/q7_cross_term.py"], cwd=HERE).decode("utf-8")
src = src.split("\ndef load(")[0].replace('os.path.dirname(os.path.abspath(__file__))', repr(os.path.join(HERE, "..", "Q7_crossterm")))
src = src.replace("cache=True", "cache=False")
g = {"__name__": "q8"}
exec(compile(src, "q7_cross_term.py", "exec"), g)
bcoeffs = g["bcoeffs"]; discrete_parts = g["discrete_parts"]; omega_upto = g["omega_upto"]
A2TAB = g["A2TAB"]; L0 = g["L0"]; DL = g["DL"]; LG = g["LG"]
a1f = g["a1f"]; a2f = g["a2f"]
q6 = g["g"]  # Q6 namespace exec'd inside Q7
sing = q6["sing"]
print("frozen code loaded", f"{time.time()-t0:.0f}s", flush=True)


@nb.njit(parallel=True, cache=True)
def hcoeffs(Ns, om):
    out = np.zeros((len(Ns), 128), dtype=np.int64)
    for k in nb.prange(len(Ns)):
        N = Ns[k]
        for n in range(2, N - 1):
            out[k, om[n] + om[N - n]] += 1
    return out


def quad2(f, N):
    r1 = integrate.quad(f, 2, N / 2, limit=400); r2 = integrate.quad(f, N / 2, N - 2, limit=400)
    return r1[0] + r2[0], r1[1] + r2[1]


# ---- predictor ell_3 ----
s = np.ones(200001, bool); s[:2] = False
for i in range(2, 448):
    if s[i]: s[i * i :: i] = False
PR = np.nonzero(s)[0]
ODDP = PR[PR > 2].astype(float)
L2_BASE = float(np.sum(np.log(ODDP) ** 2 / (ODDP * (ODDP - 2))))


def factor(N):
    fac = {}; n = N; p = 2
    while p * p <= n:
        while n % p == 0: fac[p] = fac.get(p, 0) + 1; n //= p
        p += 1
    if n > 1: fac[n] = fac.get(n, 0) + 1
    return fac


def L2(N):
    tot = L2_BASE
    for p in factor(N):
        if 2 < p <= 200000:
            tot -= math.log(p) ** 2 / (p * (p - 2))
        tot += -math.log(p) ** 2 / p
    return tot


def a1cut(x):
    return (1 - x ** -0.45) / math.log(x)


def D2a1cut(x):
    L = math.log(x); e = x ** -0.45
    return -0.45**2 * e / L - 0.9 * e / L**2 + 2 * (1 - e) / L**3


def ell3(N):
    I, err = quad2(lambda x: D2a1cut(x) * a1cut(N - x), N)
    return sing(N) * L2(N) * I, sing(N) * abs(L2(N)) * err


# ---- fresh sample ----
used = {r["N"] for r in json.loads(subprocess.check_output(["git", "show", "gb/Q5-background-contract:research/Q5_background/out/q5d_summary.json"], cwd=HERE))["rows"]}
rng = np.random.default_rng(2026112401)
s3, s0 = [], []
while len(s3) < 40 or len(s0) < 40:
    N = int(2 * rng.integers(10**9 // 2, 11 * 10**8 // 2))
    if N in used or N in s3 or N in s0: continue
    if N % 3 == 0 and len(s3) < 40: s3.append(N)
    elif N % 3 != 0 and len(s0) < 40: s0.append(N)
Ns = np.array(s3 + s0, dtype=np.int64)
om = omega_upto(int(1.1e9) + 10)
print("Omega", f"{time.time()-t0:.0f}s", flush=True)
H = hcoeffs(Ns, om)
DP = discrete_parts(Ns, om, A2TAB, L0, DL)
del om
print("counts + discrete sums", f"{time.time()-t0:.0f}s", flush=True)

rows = []
for k, N in enumerate(Ns):
    N = int(N)
    b2, b3 = bcoeffs(N)
    a12, a12err = quad2(lambda t: a1f(t) * a2f(N - t), N)
    c3 = 2 * (DP[k, 1] + DP[k, 2] - 2 * a12)
    l3, l3err = ell3(N)
    h = H[k]
    rows.append(dict(N=N, mod3=N % 3, nu2=(N & -N).bit_length() - 1, h2=int(h[2]), h3=int(h[3]), b2=float(b2), b3=float(b3),
                     S1=float(DP[k, 0]), S12=float(DP[k, 1]), S21=float(DP[k, 2]), a12=float(a12), a12_quad_err=float(a12err),
                     c3=float(c3), ell3=float(l3), ell3_quad_err=float(l3err),
                     target=float((h[3] - b3 - c3) / N), pred=float(l3 / N)))
    if k % 20 == 0: print(k, f"{time.time()-t0:.0f}s", flush=True)

T = []; P_ = []
for grp in (True, False):
    ss = [r for r in rows if (r["mod3"] == 0) == grp]
    t = np.array([r["target"] for r in ss]); p = np.array([r["pred"] for r in ss])
    T += list(t - t.mean()); P_ += list(p - p.mean())
T = np.array(T); P_ = np.array(P_)
rho = float(np.corrcoef(T, P_)[0, 1]); sdr = float(P_.std(ddof=1) / T.std(ddof=1)); R2 = float(1 - np.sum((T - P_) ** 2) / np.sum(T**2))
verdict = "support" if (rho > 0.5 and 0.5 <= sdr <= 2 and R2 > 0.25) else ("reject" if (rho < 0.2 or sdr < 0.1) else "partial")
means = {g_: dict(target=float(np.mean([r["target"] for r in rows if (r["mod3"] == 0) == (g_ == "3|N")])),
                  pred=float(np.mean([r["pred"] for r in rows if (r["mod3"] == 0) == (g_ == "3|N")]))) for g_ in ("3|N", "3!N")}
err_budget = float(max(4 * abs(r["a12_quad_err"]) + abs(r["ell3_quad_err"]) for r in rows) / np.median(Ns) / T.std(ddof=1))
print(f"rho={rho:.4f} sd_ratio={sdr:.4f} R2_pred={R2:.4f} verdict={verdict}")
print("means", means, "max quad-error / residual sd", err_budget)
json.dump(dict(rho=rho, sd_ratio=sdr, R2_pred=R2, verdict=verdict, means=means, quad_error_over_sd=err_budget, rows=rows),
          open(os.path.join(OUT, "q8_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
