"""Q5c (exploratory): first-order 'scale-shift' correction to the p-adic fixed-sum factor.
For a local state with p^a || n, p^b || N-n, the single-system density is evaluated at n/p^a instead of n,
giving a relative factor ~ 1 - (z-1) a log p / log N per side. Net correction to log R_N:
  Delta = -(z-1)/log N * sum_p log p * [ d/du log J_p(zu,zu;e)|_{u=1} - 2 d/du log Z_p(zu)|_{u=1} ]."""
import json, math, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "out", "q5_summary.json")))
s = np.ones(10**6 + 1, bool); s[:2] = False
for i in range(2, 1001):
    if s[i]: s[i*i::i] = False
PR = np.nonzero(s)[0].astype(float)

def J(p, e, z):
    v = (p - 2 + z) / (p - z)
    for _ in range(e):
        v = z * z * v / p + (1 - 1 / p)
    return v
def Zp(p, z): return (1 - 1/p) / (1 - z/p)
H = 1e-6
def dlog(f, z):  # d/du log f(z u) at u = 1
    return (np.log(np.asarray(f(z * (1 + H)), dtype=complex)) - np.log(np.asarray(f(z * (1 - H)), dtype=complex))) / (2 * H)

def bracket(p, e, z):
    return dlog(lambda x: J(p, e, x), z) - 2 * dlog(lambda x: Zp(p, x), z)

BASE = {}
def delta(N, z):
    if z not in BASE:
        BASE[z] = complex(np.sum(np.log(PR) * bracket(PR, 0, z)))
    tot = BASE[z]
    n = N; p = 2; fac = {}
    while p * p <= n:
        while n % p == 0: fac[p] = fac.get(p, 0) + 1; n //= p
        p += 1
    if n > 1: fac[n] = fac.get(n, 0) + 1
    for p, e in fac.items():
        if p <= 10**6: tot -= math.log(p) * bracket(float(p), 0, z)
        tot += math.log(p) * bracket(float(p), e, z)
    return -(z - 1) / math.log(N) * tot

ZS = [0.25, -0.25, 0.5, -0.5]
out = {}
for scale in ["1000000", "100000000"]:
    sel = [r for r in d["rows"] if r["scale"] == scale]
    for grp, f in (("3|N", lambda r: r["mod3"] == 0), ("3!N", lambda r: r["mod3"] != 0)):
        ss = [r for r in sel if f(r)]
        line = {}
        for z in ZS:
            old = np.array([r[f"E_rel_{z}"][0] for r in ss])
            new = np.array([(1 + r[f"E_rel_{z}"][0]) / math.exp(np.real(delta(r["N"], z))) - 1 for r in ss])
            line[z] = (round(float(old.mean()), 4), round(float(new.mean()), 4))
        out[f"{scale} {grp}"] = {str(k): v for k, v in line.items()}
        print(scale, grp, len(ss), " z: (old mean E/B, corrected)", line)
json.dump(out, open(os.path.join(HERE, "out", "q5c_summary.json"), "w"), indent=1)
