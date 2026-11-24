"""E1: entanglement entropy / mutual information of the fixed-sum Omega-sector table. See DESIGN.md.
Frozen dependency: Q1b model code at commit 32f5fc0 (read via git show)."""
import json, math, os, subprocess, time
import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
Q1C = "32f5fc08b09a4399bff7a2e97dddbbd4c62c97ed"
K = 21
t0 = time.time()


@nb.njit(cache=True)
def omega_upto(n):
    spf = np.zeros(n + 1, dtype=np.int32)
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            j = np.int64(i) * i
            while j <= n:
                if spf[j] == 0:
                    spf[j] = i
                j += i
    om = np.zeros(n + 1, dtype=np.int8)
    for m in range(2, n + 1):
        om[m] = om[m // spf[m]] + 1
    return om


@nb.njit(parallel=True, cache=True)
def tables(Ns, om):
    out = np.zeros((len(Ns), 40, 40), dtype=np.int64)
    for k in nb.prange(len(Ns)):
        N = Ns[k]
        for n in range(2, N - 1):
            out[k, om[n], om[N - n]] += 1
    return out


def ent(T, kmax=K):
    T = np.maximum(np.asarray(T, float)[:kmax, :kmax], 0)
    P = T / T.sum()
    s = np.linalg.svd(np.sqrt(P), compute_uv=False) ** 2
    s = s[s > 1e-300]
    S = float(-(s * np.log(s)).sum())
    r = P.sum(1); c = P.sum(0)
    H = lambda x: float(-(x[x > 0] * np.log(x[x > 0])).sum())
    I = H(r) + H(c) - H(P.ravel())
    return S, I


def ent_without11(T):
    T = np.asarray(T, float)[:K, :K].copy()
    r = T.sum(1); c = T.sum(0); tot = T.sum()
    T[1, 1] = r[1] * c[1] / tot
    return ent(T)[0]


# frozen Q1b model (base Euler product computed once)
src = subprocess.check_output(["git", "show", f"{Q1C}:research/Q1_fugacity/q1b_euler_model.py"], cwd=HERE).decode("utf-8")
src = src.split("\nNMAX = 10**8\n")[0].replace("cache=True)", "cache=False)")
src = src.replace('src = open(os.path.join(HERE, "q1q3_sectors_zeros.py"), encoding="utf-8").read()',
                  'src = subprocess.check_output(["git", "show", "' + Q1C + ':research/Q1_fugacity/q1q3_sectors_zeros.py"], cwd=HERE).decode("utf-8")')
g = {"__name__": "e1"}
exec(compile("import subprocess\nHERE = " + repr(HERE) + "\n" + src.replace("HERE = os.path.dirname(os.path.abspath(__file__))", ""), "q1b", "exec"), g)
model_coeffs = g["model_coeffs"]
print("frozen model ready", f"{time.time()-t0:.0f}s", flush=True)

NMAX = int(1.1e8)
om = omega_upto(NMAX + 10)
# locality-preserving shuffle: permute Omega within (block of 2^16, residue mod 210)
rng = np.random.default_rng(20261124)
n_idx = np.arange(len(om), dtype=np.int64)
keys = (n_idx >> 16) * 210 + (n_idx % 210)
base = np.argsort(keys, kind="stable")
perm = np.lexsort((rng.random(len(om)), keys))
om_null = np.empty_like(om); om_null[base] = om[perm]
om_null[:2] = 0
del keys, base, perm, n_idx
print("omega + shuffled", f"{time.time()-t0:.0f}s", flush=True)

rng2 = np.random.default_rng(20261125)
rows = []
for X in [10**6, 10**7, 10**8]:
    Ns = (2 * rng2.integers(X // 2, int(1.1 * X) // 2, size=30)).astype(np.int64)
    T = tables(Ns, om); Tn = tables(Ns, om_null)
    for i, N in enumerate(Ns):
        N = int(N)
        Tm = np.maximum(model_coeffs(N)[:K, :K], 0)
        S, I = ent(T[i]); Sn, In = ent(Tn[i]); Sm, Im = ent(Tm)
        S12 = ent(T[i], 13)[0]
        rows.append(dict(X=X, N=N, mod3=N % 3, S=S, I=I, S_null=Sn, I_null=In, S_model=Sm, I_model=Im,
                         S_trunc12=S12, dS11=S - ent_without11(T[i]), T=T[i][:K, :K].tolist()))
    print(X, f"{time.time()-t0:.0f}s", flush=True)

summ = {}
for X in [10**6, 10**7, 10**8]:
    rs = [r for r in rows if r["X"] == X]
    f = lambda key: float(np.median([r[key] for r in rs]))
    summ[str(X)] = dict(S=f("S"), I=f("I"), S_null=f("S_null"), S_model=f("S_model"), I_null=f("I_null"), I_model=f("I_model"),
                        rel_model_err=float(np.median([abs(r["S"] - r["S_model"]) / r["S"] for r in rs])),
                        null_ratio=float(np.median([r["S_null"] / r["S"] for r in rs])),
                        dS11_frac=float(np.median([r["dS11"] / r["S"] for r in rs])),
                        trunc_sens=float(np.median([abs(r["S"] - r["S_trunc12"]) / r["S"] for r in rs])))
    print(X, json.dumps({k: round(v, 6) for k, v in summ[str(X)].items()}), flush=True)
s8 = summ[str(10**8)]
H_local = bool(s8["rel_model_err"] < 0.05 and s8["null_ratio"] > 0.8)
print("H_local:", H_local)
json.dump(dict(summary=summ, H_local=H_local, rows=rows), open(os.path.join(OUT, "e1_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
