"""Q4b: zero counts of H_N(z) and of the untruncated analytic model M_N(z) in the disc |z+1|<0.45
(argument principle), Rouche ratio on the boundary, grouped by (N mod 3, nu_2(N)). See DESIGN.md."""
import json, math, os, time
from collections import Counter, defaultdict
import numba as nb
import numpy as np
from scipy.special import rgamma

HERE = os.path.dirname(os.path.abspath(__file__))
R = 0.45; NB = 4096; PMAX = 10**6
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
def hcoeffs(Ns, om):
    out = np.zeros((len(Ns), 128), dtype=np.int64)
    for k in nb.prange(len(Ns)):
        N = Ns[k]
        for n in range(2, N - 1):
            out[k, om[n] + om[N - n]] += 1
    return out


s = np.ones(PMAX + 1, bool); s[:2] = False
for i in range(2, int(PMAX**0.5) + 1):
    if s[i]:
        s[i * i :: i] = False
primes = np.nonzero(s)[0].astype(float)
odd = primes[primes > 2]

zb = -1 + R * np.exp(2j * np.pi * np.arange(NB) / NB)          # boundary points
# base: prod over odd p <= PMAX of J_p(z,z;0) (1-1/p)^{2z-2}, J_p(z,z;0) = (p-2+z)/(p-z)
logbase = np.zeros(NB, complex)
for p in odd:
    logbase += np.log((p - 2 + zb) / (p - zb)) + (2 * zb - 2) * math.log(1 - 1 / p)
print(f"base {time.time()-t0:.0f}s", flush=True)


def Jdiag(p, e, z):
    v = (p - 2 + z) / (p - z)
    for _ in range(e):
        v = z * z * v / p + (1 - 1 / p)
    return v


def factor(N):
    fac = {}; n = N; p = 2
    while p * p <= n:
        while n % p == 0:
            fac[p] = fac.get(p, 0) + 1; n //= p
        p += 1
    if n > 1:
        fac[n] = fac.get(n, 0) + 1
    return fac


def model_on_boundary(N):
    la = logbase.copy()
    for p, e in factor(N).items():
        if p == 2:
            la += np.log(Jdiag(2, e, zb)) + (2 * zb - 2) * math.log(0.5)
        elif p <= PMAX:
            la += np.log(Jdiag(p, e, zb)) - np.log(Jdiag(p, 0, zb))
        else:
            la += np.log(Jdiag(p, e, zb)) + (2 * zb - 2) * math.log(1 - 1 / p)
    L = math.log(N)
    return N * np.exp((2 * zb - 2) * math.log(L) + la) * rgamma(zb) ** 2


def winding(f):
    d = np.diff(np.unwrap(np.angle(np.append(f, f[0]))))
    return int(round(d.sum() / (2 * np.pi)))


om = omega_upto(int(1.1e8) + 10)
print(f"Omega {time.time()-t0:.0f}s", flush=True)
rng = np.random.default_rng(51)
res = {}
for N0 in [10**6, 10**7, 10**8]:
    Ns = (2 * rng.integers(N0 // 2, int(1.1 * N0) // 2, size=60)).astype(np.int64)
    Hc = hcoeffs(Ns, om)
    rows = []
    for i, N in enumerate(Ns):
        N = int(N)
        c = Hc[i].astype(float)
        H = np.polynomial.polynomial.polyval(zb, c)
        M = model_on_boundary(N)
        nu2 = (N & -N).bit_length() - 1
        rows.append(dict(N=N, mod3=N % 3, nu2=nu2, nH=winding(H), nM=winding(M),
                         rouche=float(np.max(np.abs(H - M) / np.abs(M))), h=[int(x) for x in Hc[i][:70]]))
    grp = defaultdict(list)
    for r_ in rows:
        grp[(r_["mod3"] == 0, min(r_["nu2"], 3))].append(r_)
    summ = {}
    for key, rs in sorted(grp.items()):
        summ[f"3|N={key[0]},nu2={key[1]}{'+' if key[1]==3 else ''}"] = dict(n=len(rs), nM=dict(Counter(r_["nM"] for r_ in rs)), nH=dict(Counter(r_["nH"] for r_ in rs)),
                                                                           agree=float(np.mean([r_["nH"] == r_["nM"] for r_ in rs])),
                                                                           rouche_ok=float(np.mean([r_["rouche"] < 1 for r_ in rs])))
    agree = float(np.mean([r_["nH"] == r_["nM"] for r_ in rows]))
    rok = float(np.mean([r_["rouche"] < 1 for r_ in rows]))
    res[str(N0)] = dict(groups=summ, agree=agree, rouche_ok=rok, rows=rows)
    print(N0, "agree", agree, "rouche_ok", rok, flush=True)
    for k, v in summ.items():
        print("   ", k, {kk: vv for kk, vv in v.items()}, flush=True)
m3 = [r_["nM"] for r_ in res["100000000"]["rows"] if r_["mod3"] == 0]
n3 = [r_["nM"] for r_ in res["100000000"]["rows"] if r_["mod3"] != 0]
mode = lambda a: Counter(a).most_common(1)[0][0]
res["P7"] = bool(mode(m3) == 2 and mode(n3) == 3)
res["P8"] = bool(res["100000000"]["agree"] >= 0.8)
print("P7:", res["P7"], "P8:", res["P8"])
json.dump(res, open(os.path.join(HERE, "out", "q4b_summary.json"), "w"), indent=1, allow_nan=False)
