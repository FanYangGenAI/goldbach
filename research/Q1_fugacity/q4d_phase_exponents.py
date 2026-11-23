"""Q4d (exploratory): error exponent alpha(theta) for z=w=0.5 e^{i theta}, |G_N - M_N| ~ N^alpha."""
import json, math, os, time
import numba as nb
import numpy as np
from scipy.special import rgamma
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "q2b_phase_decay.py"), encoding="utf-8").read().split("NMAX = int(1.1e9)")[0]
g = {"__name__": "q4d", "__file__": os.path.join(HERE, "q2b_phase_decay.py")}
exec(compile(src.replace("cache=True", "cache=False"), "q2b", "exec"), g)
t0 = time.time()
RHO = 0.5
TH = np.linspace(0, np.pi, 9)
ZS = RHO * np.exp(1j * TH)

@nb.njit(parallel=True)
def Gc(Ns, om, zs):
    K = len(zs)
    out = np.zeros((len(Ns), K), dtype=np.complex128)
    tab = np.empty((K, 130), dtype=np.complex128)
    for a in range(K):
        for t in range(130):
            tab[a, t] = zs[a] ** t
    for k in nb.prange(len(Ns)):
        N = Ns[k]
        acc = np.zeros(K, dtype=np.complex128)
        for n in range(2, N - 1):
            s = om[n] + om[N - n]
            for a in range(K):
                acc[a] += tab[a, s]
        for a in range(K):
            out[k, a] = acc[a]
    return out

primes = g["primes"]; odd = primes[primes > 2].astype(float)
logbase = np.zeros(len(ZS), complex)
for p in odd:
    logbase += np.log((p - 2 + ZS) / (p - ZS)) + (2 * ZS - 2) * math.log(1 - 1 / p)

def Jd(p, e, z):
    v = (p - 2 + z) / (p - z)
    for _ in range(e):
        v = z * z * v / p + (1 - 1 / p)
    return v

def model(N):
    la = logbase.copy(); n = N; p = 2; fac = {}
    while p * p <= n:
        while n % p == 0: fac[p] = fac.get(p, 0) + 1; n //= p
        p += 1
    if n > 1: fac[n] = fac.get(n, 0) + 1
    for p, e in fac.items():
        if p == 2: la += np.log(Jd(2, e, ZS)) + (2 * ZS - 2) * math.log(0.5)
        elif p <= g["PMAX"]: la += np.log(Jd(p, e, ZS)) - np.log(Jd(p, 0, ZS))
        else: la += np.log(Jd(p, e, ZS)) + (2 * ZS - 2) * math.log(1 - 1 / p)
    L = math.log(N)
    return N * np.exp((2 * ZS - 2) * math.log(L) + la) * rgamma(ZS) ** 2

om = g["omega_upto"](int(1.1e9) + 10)
rng = np.random.default_rng(71)
scales = [10**6, 10**7, 10**8, 10**9]
medE = []; medM = []
for N0 in scales:
    Ns = (2 * rng.integers(N0 // 2, int(1.1 * N0) // 2, size=20)).astype(np.int64)
    G = Gc(Ns, om, ZS)
    Mm = np.array([model(int(N)) for N in Ns])
    medE.append(np.median(np.abs(G - Mm), 0)); medM.append(np.median(np.abs(Mm), 0))
    print(N0, f"{time.time()-t0:.0f}s", flush=True)
medE = np.array(medE); medM = np.array(medM)
lN = np.log(np.array(scales, float))
out = []
for a, th in enumerate(TH):
    al = float(np.polyfit(lN, np.log(medE[:, a]), 1)[0])
    out.append(dict(theta=float(th), alpha_error=al, E_over_N_1e9=float(medE[-1, a] / 1e9), M_over_N_1e9=float(medM[-1, a] / 1e9)))
    print("theta=%.2f pi  alpha=%.3f  |E|/N=%.2e  |M|/N=%.2e" % (th / np.pi, al, medE[-1, a] / 1e9, medM[-1, a] / 1e9))
json.dump(out, open(os.path.join(HERE, "out", "q4d_summary.json"), "w"), indent=1, allow_nan=False)
