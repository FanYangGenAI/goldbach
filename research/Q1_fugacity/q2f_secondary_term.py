"""Q2f (exploratory): integral main term + one-variable Selberg-Delange secondary density term on each side. Based on Q2e: integral (local-density) main term instead of N(log N)^{z+w-2}. Based on Q2d: signed positive-axis error, 100 N per scale; PMAX sensitivity. Q2b base: relative error of the two-variable Selberg-Delange model at z=w=+rho and z=w=-rho vs N. See DESIGN.md."""
import json, math, os, time
import numba as nb
import numpy as np
from scipy.special import rgamma
HERE = os.path.dirname(os.path.abspath(__file__))
RHO = 0.5
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
def Gvals(Ns, om, x):
    out = np.zeros((len(Ns), 2))
    pw = np.empty(130); pm = np.empty(130)
    for t in range(130):
        pw[t] = x ** t; pm[t] = (-x) ** t
    for k in nb.prange(len(Ns)):
        N = Ns[k]; a = 0.0; b = 0.0
        for n in range(2, N - 1):
            s = om[n] + om[N - n]
            a += pw[s]; b += pm[s]
        out[k, 0] = a; out[k, 1] = b
    return out

PMAX = 10**6
s = np.ones(PMAX + 1, bool); s[:2] = False
for i in range(2, int(PMAX**0.5) + 1):
    if s[i]: s[i*i::i] = False
primes = np.nonzero(s)[0]

def J(p, e, z, w):
    Zp = (1 - 1/p) / (1 - z/p); Wp = (1 - 1/p) / (1 - w/p)
    v = 1 - 2/p + (z*Zp + w*Wp)/p
    for _ in range(e):
        v = z*w*v/p + (1 - 1/p)
    return v

def logbase(x):
    return float(sum(math.log(J(p, 0, x, x)) + (2*x - 2)*math.log(1 - 1/p) for p in primes if p > 2))
BASE = {x: logbase(x) for x in (RHO, -RHO)}

def model(N, x):
    la = BASE[x]; n = N; p = 2; fac = {}
    while p*p <= n:
        while n % p == 0: fac[p] = fac.get(p, 0) + 1; n //= p
        p += 1
    if n > 1: fac[n] = fac.get(n, 0) + 1
    sign = 1.0
    for p, e in fac.items():
        if 2 < p <= PMAX: la -= math.log(J(p, 0, x, x))
        if p == 2: la += (2*x - 2)*math.log(0.5)
        v = J(p, e, x, x)
        if v < 0: sign = -sign
        la += math.log(abs(v))
    L = math.log(N)
    return sign * N * math.exp((2*x - 2)*math.log(L) + la) * rgamma(x)**2


NMAX = int(1.1e9)
om = omega_upto(NMAX + 10)
for x in (0.25, 0.75): BASE[x] = logbase(x)

from scipy import integrate
def kappa(z):
    return 1 + z * 0.5772156649015329 + z * (1 - z) * float(np.sum(np.log(primes) / ((primes - 1.0) * (primes - z))))
KAP = {x: kappa(x) for x in (0.25, 0.5, 0.75)}
print("kappa:", KAP, flush=True)
def model_int(N, x):
    """replace N (log N)^{2x-2} by int_2^{N-2} (log t)^{x-1} (log(N-t))^{x-1} dt"""
    base = model(N, x) / (N * math.log(N) ** (2 * x - 2))
    k = (x - 1) * KAP[x]
    f = lambda u: (math.log(u * N)) ** (x - 1) * (math.log((1 - u) * N)) ** (x - 1) * (1 + k / math.log(u * N)) * (1 + k / math.log((1 - u) * N))
    I, _ = integrate.quad(f, 2 / N, 0.5, limit=200)
    return base * 2 * N * I
rng = np.random.default_rng(41)
res = {}
for N0 in [10**6, 10**7, 10**8, 10**9]:
    Ns = (2 * rng.integers(N0//2, int(1.1*N0)//2, size=100)).astype(np.int64)
    row = {}
    for x in (0.25, 0.5, 0.75):
        G = Gvals(Ns, om, x)[:, 0]
        e = np.array([G[i]/model_int(int(N), x) - 1 for i, N in enumerate(Ns)])
        row[x] = dict(mean=float(e.mean()), sd=float(e.std()))
    res[N0] = row
    print(N0, {x: (round(v["mean"], 4), round(v["sd"], 4)) for x, v in row.items()}, f"{time.time()-t0:.0f}s", flush=True)
json.dump({str(k): v for k, v in res.items()}, open(os.path.join(HERE, "out", "q2f_summary.json"), "w"), indent=1)
