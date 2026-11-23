"""Q4c (exploratory): fluctuation exponent map, |G_N(x,x) - M_N(x)| ~ N^alpha(x) along the real axis. Based on Q2b: relative error of the two-variable Selberg-Delange model at z=w=+rho and z=w=-rho vs N. See DESIGN.md."""
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
XS = [0.25, 0.5, 0.75]
for x in XS + [-x for x in XS]: BASE[x] = logbase(x)
rng = np.random.default_rng(61)
res = {}
scales = [10**6, 10**7, 10**8, 10**9]
med = {x: [] for x in [0.25, 0.5, 0.75, -0.25, -0.5, -0.75, -1.0]}
mainsz = {x: [] for x in med}
for N0 in scales:
    Ns = (2 * rng.integers(N0//2, int(1.1*N0)//2, size=20)).astype(np.int64)
    for x in XS + [1.0]:
        G = Gvals(Ns, om, x)
        for sgn, col in ((1, 0), (-1, 1)):
            xx = sgn * x
            if xx == 1.0: continue
            if xx == -1.0:
                E = np.abs(G[:, col]); Mv = np.zeros(len(Ns))
            else:
                Mv = np.array([model(int(N), xx) for N in Ns]); E = np.abs(G[:, col] - Mv)
            med[xx].append(float(np.median(E))); mainsz[xx].append(float(np.median(np.abs(Mv))))
lN = np.log(np.array(scales, float))
out = {}
for x in med:
    a = float(np.polyfit(lN, np.log(med[x]), 1)[0])
    out[x] = dict(alpha_error=a, median_abs_error=med[x], median_abs_main=mainsz[x])
    print("x=%+.2f  alpha(error)=%.3f  |E|/N at 1e9=%.2e  |M|/N at 1e9=%.2e" % (x, a, med[x][-1]/1e9, mainsz[x][-1]/1e9), flush=True)
json.dump({str(k): v for k, v in out.items()}, open(os.path.join(HERE, "out", "q4c_summary.json"), "w"), indent=1, allow_nan=False)
