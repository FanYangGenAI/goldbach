"""Q5d: confirmation run (fresh sample). Functions copied from q5_background.py (Q5) and q5c_scale_shift.py (Q5c).
Q5: background contract B_N(z) = R_N(z,z) * int delta_z(t) delta_z(N-t) dt with a Hankel-type
single-system density; tests T1-T3 on the 180 saved polynomials of Q4b. See DESIGN.md."""
import json, math, os, time, subprocess
import numba as nb
import numpy as np
from scipy import integrate, special

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SIG0 = 0.55; YMAX = 1 - SIG0; NG = 400; PMAX = 200000
t0 = time.time()

# primes
s = np.ones(10**6 + 1, bool); s[:2] = False
for i in range(2, 1001):
    if s[i]:
        s[i * i :: i] = False
P_ALL = np.nonzero(s)[0].astype(float)
P = P_ALL[P_ALL <= PMAX]
ODD = P_ALL[P_ALL > 2]

# quadrature in v, y = YMAX * v^2
gx, gw = np.polynomial.legendre.leggauss(NG)
v = (gx + 1) / 2; wv = gw / 2
Y = YMAX * v**2; WY = wv * 2 * YMAX * v
SIG = 1 - Y
LOGZETA = np.log(np.abs(special.zeta(SIG)))
# log B_z(sigma) pieces: sum_p [z log(1-p^-s) - log(1 - z p^-s)] + tail (z^2 - z)/2 * E1((2s-1) log P)
PS = P[None, :] ** (-SIG[:, None])            # NG x nP
LOG1MP = np.log1p(-PS).sum(1)
TAIL = special.exp1((2 * SIG - 1) * math.log(PMAX))


def phi(z):
    """phi_z(y) = |zeta(1-y)|^z B_z(1-y) on the y-grid (complex z allowed)."""
    logB = z * LOG1MP - np.log(1 - z * PS).sum(1) + (z * z - z) / 2 * TAIL
    return np.exp(z * LOGZETA + logB)


def delta_weights(z):
    return np.sin(np.pi * z) / np.pi * WY * phi(z)   # delta_z(t) = sum_i c_i t^{-y_i}


def cum_model(z, x):
    c = delta_weights(z)
    return np.sum(c * (x ** (1 - Y) - 2 ** (1 - Y)) / (1 - Y))


def conv_integral(z, N):
    c = delta_weights(z)
    a = Y[:, None]; b = Y[None, :]
    beta = np.exp(special.gammaln(1 - a) + special.gammaln(1 - b) - special.gammaln(2 - a - b))
    I = N ** (1 - a - b) * beta - 2 ** (1 - a) * N ** (-b) / (1 - a) - 2 ** (1 - b) * N ** (-a) / (1 - b)
    return c @ I @ c


def J(p, e, z):
    v_ = (p - 2 + z) / (p - z)
    for _ in range(e):
        v_ = z * z * v_ / p + (1 - 1 / p)
    return v_


def Zp(p, z):
    return (1 - 1 / p) / (1 - z / p)


LOGR_BASE = {}


def R_N(N, z):
    if z not in LOGR_BASE:
        LOGR_BASE[z] = np.sum(np.log(1 - (z - 1) ** 2 / (ODD - 1) ** 2))
    lr = LOGR_BASE[z]
    n = N; p = 2; fac = {}
    while p * p <= n:
        while n % p == 0:
            fac[p] = fac.get(p, 0) + 1; n //= p
        p += 1
    if n > 1:
        fac[n] = fac.get(n, 0) + 1
    for p, e in fac.items():
        if p > 2 and p <= 10**6:
            lr -= np.log(1 - (z - 1) ** 2 / (p - 1) ** 2)
        lr += np.log(J(p, e, z) / Zp(p, z) ** 2)
    return np.exp(lr)


def sing(N):
    return R_N(N, 0.0).real


def b2(N):
    f = lambda t: (1 - t ** (SIG0 - 1)) * (1 - (N - t) ** (SIG0 - 1)) / (math.log(t) * math.log(N - t))
    I1, _ = integrate.quad(f, 2, N / 2, limit=400)
    return sing(N) * 2 * I1



s = np.ones(10**6 + 1, bool); s[:2] = False
for i in range(2, 1001):
    if s[i]: s[i*i::i] = False
PR = np.nonzero(s)[0].astype(float)

def J2(p, e, z):
    v = (p - 2 + z) / (p - z)
    for _ in range(e):
        v = z * z * v / p + (1 - 1 / p)
    return v
def Zp(p, z): return (1 - 1/p) / (1 - z/p)
H = 1e-6
def dlog(f, z):  # d/du log f(z u) at u = 1
    return (np.log(np.asarray(f(z * (1 + H)), dtype=complex)) - np.log(np.asarray(f(z * (1 - H)), dtype=complex))) / (2 * H)

def bracket(p, e, z):
    return dlog(lambda x: J2(p, e, x), z) - 2 * dlog(lambda x: Zp(p, x), z)

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


@nb.njit(cache=True)
def omega_upto2(n):
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

def b2_full(N):
    I, _ = integrate.quad(lambda t: 1 / (math.log(t) * math.log(N - t)), 2, N / 2, limit=400)
    return sing(N) * 2 * I

om = omega_upto2(int(1.1e9) + 10)
print(f"Omega {time.time()-t0:.0f}s", flush=True)
rng = np.random.default_rng(2026112301)
ZS = [0.25, 0.5, -0.25, -0.5, 0.5j]
rows = []
for N0 in [10**7, 10**8, 10**9]:
    Ns = (2 * rng.integers(N0 // 2, int(1.1 * N0) // 2, size=40)).astype(np.int64)
    Hc = hcoeffs(Ns, om)
    for i, N in enumerate(Ns):
        N = int(N); h = Hc[i].astype(float)
        row = dict(N=N, scale=N0, mod3=N % 3, nu2=(N & -N).bit_length() - 1, h=[int(x) for x in Hc[i][:80]])
        bb = b2_full(N); row["rel_b2"] = h[2] / bb - 1
        for z in ZS:
            zc = complex(z)
            Hz = np.polynomial.polynomial.polyval(zc, h)
            Bz = R_N(N, zc) * conv_integral(zc, N) * np.exp(delta(N, zc))
            e = Hz / Bz - 1
            row[f"E_{z}"] = [float(e.real), float(e.imag)]
            row[f"absE_sqrtN_{z}"] = float(abs(Hz - Bz) / math.sqrt(N))
        rows.append(row)
    print(N0, f"{time.time()-t0:.0f}s", flush=True)

def stat(vals):
    v = np.array(vals); return dict(n=len(v), mean=float(v.mean()), sem=float(v.std() / math.sqrt(len(v))), ok=bool(abs(v.mean()) < 3 * v.std() / math.sqrt(len(v))))

summ = {}; P9 = True; P10 = True; P11 = True
for N0 in [10**7, 10**8, 10**9]:
    sel = [r for r in rows if r["scale"] == N0]
    s_ = {"b2": stat([r["rel_b2"] for r in sel])}
    P11 &= s_["b2"]["ok"]
    for z in ZS:
        for gname, g in (("3|N", lambda r: r["mod3"] == 0), ("3!N", lambda r: r["mod3"] != 0)):
            ss = [r for r in sel if g(r)]
            st = stat([r[f"E_{z}"][0] for r in ss]); st["med_absE_sqrtN"] = float(np.median([r[f"absE_sqrtN_{z}"] for r in ss]))
            s_[f"{z} {gname}"] = st
            if z in (0.25, 0.5) and len(ss) >= 8: P9 &= st["ok"]
            if z == -0.25 and N0 >= 10**8 and len(ss) >= 8: P10 &= st["ok"]
    summ[str(N0)] = s_
    print(N0, json.dumps({k: (round(v["mean"], 5), round(v["sem"], 5), v["n"]) for k, v in s_.items()}), flush=True)
print("P9:", P9, "P10:", P10, "P11:", P11)
json.dump(dict(summary=summ, P9=bool(P9), P10=bool(P10), P11=bool(P11), rows=rows), open(os.path.join(OUT, "q5d_summary.json"), "w"), indent=1, allow_nan=False)
