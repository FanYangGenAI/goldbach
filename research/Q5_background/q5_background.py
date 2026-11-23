"""Q5: background contract B_N(z) = R_N(z,z) * int delta_z(t) delta_z(N-t) dt with a Hankel-type
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


ZS = [0.25, -0.25, 0.5, -0.5, 0.5j]

# ---------- T1 ----------
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


X1 = 10**8
om = omega_upto(X1)
cnt = np.bincount(om[2:].astype(np.int64), minlength=64).astype(float)
T1 = {}
for z in ZS:
    exact = np.sum(cnt * np.power(complex(z), np.arange(len(cnt))))
    mod = cum_model(z, X1)
    T1[str(z)] = dict(rel_err=float(abs(mod / exact - 1)), exact=[exact.real, exact.imag], model=[complex(mod).real, complex(mod).imag])
    print("T1", z, T1[str(z)]["rel_err"], flush=True)
del om
print(f"T1 done {time.time()-t0:.0f}s", flush=True)

# ---------- T2/T3 on Q4b polynomials ----------
q4b = json.loads(subprocess.check_output(["git", "show", "gb/Q1-fugacity-sectors:research/Q1_fugacity/out/q4b_summary.json"], cwd=HERE))
rows = []
for scale in ["1000000", "10000000", "100000000"]:
    for r_ in q4b[scale]["rows"]:
        N = r_["N"]; h = np.array(r_["h"], float)
        rN = h[2]
        bb = b2(N)
        old = sing(N) * N / math.log(N) ** 2
        row = dict(N=N, scale=scale, mod3=N % 3, nu2=r_["nu2"], r=rN, b2=bb, old=old,
                   rel_b2=rN / bb - 1, rel_old=rN / old - 1)
        for z in ZS:
            Hz = np.polynomial.polynomial.polyval(complex(z), h)
            Bz = R_N(N, complex(z)) * conv_integral(complex(z), N)
            E = Hz - Bz
            row[f"E_rel_{z}"] = [float((E / Bz).real), float((E / Bz).imag)]
            row[f"E_abs_over_sqrtN_{z}"] = float(abs(E) / math.sqrt(N))
        rows.append(row)
    print(scale, f"{time.time()-t0:.0f}s", flush=True)


def summarize(sel):
    a = np.array([r_["rel_b2"] for r_ in sel]); o = np.array([r_["rel_old"] for r_ in sel])
    d = dict(n=len(sel), mean_rel_b2=float(a.mean()), sem_rel_b2=float(a.std() / math.sqrt(len(a))),
             mean_rel_old=float(o.mean()), sem_rel_old=float(o.std() / math.sqrt(len(o))))
    for z in ZS:
        e = np.array([r_[f"E_rel_{z}"][0] for r_ in sel])
        d[f"E_rel_mean_{z}"] = float(e.mean()); d[f"E_rel_sd_{z}"] = float(e.std())
        d[f"E_abs_sqrtN_median_{z}"] = float(np.median([r_[f"E_abs_over_sqrtN_{z}"] for r_ in sel]))
    return d


summ = {}
for scale in ["1000000", "10000000", "100000000"]:
    sel = [r_ for r_ in rows if r_["scale"] == scale]
    summ[scale] = dict(all=summarize(sel))
    for m3 in (True, False):
        for nu in (1, 2, 3):
            ss = [r_ for r_ in sel if (r_["mod3"] == 0) == m3 and min(r_["nu2"], 3) == nu]
            if len(ss) >= 3:
                summ[scale][f"3|N={m3},nu2={nu}{'+' if nu==3 else ''}"] = summarize(ss)
    a = summ[scale]["all"]
    print(scale, "drift b2: %.5f +- %.5f   old: %.5f +- %.5f" % (a["mean_rel_b2"], a["sem_rel_b2"], a["mean_rel_old"], a["sem_rel_old"]), flush=True)
    for z in ZS:
        print("   z=%s  E/B mean %.4f sd %.4f  |E|/sqrtN median %.3g" % (z, a[f"E_rel_mean_{z}"], a[f"E_rel_sd_{z}"], a[f"E_abs_sqrtN_median_{z}"]), flush=True)
T2 = all(abs(summ[s_]["all"]["mean_rel_b2"]) < 3 * summ[s_]["all"]["sem_rel_b2"] for s_ in summ)
T1ok = all(v_["rel_err"] < 1e-3 for v_ in T1.values())
print("T1:", T1ok, "T2:", T2)
json.dump(dict(T1=T1, T1_pass=T1ok, T2_pass=T2, summary=summ, rows=rows, params=dict(SIG0=SIG0, NG=NG, PMAX=PMAX)),
          open(os.path.join(OUT, "q5_summary.json"), "w"), indent=1, allow_nan=False)
