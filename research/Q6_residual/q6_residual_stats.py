"""Q6 (exploratory): background v2 (patched density + analytic first-order local term) and residual
statistics on the 300 saved integer polynomials. See DESIGN.md."""
import json, math, os, subprocess, time
import numpy as np
from scipy import special, integrate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SIG0 = 0.55; A = 1 - SIG0; NG = 400; PMAX = 200000
t0 = time.time()

s = np.ones(10**6 + 1, bool); s[:2] = False
for i in range(2, 1001):
    if s[i]: s[i * i :: i] = False
P_ALL = np.nonzero(s)[0].astype(float)
P = P_ALL[P_ALL <= PMAX]; ODD = P_ALL[P_ALL > 2]
gx, gw = np.polynomial.legendre.leggauss(NG)
v = (gx + 1) / 2; wv = gw / 2
Y = A * v**2; WY = wv * 2 * A * v; SIG = 1 - Y
LOGZETA = np.log(np.abs(special.zeta(SIG)))
PS = P[None, :] ** (-SIG[:, None]); LOG1MP = np.log1p(-PS).sum(1)
TAIL = special.exp1((2 * SIG - 1) * math.log(PMAX))
BETA = np.exp(special.gammaln(1 - Y[:, None]) + special.gammaln(1 - Y[None, :]) - special.gammaln(2 - Y[:, None] - Y[None, :]))
CW = {}


def cweights(z):
    if z not in CW:
        logB = z * LOG1MP - np.log(1 - z * PS).sum(1) + (z * z - z) / 2 * TAIL
        CW[z] = np.sin(np.pi * z) / np.pi * WY * np.exp(z * LOGZETA + logB)
    return CW[z]


def Ibeta(N):
    a = Y[:, None]; b = Y[None, :]
    return N ** (1 - a - b) * BETA - 2 ** (1 - a) * N ** (-b) / (1 - a) - 2 ** (1 - b) * N ** (-a) / (1 - b)


# fixed quadrature in v for u-integrals on [2/N, 1/2] (u = e^v) and [1/2, 1-2/N] (1-u = e^v)
qx, qw = np.polynomial.legendre.leggauss(200)


def side_nodes(N):
    lo, hi = math.log(2 / N), math.log(0.5)
    vv = lo + (qx + 1) / 2 * (hi - lo); ww = qw / 2 * (hi - lo)
    u = np.exp(vv); return u, ww * u            # du = u dv


def corr_integrals(N, c):
    """J1 = int delta(N-t) t^{-A}/log t dt ; J2 = int t^{-A}(N-t)^{-A}/(log t log(N-t)) dt."""
    u, du = side_nodes(N)
    t = N * u; s_ = N - t
    g = lambda x: x ** (-A) / np.log(x)
    dl = lambda x: (x[:, None] ** (-Y[None, :])) @ c
    J1 = N * (np.sum(du * dl(s_) * g(t)) + np.sum(du * dl(t) * g(s_)))
    J2 = 2 * N * np.sum(du * g(t) * g(s_))
    return J1, J2


def J(p, e, z):
    x = (p - 2 + z) / (p - z)
    for _ in range(e):
        x = z * z * x / p + (1 - 1 / p)
    return x


def Zp(p, z): return (1 - 1 / p) / (1 - z / p)


def factor(N):
    fac = {}; n = N; p = 2
    while p * p <= n:
        while n % p == 0: fac[p] = fac.get(p, 0) + 1; n //= p
        p += 1
    if n > 1: fac[n] = fac.get(n, 0) + 1
    return fac


BASE = {}


def RT(N, z):
    """R_N(z) and T_N(z) = sum_p log p * z d/dz log r_p(z)."""
    if z not in BASE:
        w = z - 1
        BASE[z] = (np.sum(np.log(1 - w * w / (ODD - 1) ** 2)),
                   np.sum(np.log(ODD) * z * (-2 * w) / ((ODD - 1) ** 2 - w * w)))
    lr, T = BASE[z]
    h = 1e-6
    for p, e in factor(N).items():
        if p > 2 and p <= 10**6:
            w = z - 1
            lr -= np.log(1 - w * w / (p - 1) ** 2); T -= math.log(p) * z * (-2 * w) / ((p - 1) ** 2 - w * w)
        r = lambda x: J(p, e, x) / Zp(p, x) ** 2
        lr += np.log(r(z))
        T += math.log(p) * z * (np.log(r(z + h)) - np.log(r(z - h))) / (2 * h)
    return np.exp(lr), T


def background(N, z):
    c = cweights(z); IB = Ibeta(N)
    I0 = c @ IB @ c
    J1, J2 = corr_integrals(N, c)
    I2 = I0 + 2 * z * J1 + z * z * J2
    I1 = (c * (-Y)) @ IB @ c
    R, T = RT(N, z)
    Bv2 = R * (I2 - T * I1)
    # version A (Q5d): truncated density, exp(Delta)
    BA = R * I0 * np.exp(-(z - 1) / math.log(N) * T)
    return Bv2, BA


def sing(N): return float(np.real(RT(N, 0.0)[0]))


def b2_full(N):
    I, _ = integrate.quad(lambda t: 1 / (math.log(t) * math.log(N - t)), 2, N / 2, limit=400)
    return sing(N) * 2 * I


def load(commit, path):
    return json.loads(subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=HERE))

q4b = load("gb/Q1-fugacity-sectors", "research/Q1_fugacity/out/q4b_summary.json")
q5d = load("gb/Q5-background-contract", "research/Q5_background/out/q5d_summary.json")
samples = []
for sc in ["1000000", "10000000", "100000000"]:
    for r in q4b[sc]["rows"]:
        samples.append(dict(src="Q4b", X=int(sc), N=r["N"], h=r["h"]))
for r in q5d["rows"]:
    samples.append(dict(src="Q5d", X=r["scale"], N=r["N"], h=r["h"]))
print(len(samples), "polynomials", flush=True)

TH = 2 * np.pi * np.arange(16) / 16
ZC = 0.5 * np.exp(1j * TH)
rows = []
for k, sm in enumerate(samples):
    N = sm["N"]; h = np.array(sm["h"], float)
    eps = []; sens = []; D = []
    for z in ZC:
        z = complex(np.round(z.real, 12), np.round(z.imag, 12))
        Bv2, BA = background(N, z)
        Hz = np.polynomial.polynomial.polyval(z, h)
        E = Hz - Bv2
        eps.append(E / Bv2); sens.append(abs(Bv2 / BA - 1)); D.append(E / z**2)
    rows.append(dict(src=sm["src"], X=sm["X"], N=N, mod3=N % 3, nu2=(N & -N).bit_length() - 1,
                     drift=float(h[2] / b2_full(N) - 1),
                     eps=[[float(e.real), float(e.imag)] for e in eps], sens=[float(x) for x in sens],
                     D_over_N=[[float((d / N).real), float((d / N).imag)] for d in D]))
    if k % 50 == 0:
        print(k, f"{time.time()-t0:.0f}s", flush=True)

summ = {}
for src, X in sorted({(r["src"], r["X"]) for r in rows}):
    for gname, g in (("3|N", lambda r: r["mod3"] == 0), ("3!N", lambda r: r["mod3"] != 0), ("all", lambda r: True)):
        ss = [r for r in rows if r["src"] == src and r["X"] == X and g(r)]
        if len(ss) < 5: continue
        E = np.array([[complex(*e) for e in r["eps"]] for r in ss])     # n x 16
        mu = E.mean(0); sem = E.std(0, ddof=1) / math.sqrt(len(ss))
        Xc = E - mu
        C = (Xc.T @ Xc.conj()) / (len(ss) - 1)
        w, V = np.linalg.eigh(C)
        top = V[:, -1]; top = top * np.exp(-1j * np.angle(top[0]))
        dr = np.array([r["drift"] for r in ss])
        summ[f"{src} {X} {gname}"] = dict(
            n=len(ss), drift_mean=float(dr.mean()), drift_sem=float(dr.std(ddof=1) / math.sqrt(len(ss))),
            sens_median=float(np.median([r["sens"] for r in ss])),
            mu_re=[float(x) for x in mu.real], mu_im=[float(x) for x in mu.imag],
            sem_re=[float(x) for x in sem.real], sem_im=[float(x) for x in sem.imag],
            rms_centered=[float(x) for x in np.sqrt(np.real(np.diag(C)))],
            top_eig_fraction=float(w[-1] / w.sum()), second_eig_fraction=float(w[-2] / w.sum()),
            top_eig_abs=[float(x) for x in np.abs(top)], top_eig_phase=[float(x) for x in np.angle(top)])
        s_ = summ[f"{src} {X} {gname}"]
        print(f"{src} {X} {gname} n={len(ss)} drift={s_['drift_mean']:.5f}+-{s_['drift_sem']:.5f} sens={s_['sens_median']:.2e} "
              f"mu_re(th=0,pi/2,pi)=({mu.real[0]:.4f},{mu.real[4]:.4f},{mu.real[8]:.4f}) mu_im(pi/2)={mu.imag[4]:.4f} "
              f"rms(0,pi/2,pi)=({s_['rms_centered'][0]:.4f},{s_['rms_centered'][4]:.4f},{s_['rms_centered'][8]:.4f}) top={s_['top_eig_fraction']:.2f}", flush=True)
json.dump(dict(summary=summ, rows=rows, theta=list(TH)), open(os.path.join(OUT, "q6_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
