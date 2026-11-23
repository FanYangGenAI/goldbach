"""Q1/Q3: sector transition matrix T_jk(N) of the reflection n -> N-n in the Omega-grading,
and zeros of H_N(z) = sum_n z^{Omega(n)+Omega(N-n)}. See DESIGN.md."""
import json, math, os, time
import numba as nb
import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
C2 = 0.6601618158468695739278121100145
KMAX = 40
t0 = time.time()


@nb.njit(cache=True)
def omega_sieve(n):
    """Omega(m) (with multiplicity) for m <= n via smallest prime factor."""
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
def sector_tables(Ns, om):
    out = np.zeros((len(Ns), KMAX, KMAX), dtype=np.int64)
    for k in nb.prange(len(Ns)):
        N = Ns[k]
        for n in range(2, N - 1):
            out[k, om[n], om[N - n]] += 1
    return out


def sing(N):
    v = 2 * C2; n = N
    while n % 2 == 0:
        n //= 2
    p = 3
    while p * p <= n:
        if n % p == 0:
            v *= (p - 1) / (p - 2)
            while n % p == 0:
                n //= p
        p += 2
    if n > 1:
        v *= (n - 1) / (n - 2)
    return v


def zeros_of(coeffs):
    """coeffs[m] = coefficient of z^m; returns roots of the polynomial (highest degree first for mpmath)."""
    mp.mp.dps = 50
    c = [int(x) for x in coeffs]
    while c and c[-1] == 0:
        c.pop()
    lo = 0
    while c[lo] == 0:
        lo += 1
    c = c[lo:]
    roots = mp.polyroots(list(reversed(c)), maxsteps=400, extraprec=400)
    return lo, [complex(r) for r in roots]


NMAX = 10**8
om = omega_sieve(int(1.1 * NMAX) + 10)
print(f"Omega sieve {time.time()-t0:.1f}s", flush=True)
rng = np.random.default_rng(11)
res = {"Q1": {}, "Q3": {}}
for N0 in [10**6, 10**7, 10**8]:
    Ns = (2 * rng.integers(N0 // 2, int(1.1 * N0) // 2, size=30)).astype(np.int64)
    T = sector_tables(Ns, om)
    ratios = []; Cs = []
    for i, N in enumerate(Ns):
        Ti = T[i].astype(float)
        row = Ti.sum(1); col = Ti.sum(0); tot = Ti.sum()
        Ind = np.outer(row, col) / tot
        with np.errstate(divide="ignore", invalid="ignore"):
            C = np.where(Ind > 0, Ti / Ind, np.nan)
        Cs.append(C[:9, :9])
        ratios.append(C[1, 1] / sing(int(N)))
    Cs = np.array(Cs)
    Smat = np.array([sing(int(N)) for N in Ns])
    # dependence on S(N): correlation of log C_jk with log S(N)
    dep = np.full((9, 9), np.nan)
    for j in range(1, 9):
        for k in range(1, 9):
            v = Cs[:, j, k]
            if np.all(np.isfinite(v)) and np.all(v > 0):
                dep[j, k] = np.polyfit(np.log(Smat), np.log(v), 1)[0]
    res["Q1"][N0] = dict(C11_over_S_median=float(np.median(ratios)), C11_over_S_min=float(np.min(ratios)), C11_over_S_max=float(np.max(ratios)),
                         C_mean=np.nanmean(Cs, 0)[1:9, 1:9].round(4).tolist(),
                         slope_logC_vs_logS=np.round(dep[1:9, 1:9], 3).tolist())
    print(N0, "C11/S median", res["Q1"][N0]["C11_over_S_median"], f"{time.time()-t0:.1f}s", flush=True)
    # Q3: zeros of H_N for 5 of the N
    zs = []
    for i in range(5):
        N = int(Ns[i]); Ti = T[i]
        h = np.zeros(2 * KMAX, dtype=np.int64)
        for j in range(KMAX):
            for k in range(KMAX):
                h[j + k] += Ti[j, k]
        lo, roots = zeros_of(h)
        roots = np.array(roots)
        real_neg = np.sum((np.abs(roots.imag) < 1e-12 * np.maximum(1, np.abs(roots))) & (roots.real < 0))
        zs.append(dict(N=N, order_at_0=int(lo), degree=len(roots), n_real_negative=int(real_neg),
                       max_abs_imag=float(np.abs(roots.imag).max()), min_modulus=float(np.abs(roots).min()),
                       r_over_h3=float(h[2] / h[3]), roots=[[float(r.real), float(r.imag)] for r in roots]))
        print("  zeros N=%d: order0=%d deg=%d realneg=%d maxImag=%.3g minMod=%.4f r/h3=%.4f" % (N, lo, len(roots), real_neg, zs[-1]["max_abs_imag"], zs[-1]["min_modulus"], zs[-1]["r_over_h3"]), flush=True)
    res["Q3"][N0] = zs
res["P1"] = all(0.9 <= res["Q1"][N0]["C11_over_S_median"] <= 1.1 for N0 in res["Q1"])
# single-system control: F_x(z) = sum_{n<=x} z^Omega(n)
ctrl = {}
for x in [10**6, 10**8]:
    cnt = np.bincount(om[2 : x + 1].astype(np.int64), minlength=KMAX)
    lo, roots = zeros_of(cnt); roots = np.array(roots)
    ctrl[x] = dict(order_at_0=lo, degree=len(roots), max_abs_imag=float(np.abs(roots.imag).max()),
                   n_real_negative=int(np.sum((np.abs(roots.imag) < 1e-12 * np.maximum(1, np.abs(roots))) & (roots.real < 0))),
                   roots=[[float(r.real), float(r.imag)] for r in roots])
    print("control F_x x=%d: deg=%d realneg=%d maxImag=%.3g" % (x, len(roots), ctrl[x]["n_real_negative"], ctrl[x]["max_abs_imag"]))
res["control_single_system"] = ctrl
json.dump(res, open(os.path.join(OUT, "q1q3_summary.json"), "w"), indent=1, default=float)
print("P1:", res["P1"], f"done {time.time()-t0:.1f}s")
