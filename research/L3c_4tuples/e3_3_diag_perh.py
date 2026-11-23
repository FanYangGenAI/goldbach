"""Diagnostic (exploratory, not preregistered): exact 4-tuple counts
C_h(N) = sum_a g(a) g(a+h), g(a) = theta(a) theta(N-a), versus the HL prediction S4(h,N)(N-|h|).
Locates the discrepancy behind the factor ~2 of E3.3."""
import json, math, os, sys
import numpy as np
import numba as nb

HERE = os.path.dirname(os.path.abspath(__file__))
C2 = 0.6601618158468695739278121100145


@nb.njit
def spf_sieve(n):
    spf = np.zeros(n + 1, dtype=np.int32)
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            j = np.int64(i) * i
            while j <= n:
                if spf[j] == 0:
                    spf[j] = i
                j += i
    return spf


@nb.njit
def nu(p, h, N):
    r1 = (-h) % p; r2 = (N - h) % p; r3 = N % p
    c = 1
    if r1 != 0: c += 1
    if r2 != 0 and r2 != r1: c += 1
    if r3 != 0 and r3 != r1 and r3 != r2: c += 1
    return c


@nb.njit
def add_primes(m, spf, buf, cnt):
    while m > 1:
        p = spf[m]
        while m % p == 0:
            m //= p
        if p >= 5:
            seen = False
            for t in range(cnt):
                if buf[t] == p:
                    seen = True; break
            if not seen:
                buf[cnt] = p; cnt += 1
    return cnt


@nb.njit
def s4_array(N, spf, C5):
    out = np.zeros(N)
    buf = np.zeros(64, dtype=np.int64); Np = np.zeros(32, dtype=np.int64)
    nN = add_primes(N, spf, Np, 0)
    for h in range(2, N, 2):
        gg = 8.0 * (1 - nu(3, h, N) / 3.0) / ((2.0 / 3.0) ** 4)
        if gg == 0.0:
            continue
        cnt = 0
        for t in range(nN):
            buf[cnt] = Np[t]; cnt += 1
        cnt = add_primes(h, spf, buf, cnt); cnt = add_primes(N - h, spf, buf, cnt); cnt = add_primes(N + h, spf, buf, cnt)
        s = C5 * gg
        for t in range(cnt):
            p = buf[t]
            s *= (1 - nu(p, h, N) / p) / (1 - 4.0 / p)
        out[h] = s
    return out


def sing(N):
    v = 2 * C2; n = N
    while n % 2 == 0: n //= 2
    p = 3
    while p * p <= n:
        if n % p == 0:
            v *= (p - 1) / (p - 2)
            while n % p == 0: n //= p
        p += 2
    if n > 1: v *= (n - 1) / (n - 2)
    return v


LOG2 = int(sys.argv[1]) if len(sys.argv) > 1 else 20
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
x = 1 << LOG2
spf = spf_sieve(4 * x + 10)
isp = (spf == np.arange(len(spf), dtype=np.int32)); isp[:2] = False
theta = np.where(isp, np.log(np.maximum(np.arange(len(spf)), 1)), 0.0)
ps = np.nonzero(isp[: 4 * x])[0]; ps = ps[ps >= 5].astype(float)
C5 = float(np.exp(np.sum(np.log1p(-4 / ps) - 4 * np.log1p(-1 / ps))))
rng = np.random.default_rng(5)
res = []
for N in (2 * rng.integers(x // 2, x, size=NS)):
    N = int(N); S = sing(N); L = math.log(N)
    gvec = theta[: N + 1] * theta[N::-1]
    Lf = 1 << int(np.ceil(np.log2(2 * (N + 1))))
    Gf = np.fft.rfft(gvec, Lf)
    C = np.fft.irfft(Gf * np.conj(Gf), Lf)[: N + 1]  # C[h], h >= 0
    R = gvec.sum()
    F = R - N * S
    s4 = s4_array(N, spf, C5)
    h = np.arange(1, N)
    w = (N - h).astype(float)
    pred = s4[1:N] * w
    act = C[1:N]
    # exact identity: R^2 = C[0] + 2 sum_{h>=1} C[h]
    diag_act = 2 * C[0]  # C[0] = sum g^2 ; antidiagonal lives inside h>0 sum: remove it
    anti = np.zeros(N - 1)
    a = (N - h) / 2
    m = (N - h) % 2 == 0
    ai = ((N - h[m]) // 2).astype(int)
    anti[m] = gvec[ai] * gvec[ai + h[m]]
    act_off = act - anti
    D_HL = 2 * N * S * (L * L - 2 * L + 2 - math.pi**2 / 6)
    err = act_off - pred
    cum = np.cumsum(err[::-1])[::-1]  # tail sums from h to N
    res.append(dict(N=N, S=S, F2=F * F, D_act=float(2 * C[0]), D_HL=float(D_HL),
                    off_act_minus_main=float(2 * act_off.sum() - S * S * (N * N - N)),
                    off_HL=float(2 * (pred.sum()) - S * S * (N * N - N)),
                    HL_err_total=float(2 * err.sum()),
                    HL_err_small_h=float(2 * err[: int(N ** 0.5)].sum()),
                    HL_err_h_lt_N_over_10=float(2 * err[: N // 10].sum()),
                    twoNSF=float(2 * N * S * F)))
    print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in res[-1].items()}, flush=True)
agg = {k: float(np.mean([r[k] / r["S"] for r in res])) for k in res[0] if k not in ("N", "S")}
print("means (/S):", {k: round(v / x, 2) for k, v in agg.items()}, "(divided by x)")
json.dump(dict(rows=res, means_over_S=agg), open(os.path.join(HERE, "out", f"e3_3_diag_perh_{LOG2}.json"), "w"), indent=2)
