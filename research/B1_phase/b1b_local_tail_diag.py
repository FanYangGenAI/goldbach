"""B1b (exploratory, not preregistered): is the excess N-variation of the non-kept-band term V explained by the
singular-series tail (primes > 7 resp. > 11 dividing N)?  Recomputes V with the frozen B1 setup (same N, same masks)."""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d0 = json.load(open(os.path.join(HERE, "out", "b1_summary.json")))
Ns = np.array(d0["Ns"]); b2 = np.array(d0["b2"]); grp = np.array([N % 3 == 0 for N in Ns])
M = 1_100_000; L = 2310 * 953
s = np.ones(M + 1, bool); s[:2] = False
for i in range(2, int(M**0.5) + 1):
    if s[i]: s[i * i :: i] = False
d = np.zeros(L); d[2 : M + 1] = 1 / np.log(np.arange(2, M + 1))
q = np.zeros(L); q[: M + 1] = s; q -= d
Q = np.fft.fft(q)


def kept_mask(step):
    k = np.arange(L); mask = np.zeros(L, bool)
    for c in np.arange(0, L, step):
        mask[(c + np.arange(-16, 17)) % L] = True
    mask |= mask[(-k) % L]; mask[0] = True; mask[L // 2] = True
    return mask


def center(x):
    x = np.asarray(x, float).copy()
    for g_ in (True, False):
        x[grp == g_] -= x[grp == g_].mean()
    return x


def oddprimes(N):
    ps = []; m = N
    while m % 2 == 0: m //= 2
    p = 3
    while p * p <= m:
        if m % p == 0:
            ps.append(p)
            while m % p == 0: m //= p
        p += 2
    if m > 1: ps.append(m)
    return ps


def S_upto(N, pmax):
    v = 1.0
    for p in oddprimes(N):
        if p <= pmax: v *= (p - 1) / (p - 2)
    return v


def S_full_ratio(N):
    v = 1.0; m = N
    while m % 2 == 0: m //= 2
    p = 3
    while p * p <= m:
        if m % p == 0:
            v *= (p - 1) / (p - 2)
            while m % p == 0: m //= p
        p += 2
    if m > 1: v *= (m - 1) / (m - 2)
    return v


out = {}
for label, step, pmax in (("mod210", L // 210, 7), ("mod2310", L // 2310, 11)):
    mask = kept_mask(step)
    V = np.real(np.fft.ifft((Q * (~mask)) ** 2))[Ns]
    tail = np.array([b2[i] * (1 - S_upto(int(N), pmax) / S_full_ratio(int(N))) for i, N in enumerate(Ns)])  # part of b2 from primes > pmax dividing N
    y = center(V / Ns); x = center(tail / Ns)
    rho = float(np.corrcoef(y, x)[0, 1]); slope = float(np.dot(x, y) / np.dot(x, x))
    out[label] = dict(corr_V_tail=rho, ls_slope=slope, sd_V=float(y.std(ddof=1)), sd_tail=float(x.std(ddof=1)),
                      sd_V_minus_tail=float((y - x).std(ddof=1)), frac_N_with_tail=float(np.mean(np.abs(tail) > 0)))
    print(label, json.dumps({k: round(v, 5) for k, v in out[label].items()}))
json.dump(out, open(os.path.join(HERE, "out", "b1b_summary.json"), "w"), indent=1, allow_nan=False)
