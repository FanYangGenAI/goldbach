"""E3.3: predict Var(R(N)) from the Hardy-Littlewood 4-tuple conjecture and compare with
the GPU-measured variance of E3.2. See DESIGN.md.  Run with base python (numba).
"""
from __future__ import annotations

import json
import math
import os
import time

import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
GPU_ROWS = os.path.join(HERE, "..", "L3b_gpu", "out", "e3_2_rows_22_32.json")
JS = [23, 24, 25, 26]
H = 1 << 22
NSAMP = 48
C2 = 0.6601618158468695739278121100145
t0 = time.time()


@nb.njit(cache=True)
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


@nb.njit(cache=True)
def nu(p, h, N):
    r0 = 0; r1 = (-h) % p; r2 = (N - h) % p; r3 = N % p
    c = 1
    if r1 != r0:
        c += 1
    if r2 != r0 and r2 != r1:
        c += 1
    if r3 != r0 and r3 != r1 and r3 != r2:
        c += 1
    return c


@nb.njit(cache=True)
def add_primes(m, spf, buf, cnt):
    while m > 1:
        p = spf[m]
        while m % p == 0:
            m //= p
        if p >= 5:
            seen = False
            for t in range(cnt):
                if buf[t] == p:
                    seen = True
                    break
            if not seen:
                buf[cnt] = p
                cnt += 1
    return cnt


@nb.njit(cache=True)
def hl_variance(N, spf, C5, S):
    """returns (diag, offdiag, mean S4 over even h / S^2)."""
    L = math.log(N)
    diag = 2.0 * N * S * (L * L - 2 * L + 2 - math.pi**2 / 6)
    S2 = S * S
    buf = np.zeros(64, dtype=np.int64)
    Nprimes = np.zeros(32, dtype=np.int64)
    nN = add_primes(N, spf, Nprimes, 0)
    off = 0.0; comp = 0.0
    s4sum = 0.0; ne = 0
    for h in range(1, N):
        w = float(N - h)
        if h % 2 == 1:
            term = -S2 * w
        else:
            # p = 2: residues all even -> nu = 1 ; factor (1 - 1/2)/(1/2)^4 = 8
            g = 8.0
            n3 = nu(3, h, N)
            g *= (1 - n3 / 3.0) / ((2.0 / 3.0) ** 4)
            if g == 0.0:
                s4 = 0.0
            else:
                cnt = 0
                for t in range(nN):
                    buf[cnt] = Nprimes[t]; cnt += 1
                cnt = add_primes(h, spf, buf, cnt)
                cnt = add_primes(N - h, spf, buf, cnt)
                cnt = add_primes(N + h, spf, buf, cnt)
                s4 = C5 * g
                for t in range(cnt):
                    p = buf[t]
                    s4 *= (1 - nu(p, h, N) / p) / (1 - 4.0 / p)
            s4sum += s4; ne += 1
            term = (s4 - S2) * w
        # Kahan summation
        y = 2.0 * term - comp
        tt = off + y
        comp = (tt - off) - y
        off = tt
    return diag, off, (s4sum / ne) / S2


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


# C5 = prod_{p>=5} (1-4/p)/(1-1/p)^4
PM = 20_000_000
ps = np.nonzero(spf_sieve(PM) == np.arange(PM + 1, dtype=np.int32))[0]
ps = ps[ps >= 5].astype(np.float64)
C5 = float(np.exp(np.sum(np.log1p(-4 / ps) - 4 * np.log1p(-1 / ps))))
# tail correction: log factor ~ -6/p^2 summed over p > PM ~ -6/(PM log PM)
C5 *= math.exp(-6 / (PM * math.log(PM)))
print(f"C5={C5:.10f} ({time.time()-t0:.1f}s)", flush=True)

maxN = (1 << max(JS)) + H
spf = spf_sieve(2 * maxN + 10)
print(f"spf to {2*maxN:,}: {time.time()-t0:.1f}s", flush=True)

gpu = {r["j"]: r for r in json.load(open(GPU_ROWS))}
rng = np.random.default_rng(33)
rows = []
for j in JS:
    x = 1 << j
    Ns = (2 * rng.integers(x // 2, (x + H) // 2, size=NSAMP)).astype(np.int64)
    vals = []
    for N in Ns:
        S = sing(int(N))
        d, o, m = hl_variance(int(N), spf, C5, S)
        vals.append((int(N), S, d, o, m))
    V = np.array([(d + o) / S for (_, S, d, o, _) in vals])
    D = np.array([d / S for (_, S, d, o, _) in vals]); O = np.array([o / S for (_, S, d, o, _) in vals])
    M = np.array([m for *_, m in vals])
    row = dict(j=j, sigma2_HL=float(V.mean()), sigma2_HL_sem=float(V.std() / np.sqrt(len(V))),
               sigma2_obs=gpu[j]["sigma2"], ratio=float(V.mean() / gpu[j]["sigma2"]),
               diag_mean=float(D.mean()), offdiag_mean=float(O.mean()), offdiag_over_diag=float(O.mean() / D.mean()),
               mean_S4_even_over_S2=float(M.mean()), samples=vals)
    rows.append(row)
    print({k: v for k, v in row.items() if k != "samples"}, f"{time.time()-t0:.1f}s", flush=True)
    json.dump(rows, open(os.path.join(OUT, "e3_3_rows.json"), "w"), indent=2, default=float)

checks = dict(P1=all(0.9 <= r["ratio"] <= 1.1 for r in rows),
              P2=all(0.6 <= -r["offdiag_over_diag"] <= 0.8 for r in rows),
              P3=all(abs(r["mean_S4_even_over_S2"] / 2 - 1) < 0.01 for r in rows))
print(checks)
json.dump(dict(rows=rows, checks=checks), open(os.path.join(OUT, "e3_3_rows.json"), "w"), indent=2, default=float)
print(f"done {time.time()-t0:.1f}s")
