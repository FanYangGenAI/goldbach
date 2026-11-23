"""E3.2: exact theta-weighted Goldbach counts R(N) on windows N in [x, x+H) up to x = 2^32,
computed on GPU by blocked FFT convolution. See DESIGN.md.
Run with the CUDA venv:  .venv\\Scripts\\python.exe research/L3b_gpu/e3_2_gpu_windows.py [jmin] [jmax]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
JMIN = int(sys.argv[1]) if len(sys.argv) > 1 else 22
JMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 32
H = 1 << 22
C2 = 0.6601618158468695739278121100145
dev = torch.device("cuda")
t0 = time.time()

TOP = (1 << JMAX) + 3 * H + 2
# ---- GPU sieve of [0, TOP) ----
isp = torch.ones(TOP, dtype=torch.uint8, device=dev)
isp[0] = 0; isp[1] = 0; isp[4::2] = 0
r = int(math.isqrt(TOP)) + 1
small = np.ones(r + 1, dtype=bool); small[:2] = False
for i in range(2, int(r**0.5) + 1):
    if small[i]:
        small[i * i :: i] = False
base = np.nonzero(small)[0]
for p in base[1:]:
    p = int(p)
    isp[p * p :: 2 * p] = 0  # odd multiples suffice (evens already cleared)
torch.cuda.synchronize()
print(f"GPU sieve to {TOP:,}: {time.time()-t0:.1f}s, pi={int(isp.sum().item()):,}", flush=True)


def theta(lo, n):
    """theta weights for integers lo .. lo+n-1 (zeros for negative / non-prime)."""
    out = torch.zeros(n, dtype=torch.float64, device=dev)
    a = max(lo, 0); b = min(lo + n, TOP)
    if b <= a:
        return out
    idx = torch.arange(a, b, dtype=torch.float64, device=dev)
    out[a - lo : b - lo] = torch.log(idx.clamp(min=1.0)) * isp[a:b].to(torch.float64)
    return out


def window_R(x):
    """R(N) for N = x .. x+H-1 (all N; caller keeps even ones)."""
    L = 4 * H
    acc = torch.zeros(H, dtype=torch.float64, device=dev)
    K = (x + H) // H + 1
    for k in range(K):
        a = theta(k * H, H)
        if k * H > x + H:
            break
        s = x - (k + 1) * H + 1
        b = theta(s, 2 * H)
        c = torch.fft.irfft(torch.fft.rfft(a, L) * torch.fft.rfft(b, L), L)
        acc += c[H - 1 : 2 * H - 1]
    return acc.cpu().numpy()


def singular_series(x):
    N = np.arange(x, x + H, dtype=np.int64)
    S = np.full(H, 2 * C2)
    cof = N.copy()
    while True:  # remove factors 2
        m = cof % 2 == 0
        if not m.any():
            break
        cof[m] //= 2
    for p in base[1:]:
        p = int(p)
        st = (-x) % p
        sl = slice(st, H, p)
        S[sl] *= (p - 1) / (p - 2)
        sub = cof[sl]
        while True:
            m = sub % p == 0
            if not m.any():
                break
            sub[m] //= p
        cof[sl] = sub
    big = cof > 1  # remaining cofactor is a prime > sqrt(TOP)
    S[big] *= (cof[big] - 1) / (cof[big] - 2)
    return N, S


rows = []
for j in range(JMIN, JMAX + 1):
    x = 1 << j
    tj = time.time()
    R = window_R(x)
    N, S = singular_series(x)
    ev = (N % 2 == 0)
    N, S, R = N[ev], S[ev], R[ev]
    F = R - N * S
    Z = F / np.sqrt(S)
    zc = Z - Z.mean(); m2 = (zc**2).mean()
    zs = zc / np.sqrt(m2)
    rel = F / (N * S)
    row = dict(j=j, x=float(x), H=H, n=len(N), mean_F_over_NS=float(rel.mean()), sigma2=float(m2),
               m4_ratio=float((zc**4).mean() / m2**2), m6_ratio=float((zc**6).mean() / m2**3),
               maxz_over_sqrt2lnn=float(np.abs(zs).max() / np.sqrt(2 * np.log(len(N)))),
               K_sigmas=float((N * S / (np.sqrt(m2) * np.sqrt(S))).min()),
               B4_window=float((rel**4).sum()), B2_window=float((rel**2).sum()),
               B4_dyadic_est=float((rel**4).sum() * x / H), B2_dyadic_est=float((rel**2).sum() * x / H),
               max_abs_rel=float(np.abs(rel).max()), min_R=float(R.min()), secs=time.time() - tj)
    rows.append(row)
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    np.save(os.path.join(OUT, f"e3_2_rel_j{j}.npy"), rel.astype(np.float32))
    json.dump(rows, open(os.path.join(OUT, f"e3_2_rows_{JMIN}_{JMAX}.json"), "w"), indent=2)
print(f"done {time.time()-t0:.1f}s")
