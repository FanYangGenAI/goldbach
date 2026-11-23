"""V3 (GPU): (a) fixed-ratio windows H = x/64, j = 20..32; (b) stratified dyadic moment estimates
(8 windows of width 2^20 at x(1+i/8)), calibrated against the CPU full dyadic values of E3.1.
Run with the CUDA venv:  .venv\\Scripts\\python.exe research/R1_review/v3_gpu_windows.py
"""
import json, math, os, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
C2 = 0.6601618158468695739278121100145
dev = torch.device("cuda")
JMAX = 32
TOP = (1 << JMAX) + 4 * (1 << (JMAX - 6)) + 2  # fixed j<=32 windows; stratified j<=31 reach 2^32
t0 = time.time()
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
    isp[p * p :: 2 * p] = 0
torch.cuda.synchronize()
print(f"sieve to {TOP:,}: {time.time()-t0:.1f}s", flush=True)


def theta(lo, n):
    out = torch.zeros(n, dtype=torch.float64, device=dev)
    a = max(lo, 0); b = min(lo + n, TOP)
    if b <= a:
        return out
    idx = torch.arange(a, b, dtype=torch.float64, device=dev)
    out[a - lo : b - lo] = torch.log(idx.clamp(min=1.0)) * isp[a:b].to(torch.float64)
    return out


def window_R(x, H):
    L = 4 * H
    acc = torch.zeros(H, dtype=torch.float64, device=dev)
    for k in range((x + H) // H + 2):
        if k * H > x + H:
            break
        a = theta(k * H, H)
        b = theta(x - (k + 1) * H + 1, 2 * H)
        acc += torch.fft.irfft(torch.fft.rfft(a, L) * torch.fft.rfft(b, L), L)[H - 1 : 2 * H - 1]
    return acc.cpu().numpy()


def singular_series(x, H):
    N = np.arange(x, x + H, dtype=np.int64)
    S = np.full(H, 2 * C2)
    cof = N.copy()
    while True:
        m = cof % 2 == 0
        if not m.any():
            break
        cof[m] //= 2
    for p in base[1:]:
        p = int(p)
        if p > x + H:
            break
        sl = slice((-x) % p, H, p)
        S[sl] *= (p - 1) / (p - 2)
        sub = cof[sl]
        while True:
            m = sub % p == 0
            if not m.any():
                break
            sub[m] //= p
        cof[sl] = sub
    big = cof > 1
    S[big] *= (cof[big] - 1) / (cof[big] - 2)
    return N, S


def stats(x, H):
    R = window_R(x, H)
    N, S = singular_series(x, H)
    ev = N % 2 == 0
    N, S, R = N[ev], S[ev], R[ev]
    F = R - N * S
    Z = F / np.sqrt(S)
    zc = Z - Z.mean(); m2 = (zc**2).mean()
    rel = F / (N * S)
    return dict(x=x, H=H, n=int(len(N)), sigma2=float(m2), mean_Z=float(Z.mean()),
                m4_ratio=float((zc**4).mean() / m2**2),
                maxz_over_sqrt2lnn=float(np.abs(zc).max() / np.sqrt(m2) / np.sqrt(2 * np.log(len(N)))),
                B2_window=float((rel**2).sum()), B4_window=float((rel**4).sum()),
                K_centered=float(((N * S + F.mean()) / np.sqrt(m2 * S)).min()))


fixed = []
for j in range(20, JMAX + 1):
    x = 1 << j; H = x >> 6
    st = stats(x, H); st["j"] = j
    fixed.append(st)
    print("fixed", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in st.items()}, f"{time.time()-t0:.1f}s", flush=True)
    json.dump(dict(fixed=fixed), open(os.path.join(OUT, "v3_gpu.json"), "w"), indent=2)

strat = []
Hs = 1 << 20
for j in range(22, JMAX):
    x = 1 << j
    B2 = B4 = 0.0; parts = []
    for i in range(8):
        xi = x + i * (x >> 3)
        st = stats(xi, Hs)
        scale = (x >> 3) / Hs
        B2 += st["B2_window"] * scale; B4 += st["B4_window"] * scale
        parts.append(st)
    strat.append(dict(j=j, B2_strat=B2, B4_strat=B4, parts=parts))
    print("strat", j, B2, B4, f"{time.time()-t0:.1f}s", flush=True)
    json.dump(dict(fixed=fixed, strat=strat), open(os.path.join(OUT, "v3_gpu.json"), "w"), indent=2)
print(f"done {time.time()-t0:.1f}s")
