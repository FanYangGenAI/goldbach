"""O8 integer part (exploratory, GPU): Goldbach variance constant c^2 = Var[(r(N) - b2(N)) / sqrt(b2(N))]
over all even N in one window per scale j = 16..33 (b2 as in O2).  Run with .venv python.  [--smoke]"""
import json, math, os, sys, time
import numpy as np
import torch
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv
C2 = 0.6601618158468695739278121100145
dev = torch.device("cuda")
JS = [16, 18, 20] if SMOKE else list(range(16, 34))
HJ = {j: 1 << min(22, j - 4) for j in JS}
TOP = (1 << max(JS)) + (1 << (max(JS) - 3)) + 4 * max(HJ.values()) + 2
CH = 1 << 27
t0 = time.time()

isp = torch.ones(TOP, dtype=torch.uint8, device=dev)
isp[0] = 0; isp[1] = 0; isp[4::2] = 0
r_ = int(math.isqrt(TOP)) + 1
small = np.ones(r_ + 1, dtype=bool); small[:2] = False
for i in range(2, int(r_**0.5) + 1):
    if small[i]: small[i * i :: i] = False
base = np.nonzero(small)[0]
for p in base[1:]:
    p = int(p); isp[p * p :: 2 * p] = 0
torch.cuda.synchronize()
print(f"GPU sieve to {TOP:,}: {time.time()-t0:.1f}s", flush=True)


def ind(lo, n):
    out = torch.zeros(n, dtype=torch.float64, device=dev)
    a = max(lo, 0); b = min(lo + n, TOP)
    if b > a:
        out[a - lo : b - lo] = isp[a:b].to(torch.float64)
    return out


def window_r(x, H):
    L = 4 * H
    acc = torch.zeros(H, dtype=torch.float64, device=dev)
    for k in range((x + H) // H + 2):
        if k * H > x + H: break
        a = ind(k * H, H); b = ind(x - (k + 1) * H + 1, 2 * H)
        acc += torch.fft.irfft(torch.fft.rfft(a, L) * torch.fft.rfft(b, L), L)[H - 1 : 2 * H - 1]
    return torch.round(acc).cpu().numpy()


def singular_series(x, H):
    N = np.arange(x, x + H, dtype=np.int64)
    S = np.full(H, 2 * C2); cof = N.copy()
    while True:
        m = cof % 2 == 0
        if not m.any(): break
        cof[m] //= 2
    for p in base[1:]:
        p = int(p)
        if p > x + H: break
        sl = slice((-x) % p, H, p)
        S[sl] *= (p - 1) / (p - 2)
        sub = cof[sl]
        while True:
            m = sub % p == 0
            if not m.any(): break
            sub[m] //= p
        cof[sl] = sub
    big = cof > 1
    S[big] *= (cof[big] - 1) / (cof[big] - 2)
    return N, S


def Iint(N, eps=1.49e-8):
    I, _ = integrate.quad(lambda t: 1 / (math.log(t) * math.log(N - t)), 2, N / 2, limit=800, epsabs=0, epsrel=eps)
    return 2 * I


def prime_sum(M):
    """sum_{2<=p<=M-2} 1/log(M-p), float64 on GPU."""
    s = torch.zeros((), dtype=torch.float64, device=dev)
    for a in range(2, M - 1, CH):
        b = min(a + CH, M - 1)
        n = torch.arange(a, b, dtype=torch.float64, device=dev)
        s += (isp[a:b].to(torch.float64) / torch.log(M - n)).sum()
    return float(s)


def rel_pred(M):
    I = Iint(M, 1e-12)
    return 2 * (prime_sum(M) - I) / I



rows = []
for j in JS:
    H = HJ[j]; x = (1 << j) + (1 << (j - 3)); x -= x % 2
    tw = time.time()
    r = window_r(x, H)
    N, S = singular_series(x, H)
    ev = N % 2 == 0
    N, S, r = N[ev], S[ev], r[ev]
    grid = np.linspace(N[0], N[-1], 41)
    b2 = S * np.interp(N, grid, np.array([Iint(float(g)) for g in grid]))
    z = (r - b2) / np.sqrt(b2)
    m3 = N % 3 == 0
    row = dict(j=j, x=int(x), H=H, n=int(len(N)), mean_r=float(r.mean()), c2=float(z.var()), mean_z=float(z.mean()),
               c2_mod3_0=float(z[m3].var()), c2_mod3_nz=float(z[~m3].var()), secs=time.time() - tw)
    rows.append(row)
    print(json.dumps(row), flush=True)
    json.dump(dict(rows=rows, smoke=SMOKE), open(os.path.join(OUT, "o8_int_smoke.json" if SMOKE else "o8_int_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
