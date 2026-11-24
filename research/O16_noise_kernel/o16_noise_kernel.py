r"""O16 (GPU, preregistered): covariance kernel of the integer Goldbach residual z_N across N.  Autocorrelation
rho(L) = E[zc_N zc_{N+2L}] for L = 1..400, compared with the twin singular-series ratio s(2L) = prod_{p | L, p > 2} (p-1)/(p-2).
Run:  .venv\Scripts\python.exe research/O16_noise_kernel/o16_noise_kernel.py [--smoke]"""
import json, math, os, sys, time
import numpy as np
import torch
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv
C2 = 0.6601618158468695739278121100145
dev = torch.device("cuda")
H = 1 << (16 if SMOKE else 22)
XS = [2**22, 2**23] if SMOKE else [10**9, 10**10, 3 * 10**10]
TOP = max(XS) + 4 * H + 2
CH = 1 << 25
t0 = time.time()

# odd-only sieve: odd[i] = 1 iff 2i+1 is prime (the prime 2 never occurs in r(N) for even N > 4)
NO = TOP // 2 + 1
odd = torch.ones(NO, dtype=torch.uint8, device=dev)
odd[0] = 0
r_ = int(math.isqrt(TOP)) + 1
small = np.ones(r_ + 1, dtype=bool); small[:2] = False
for i in range(2, int(r_**0.5) + 1):
    if small[i]: small[i * i :: i] = False
base = np.nonzero(small)[0]
for p in base[1:]:
    p = int(p); odd[(p * p - 1) // 2 :: p] = 0
torch.cuda.synchronize()
print(f"GPU odd sieve to {TOP:,}: {time.time()-t0:.1f}s, {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)


def ind(lo, n):
    """float64 indicator of odd primes on [lo, lo+n)."""
    out = torch.zeros(n, dtype=torch.float64, device=dev)
    a = max(lo, 0); b = min(lo + n, TOP)
    if b <= a: return out
    s = a if a % 2 == 1 else a + 1
    if s >= b: return out
    i0 = (s - 1) // 2; cnt = (b - s + 1) // 2
    out[s - lo : b - lo : 2] = odd[i0 : i0 + cnt].to(torch.float64)
    return out


def window_r(x):
    L = 4 * H
    acc = torch.zeros(H, dtype=torch.float64, device=dev)
    for k in range((x + H) // H + 2):
        if k * H > x + H: break
        a = ind(k * H, H); b = ind(x - (k + 1) * H + 1, 2 * H)
        acc += torch.fft.irfft(torch.fft.rfft(a, L) * torch.fft.rfft(b, L), L)[H - 1 : 2 * H - 1]
    return torch.round(acc).cpu().numpy()


def singular_series(x):
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
    """sum_{3<=p<=M-2, p odd prime} 1/log(M-p)  (p = 2 contributes 1/log(M-2); added explicitly)."""
    s = torch.zeros((), dtype=torch.float64, device=dev)
    nmax = (M - 3) // 2                      # odd numbers 2i+1 <= M-2
    for a in range(1, nmax + 1, CH):
        b = min(a + CH, nmax + 1)
        i = torch.arange(a, b, dtype=torch.float64, device=dev)
        s += (odd[a:b].to(torch.float64) / torch.log(M - (2 * i + 1))).sum()
    return float(s) + 1 / math.log(M - 2)


def rel_pred(M):
    I = Iint(M, 1e-12)
    return 2 * (prime_sum(M) - I) / I





LMAX = 400
SEEN = {1, 2, 3, 4, 5, 6, 8, 10, 15, 21, 30, 50, 105, 210}


def s_twin(h):
    v = 1.0; m = h
    while m % 2 == 0: m //= 2
    p = 3
    while p * p <= m:
        if m % p == 0:
            v *= (p - 1) / (p - 2)
            while m % p == 0: m //= p
        p += 2
    if m > 1: v *= (m - 1) / (m - 2)
    return v


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


rows = []
for x in XS:
    x -= x % 2
    tw = time.time()
    r = window_r(x)
    N, S = singular_series(x)
    ev = N % 2 == 0
    N, S, r = N[ev], S[ev], r[ev]
    grid = np.linspace(N[0], N[-1], 41)
    b2 = S * np.interp(N, grid, np.array([Iint(float(g)) for g in grid]))
    rp = float(np.mean([rel_pred(M) for M in (int(N[0]), int(N[len(N) // 2]), int(N[-1]))]))
    z = (r - b2 * (1 + rp)) / np.sqrt(b2)
    zc = (z - z.mean()) / z.std()
    n = len(zc)
    f = np.fft.rfft(zc, 2 * n); acf = np.fft.irfft(np.abs(f) ** 2, 2 * n)[: LMAX + 1]
    acf = acf / np.arange(n, n - LMAX - 1, -1)          # unbiased lag means
    Ls = np.arange(1, LMAX + 1)
    rho = acf[1:]
    s = np.array([s_twin(2 * L) for L in Ls])
    fresh = np.array([L not in SEEN for L in Ls])
    A = np.column_stack([np.ones(fresh.sum()), s[fresh]])
    beta, *_ = np.linalg.lstsq(A, rho[fresh], rcond=None)
    res = rho[fresh] - A @ beta
    row = dict(x=int(x), n=int(n), rho=[float(v) for v in rho], s=[float(v) for v in s],
               spearman_fresh=spearman(rho[fresh], s[fresh]), pearson_fresh=float(np.corrcoef(rho[fresh], s[fresh])[0, 1]),
               fit_intercept=float(beta[0]), fit_slope=float(beta[1]), R2_fit=float(1 - res.var() / rho[fresh].var()),
               noise_se=float(1 / math.sqrt(n)), secs=time.time() - tw)
    rows.append(row)
    print(json.dumps({k: row[k] for k in ("x", "spearman_fresh", "pearson_fresh", "fit_intercept", "fit_slope", "R2_fit", "noise_se", "secs")}), flush=True)
    json.dump(dict(rows=rows, H=H, smoke=SMOKE), open(os.path.join(OUT, "o16_smoke.json" if SMOKE else "o16_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
