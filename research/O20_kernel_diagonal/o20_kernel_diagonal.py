r"""O20 (GPU, exploratory): exact diagonal part of the Goldbach covariance.  r(N) r(N+h) = sum over pairs (n, n');
coincident pairs give 4 (P * Q_h)(N) with Q_h(m) = 1_P(m) 1_P(m+h) (h > 0) and 2 r(N) (h = 0).  Compare the empirical
covariance of e_N = r(N) - b2(N)(1+rel_pred) with the diagonal part, lag by lag; the remainder is the off-diagonal
(four distinct primes) contribution.  Run: .venv\Scripts\python.exe research/O20_kernel_diagonal/o20_kernel_diagonal.py [--smoke]"""
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
XS = [2**22] if SMOKE else [10**9, 10**10]
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


def window_conv(fa, fb, x):
    L = 4 * H
    acc = torch.zeros(H, dtype=torch.float64, device=dev)
    for k in range((x + H) // H + 2):
        if k * H > x + H: break
        a = fa(k * H, H); b = fb(x - (k + 1) * H + 1, 2 * H)
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



HS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 21, 30, 35, 105]          # lags in units of 2: h = 2L
rows = []
for x in XS:
    x -= x % 2
    tw = time.time()
    r = window_conv(ind, ind, x)
    N, S = singular_series(x)
    ev = N % 2 == 0
    grid = np.linspace(N[ev][0], N[ev][-1], 41)
    Ig = np.array([Iint(float(g)) for g in grid])
    b2 = S * np.interp(N, grid, Ig)
    rp = float(np.mean([rel_pred(M) for M in (int(N[ev][0]), int(N[ev][len(N[ev]) // 2]), int(N[ev][-1]))]))
    e = np.where(ev, r - b2 * (1 + rp), 0.0)
    Ne = N[ev]; ee = e[ev]; ee = ee - ee.mean(); rr = r[ev]; bb = b2[ev]
    var = float(np.mean(ee**2)); mean_r = float(rr.mean())
    rec = dict(x=int(x), var_e=var, mean_r=mean_r, mean_b2=float(bb.mean()), c2=var / float(bb.mean()),
               diag0=float(2 * mean_r), offdiag0_excess=float(var - 2 * mean_r), lags={})
    for L in HS:
        h = 2 * L
        fq = lambda lo, n, h=h: ind(lo, n) * ind(lo + h, n)
        T = window_conv(ind, fq, x)              # T(M) = sum_n P(n) Q_h(M - n)
        T = T[ev]
        m = len(ee) - L
        cov = float(np.mean(ee[:m] * ee[L:]))
        diag = float(4 * np.mean(T[:m]))
        rec["lags"][str(L)] = dict(h=h, cov=cov, diag=diag, offdiag_excess=cov - diag, cov_over_diag=cov / diag,
                                    rho=cov / var, s=float(s_twin(h)) if True else None)
        print(x, L, round(cov / var, 4), round(cov / diag, 4), flush=True)
    rec["secs"] = time.time() - tw
    rows.append(rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "lags"}), flush=True)
    json.dump(dict(rows=rows, H=H, smoke=SMOKE), open(os.path.join(OUT, "o20_smoke.json" if SMOKE else "o20_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
