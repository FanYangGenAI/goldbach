"""O13 (GPU, preregistered): is the Goldbach residual local in the integers?  For every even N in a window,
z_N = (r(N) - b2(N)(1 + rel_pred)) / sqrt(b2(N)) (HL background + O5 first-order correction), regressed on local
features of N.  Run:  .venv\Scripts\python.exe research/O13_noise_locality/o13_noise_locality.py [--smoke]"""
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
XS = [2**22, 2**23] if SMOKE else [10**9, 10**10]
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



def features(N):
    """omega_small (distinct odd p <= N^(1/3)), omega_big (distinct p > N^(1/3)), log of largest prime factor / log N,
    and v2 (2-adic valuation), for an array of even N (trial division by primes up to N^(1/2) via a segmented approach)."""
    N = N.astype(np.int64); cof = N.copy(); n = len(N)
    z = np.cbrt(N.astype(np.float64))
    v2 = np.zeros(n); osm = np.zeros(n); obg = np.zeros(n); lpf = np.ones(n)
    while True:
        m = cof % 2 == 0
        if not m.any(): break
        cof[m] //= 2; v2[m] += 1
    lim = int(math.isqrt(int(N.max()))) + 1
    for p in base[1:]:
        p = int(p)
        if p > lim: break
        m = cof % p == 0
        if not m.any(): continue
        osm[m & (p <= z)] += 1; obg[m & (p > z)] += 1; lpf[m] = np.maximum(lpf[m], p)
        while True:
            m = cof % p == 0
            if not m.any(): break
            cof[m] //= p
    big = cof > 1
    obg[big & (cof > z)] += 1; osm[big & (cof <= z)] += 1; lpf[big] = np.maximum(lpf[big], cof[big])
    return np.column_stack([osm, obg, np.log(lpf) / np.log(N), v2])


def r2(y, X):
    X = np.column_stack([np.ones(len(y))] + [X[:, j] for j in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - (y - X @ beta).var() / y.var())


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
    zz = (r - b2 * (1 + rp)) / np.sqrt(b2)
    F = features(N)
    logS = np.log(S)
    Xf = np.column_stack([F, logS])
    m3 = N % 3 == 0
    # variance heterogeneity: c2 by classes
    cls = {}
    for name, m in (("3|N", m3), ("3!N", ~m3), ("omega_big=0", F[:, 1] == 0), ("omega_big=1", F[:, 1] == 1), ("omega_small=0", F[:, 0] == 0),
                    ("omega_small>=2", F[:, 0] >= 2)):
        if m.sum() > 100: cls[name] = dict(n=int(m.sum()), mean_z=float(zz[m].mean()), var_z=float(zz[m].var()))
    row = dict(x=int(x), n=int(len(N)), mean_z=float(zz.mean()), c2=float(zz.var()), R2_mean_features=r2(zz, Xf),
               R2_each=[r2(zz, Xf[:, [j]]) for j in range(Xf.shape[1])], R2_var_features=r2(zz**2, Xf), classes=cls, secs=time.time() - tw)
    rows.append(row)
    print(json.dumps(row), flush=True)
    json.dump(dict(rows=rows, H=H, smoke=SMOKE), open(os.path.join(OUT, "o13_smoke.json" if SMOKE else "o13_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
