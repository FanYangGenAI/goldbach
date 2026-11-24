"""O9 (GPU, preregistered): push the O5 drift predictor and the O8 variance constant to x ~ 1.7e10 and 3.4e10.
Odd-only uint8 sieve on the RTX 4090.  For each window: rel = r/b2 - 1 (as O2), rel_pred (as O5), and
c2 = Var[(r - b2)/sqrt(b2)] (as O8).  See DESIGN.md.  Run:  .venv\\Scripts\\python.exe research/O9_gpu_2e10/o9_gpu_2e10.py [--smoke]"""
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
JS = [22, 23] if SMOKE else [34, 35]
NW = 3 if SMOKE else 8
XS = [(j, (1 << j) + w * (1 << (j - 4))) for j in JS for w in range(NW)]
TOP = max(x for _, x in XS) + 4 * H + 2
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


if SMOKE:
    # cross-check the odd-only sieve against a CPU sieve
    lim = 2 * 10**6
    cpu = np.ones(lim + 1, dtype=bool); cpu[:2] = False
    for i in range(2, int(lim**0.5) + 1):
        if cpu[i]: cpu[i * i :: i] = False
    g = ind(0, lim + 1).cpu().numpy().astype(bool); cpu[2] = False
    assert (g == cpu).all(); print("odd sieve check ok", flush=True)

rows = []
for j, x in XS:
    x -= x % 2
    tw = time.time()
    r = window_r(x)
    if SMOKE and len(rows) == 0:
        ip = ind(0, x + H + 2).cpu().numpy().astype(bool)
        for i0 in (0, 2, 1000):
            Nn = x + i0
            brute = int(np.sum(ip[2 : Nn - 1] & ip[Nn - 2 : 1 : -1]))
            assert brute == r[i0], (Nn, brute, r[i0])
        print("brute check ok", flush=True)
    N, S = singular_series(x)
    ev = N % 2 == 0
    N, S, r = N[ev], S[ev], r[ev]
    grid = np.linspace(N[0], N[-1], 41)
    b2 = S * np.interp(N, grid, np.array([Iint(float(g)) for g in grid]))
    rel = r / b2 - 1
    zz = (r - b2) / np.sqrt(b2)
    Ms = [int(N[0]), int(N[len(N) // 2]), int(N[-1])]
    pred = float(np.mean([rel_pred(M) for M in Ms]))
    row = dict(j=j, x=int(x), mean=float(rel.mean()), sem=float(rel.std(ddof=1) / math.sqrt(len(rel))), pred=pred,
               sqrt_x=math.sqrt(x), c2=float(zz.var()), min_r=float(r.min()), log_x=math.log(x), secs=time.time() - tw)
    rows.append(row)
    print(json.dumps(row), flush=True)
    json.dump(dict(rows=rows, H=H, smoke=SMOKE), open(os.path.join(OUT, "o9_smoke.json" if SMOKE else "o9_summary.json"), "w"), indent=1, allow_nan=False)

res = {}
for j in JS:
    rs = [r for r in rows if r["j"] == j]
    resid = [abs(r["mean"] - r["pred"]) * r["sqrt_x"] for r in rs]
    res[str(j)] = dict(mean_abs_resid=float(np.mean(resid)), mean_abs_obs=float(np.mean([abs(r["mean"]) * r["sqrt_x"] for r in rs])),
                       c2_mean=float(np.mean([r["c2"] for r in rs])), c2_pred=float(np.mean([1 - 7.2 / r["log_x"] for r in rs])))
    res[str(j)]["P1"] = bool(res[str(j)]["mean_abs_resid"] < 0.2)
    res[str(j)]["P2"] = bool(abs(res[str(j)]["c2_mean"] - res[str(j)]["c2_pred"]) < 0.03)
print(json.dumps(res), flush=True)
json.dump(dict(rows=rows, H=H, smoke=SMOKE, checks=res), open(os.path.join(OUT, "o9_smoke.json" if SMOKE else "o9_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
