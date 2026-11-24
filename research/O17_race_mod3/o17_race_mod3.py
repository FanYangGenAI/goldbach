r"""O17 (GPU, preregistered): first-order prime-counting term split by residue class mod 3 (Chebyshev race,
zeros of L(s, chi_3)).  For N = 1 mod 3 only primes p = 2 mod 3 pair with primes N - p; for N = 2 mod 3 only p = 1 mod 3.
Predictor per class c in {1, 2}: rel_c = 2 [ sum_{p = a_c (3)} 1/log(N-p) - I/2 ] / (I/2), a_1 = 2, a_2 = 1.
Test: across windows, the observed class difference d = mean rel(N=1) - mean rel(N=2) vs predicted.
Run:  .venv\Scripts\python.exe research/O17_race_mod3/o17_race_mod3.py [--smoke]"""
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
LO, HI, NWIN = (22, 25, 6) if SMOKE else (30, 34, 64)
XS = []
for i in range(NWIN):
    x = int(round(2 ** (LO + (HI - LO) * i / NWIN))); x -= x % 6
    XS.append(x)
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




def prime_sum_class(M, a):
    """sum over primes p = a (mod 3), 5 <= p <= M - 2, of 1/log(M - p)."""
    s = torch.zeros((), dtype=torch.float64, device=dev)
    nmax = (M - 3) // 2
    for lo in range(1, nmax + 1, CH):
        hi = min(lo + CH, nmax + 1)
        i = torch.arange(lo, hi, dtype=torch.float64, device=dev)
        v = 2 * i + 1
        cls = torch.remainder(v, 3)
        s += (odd[lo:hi].to(torch.float64) * (cls == a) / torch.log(M - v)).sum()
    return float(s)


rows = []
for wi, x in enumerate(XS):
    tw = time.time()
    r = window_r(x)
    N, S = singular_series(x)
    ev = N % 2 == 0
    N, S, r = N[ev], S[ev], r[ev]
    grid = np.linspace(N[0], N[-1], 41)
    b2 = S * np.interp(N, grid, np.array([Iint(float(g)) for g in grid]))
    rel = r / b2 - 1
    out = dict(i=wi, x=int(x), sqrt_x=math.sqrt(x))
    for c, a in ((1, 2), (2, 1)):
        m = N % 3 == c
        out[f"obs_{c}"] = float(rel[m].mean()); out[f"sem_{c}"] = float(rel[m].std(ddof=1) / math.sqrt(m.sum()))
        Ms = [int(N[m][0]), int(N[m][len(N[m]) // 2]), int(N[m][-1])]
        preds = []
        for M in Ms:
            I = Iint(M, 1e-12)
            preds.append(2 * (prime_sum_class(M, a) - I / 2) / (I / 2))
        out[f"pred_{c}"] = float(np.mean(preds))
    out["d_obs"] = out["obs_1"] - out["obs_2"]; out["d_pred"] = out["pred_1"] - out["pred_2"]
    out["secs"] = time.time() - tw
    rows.append(out)
    print(json.dumps(out), flush=True)
    json.dump(dict(rows=rows, H=H, smoke=SMOKE), open(os.path.join(OUT, "o17_smoke.json" if SMOKE else "o17_summary.json"), "w"), indent=1, allow_nan=False)

do = np.array([r["d_obs"] * r["sqrt_x"] for r in rows]); dp = np.array([r["d_pred"] * r["sqrt_x"] for r in rows])
res = dict(corr=float(np.corrcoef(do, dp)[0, 1]), slope=float(np.polyfit(dp, do, 1)[0]), mean_abs_obs=float(np.abs(do).mean()),
           mean_abs_resid=float(np.abs(do - dp).mean()), sd_obs=float(do.std()), sd_pred=float(dp.std()))
res["R1"] = bool(res["corr"] > 0.5 and 0.5 <= res["slope"] <= 1.5)
res["R2"] = bool(res["mean_abs_resid"] < 0.5 * res["mean_abs_obs"])
print(json.dumps(res), flush=True)
json.dump(dict(rows=rows, H=H, smoke=SMOKE, checks=res), open(os.path.join(OUT, "o17_smoke.json" if SMOKE else "o17_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
