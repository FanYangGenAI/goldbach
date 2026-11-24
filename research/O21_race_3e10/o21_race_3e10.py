r"""O21 (GPU, preregistered; O18 extended to 1.2e10-3.6e10): Chebyshev races in Goldbach counts for moduli q = 4 (null), 5, 7, 12.
For N = c (mod q) only primes p with p and c - p units mod q pair up; A_c = {a unit : c - a unit}.
Predictor rel_c = 2 [S_{A_c}(M) - (|A_c|/phi(q)) I] / ((|A_c|/phi(q)) I), S_A(M) = sum_{p mod q in A, p > 7} 1/log(M - p).
Test: class-centred observed vs predicted window means.  Run: .venv\Scripts\python.exe research/O21_race_3e10/o21_race_3e10.py [--smoke]"""
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
LO, HI, NWIN = (22, 25, 6) if SMOKE else (33.5, 35.05, 12)
XS = []
for i in range(NWIN):
    x = int(round(2 ** (LO + (HI - LO) * i / NWIN))); x -= x % 840
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





MODS = [4, 5, 7, 12]
LCM = 420


def residue_sums(M):
    """S[r] = sum over primes p = r (mod 420), 11 <= p <= M - 2, of 1/log(M - p)."""
    acc = torch.zeros(LCM, dtype=torch.float64, device=dev)
    nmax = (M - 3) // 2
    for lo in range(5, nmax + 1, CH):          # odd index 5 <-> 11
        hi = min(lo + CH, nmax + 1)
        i = torch.arange(lo, hi, dtype=torch.int64, device=dev)
        v = 2 * i + 1
        w = odd[lo:hi].to(torch.float64) / torch.log((M - v).to(torch.float64))
        acc.index_add_(0, torch.remainder(v, LCM), w)
    return acc.cpu().numpy()


def units(q): return [a for a in range(q) if math.gcd(a, q) == 1]


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
    Ms = [int(N[0]), int(N[len(N) // 2]), int(N[-1])]
    RS = [residue_sums(M) for M in Ms]; Is = [Iint(M, 1e-12) for M in Ms]
    out = dict(i=wi, x=int(x), sqrt_x=math.sqrt(x), mods={})
    for q in MODS:
        U = units(q); phi = len(U); recs = []
        for c in sorted(set(int(v) for v in np.unique(N % q))):
            A = [a for a in U if math.gcd((c - a) % q, q) == 1]
            if not A: continue
            m = N % q == c
            preds = []
            for Rs, I in zip(RS, Is):
                SA = sum(Rs[rr] for rr in range(LCM) if (rr % q) in A)
                f = len(A) / phi
                preds.append(2 * (SA - f * I) / (f * I))
            recs.append(dict(c=c, A=A, obs=float(rel[m].mean()), pred=float(np.mean(preds)), n=int(m.sum())))
        out["mods"][str(q)] = recs
    out["secs"] = time.time() - tw
    rows.append(out)
    print(json.dumps({"i": wi, "x": x, "secs": round(out["secs"], 1),
                      **{q: [(d["c"], round(d["obs"] * out["sqrt_x"], 2), round(d["pred"] * out["sqrt_x"], 2)) for d in out["mods"][q]] for q in out["mods"]}}), flush=True)
    json.dump(dict(rows=rows, H=H, smoke=SMOKE), open(os.path.join(OUT, "o21_smoke.json" if SMOKE else "o21_summary.json"), "w"), indent=1, allow_nan=False)

res = {}
for q in MODS:
    o, p = [], []
    for row in rows:
        recs = row["mods"][str(q)]
        mo = np.mean([d["obs"] for d in recs]); mp = np.mean([d["pred"] for d in recs])
        for d in recs:
            o.append((d["obs"] - mo) * row["sqrt_x"]); p.append((d["pred"] - mp) * row["sqrt_x"])
    o, p = np.array(o), np.array(p)
    rr = dict(n=len(o), sd_obs=float(o.std()), sd_pred=float(p.std()), mean_abs_obs=float(np.abs(o).mean()), mean_abs_resid=float(np.abs(o - p).mean()))
    if p.std() > 1e-9:
        rr["corr"] = float(np.corrcoef(o, p)[0, 1]); rr["slope"] = float(np.polyfit(p, o, 1)[0])
    res[str(q)] = rr
for q in ("5", "7", "12"):
    res[q]["G1"] = bool(res[q].get("corr", 0) > 0.8 and 0.7 <= res[q].get("slope", 0) <= 1.3)
res["4"]["G0"] = bool(res["4"]["mean_abs_obs"] < 0.3)
print(json.dumps(res), flush=True)
json.dump(dict(rows=rows, H=H, smoke=SMOKE, checks=res), open(os.path.join(OUT, "o21_smoke.json" if SMOKE else "o21_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
