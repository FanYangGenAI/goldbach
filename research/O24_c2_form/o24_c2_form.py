r"""O24 (GPU, preregistered): the Goldbach variance constant c2 = Var[(r - b2)/sqrt(b2)] on 4 windows per scale,
j = 14..34, to discriminate the functional forms fitted in O8 (parameters frozen from O8).
Run: .venv\Scripts\python.exe research/O24_c2_form/o24_c2_form.py [--smoke]"""
import json, math, os, sys, time
import numpy as np
import torch
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv
C2 = 0.6601618158468695739278121100145
dev = torch.device("cuda")
JS = [14, 16] if SMOKE else list(range(14, 35))
NW = 4
HJ = {j: 1 << min(22, j - 3) for j in JS}
XS = [(j, (1 << j) + w * (1 << (j - 3))) for j in JS for w in range(NW)]
TOP = max(x for _, x in XS) + 4 * max(HJ.values()) + 2
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



FORMS = {"a*L^b": lambda L: 0.1354 * L**0.5147, "A(1-C/sqrtL)": lambda L: 1.2484 * (1 - 2.199 / math.sqrt(L)),
         "A-C/L": lambda L: 0.9276 - 5.8465 / L, "1-C/L": lambda L: 1 - 7.2562 / L, "A-C/L^2": lambda L: 0.7608 - 49.8224 / L**2}
rows = []
for j, x in XS:
    H = HJ[j]; x -= x % 2
    tw = time.time()
    r = window_r(x, H)
    N, S = singular_series(x, H)
    ev = N % 2 == 0
    N, S, r = N[ev], S[ev], r[ev]
    grid = np.linspace(N[0], N[-1], 41)
    b2 = S * np.interp(N, grid, np.array([Iint(float(g)) for g in grid]))
    z = (r - b2) / np.sqrt(b2)
    rows.append(dict(j=j, x=int(x), H=H, n=int(len(N)), c2=float(z.var()), logx=math.log(x), secs=time.time() - tw))
    print(json.dumps(rows[-1]), flush=True)
    json.dump(dict(rows=rows, smoke=SMOKE), open(os.path.join(OUT, "o24_smoke.json" if SMOKE else "o24_summary.json"), "w"), indent=1, allow_nan=False)
# per-scale means, then mean squared error of each frozen form
res = {}
byj = {}
for r_ in rows: byj.setdefault(r_["j"], []).append(r_)
pts = [(np.mean([v["logx"] for v in vs]), np.mean([v["c2"] for v in vs]), np.std([v["c2"] for v in vs], ddof=1) / math.sqrt(len(vs))) for j, vs in sorted(byj.items())]
for k, f in FORMS.items():
    res[k] = dict(mse=float(np.mean([(c - f(L)) ** 2 for L, c, _ in pts])), chi2=float(np.sum([((c - f(L)) / se) ** 2 for L, c, se in pts])))
res["points"] = [[float(a), float(b), float(c)] for a, b, c in pts]
best = min(FORMS, key=lambda k: res[k]["mse"])
res["best_by_mse"] = best
res["F1"] = bool(res["a*L^b"]["mse"] < res["1-C/L"]["mse"])
print(json.dumps(res), flush=True)
json.dump(dict(rows=rows, smoke=SMOKE, checks=res), open(os.path.join(OUT, "o24_smoke.json" if SMOKE else "o24_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
