"""O2 (GPU): exact unweighted Goldbach counts r(N) on windows up to ~8.8e9 and the drift of r/b2_full,
b2_full = S(N) * int_2^{N-2} dt/(log t log(N-t)).  See DESIGN.md.
Run with the CUDA venv:  .venv\\Scripts\\python.exe research/O2_gpu_drift/o2_gpu_drift.py [--smoke]"""
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
JS = [20, 21] if SMOKE else [30, 31, 32, 33]
NW = 4                                   # windows per scale, at x + i*2^(j-7)
TOP = (1 << max(JS)) + NW * (1 << (max(JS) - 7)) + 4 * H + 2
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


def window_r(x):
    """r(N) = #{(p,q): p+q=N} for N = x .. x+H-1."""
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


def Iint(N):
    I, _ = integrate.quad(lambda t: 1 / (math.log(t) * math.log(N - t)), 2, N / 2, limit=400)
    return 2 * I


rows = []
for j in JS:
    x0 = 1 << j
    for w in range(NW):
        x = x0 + w * (1 << (j - 7))
        x -= x % 2
        tw = time.time()
        r = window_r(x)
        if SMOKE and w == 0:
            ip = isp[: x + H + 2].cpu().numpy().astype(bool)
            for i0 in (0, 2, 1000):
                Nn = x + i0
                if Nn % 2: continue
                brute = int(np.sum(ip[2 : Nn - 1] & ip[Nn - 2 : 1 : -1]))
                print('check', Nn, brute, r[i0], flush=True)
                assert brute == r[i0]
        N, S = singular_series(x)
        ev = N % 2 == 0
        N, S, r = N[ev], S[ev], r[ev]
        grid = np.linspace(N[0], N[-1], 41)
        Ig = np.array([Iint(float(g)) for g in grid])
        I = np.interp(N, grid, Ig)
        b2 = S * I
        rel = r / b2 - 1
        m3 = N % 3 == 0
        def st(v):
            return dict(n=int(len(v)), mean=float(v.mean()), sem=float(v.std(ddof=1) / math.sqrt(len(v))), sd=float(v.std(ddof=1)),
                        skew=float(((v - v.mean()) ** 3).mean() / v.std() ** 3), exkurt=float(((v - v.mean()) ** 4).mean() / v.std() ** 4 - 3))
        row = dict(j=j, x=int(x), all=st(rel), mod3_0=st(rel[m3]), mod3_nz=st(rel[~m3]), min_rel=float(rel.min()), max_rel=float(rel.max()),
                   min_r=float(r.min()), secs=time.time() - tw)
        rows.append(row)
        print(json.dumps({"j": j, "x": x, "mean": row["all"]["mean"], "sem": row["all"]["sem"], "sd": row["all"]["sd"], "min_r": row["min_r"], "secs": round(row["secs"], 1)}), flush=True)
        json.dump(dict(rows=rows, H=H, NW=NW, smoke=SMOKE), open(os.path.join(OUT, "o2_smoke.json" if SMOKE else "o2_summary.json"), "w"), indent=1, allow_nan=False)

# preregistered comparison (full run only)
if not SMOKE:
    def pooled(j):
        rs = [r for r in rows if r["j"] == j]
        n = sum(r["all"]["n"] for r in rs)
        mean = sum(r["all"]["mean"] * r["all"]["n"] for r in rs) / n
        sem = math.sqrt(sum((r["all"]["sem"] * r["all"]["n"]) ** 2 for r in rs)) / n
        return mean, sem
    m30, s30 = pooled(30); m33, s33 = pooled(33)
    res = dict(pooled_30=[m30, s30], pooled_33=[m33, s33], P1=bool(abs(m33) < abs(m30)), P2=bool(abs(m33) < 5e-4))
    print("pooled drift j=30: %.3e +- %.3e ; j=33: %.3e +- %.3e" % (m30, s30, m33, s33), res, flush=True)
    json.dump(dict(rows=rows, H=H, NW=NW, checks=res), open(os.path.join(OUT, "o2_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
