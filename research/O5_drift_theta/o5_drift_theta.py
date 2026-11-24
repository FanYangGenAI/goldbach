"""O5 (GPU, preregistered): is the O2 drift of r/b2 - 1 (about -1.5/sqrt(N)) the first-order prime-counting
correction?  Predictor (no free parameters), for window centre M:
    rel_pred(M) = 2 * [ sum_{2<=p<=M-2} rho(M-p) - int_2^{M-2} rho(t) rho(M-t) dt ] / int_2^{M-2} rho(t) rho(M-t) dt,
rho = 1/log.  This is the linear term of r = sum (rho+delta)(n)(rho+delta)(M-n) with delta = d(pi - li).
See DESIGN.md.  Run:  .venv\\Scripts\\python.exe research/O5_drift_theta/o5_drift_theta.py [--smoke]"""
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
LO, HI, NWIN = (22, 25, 8) if SMOKE else (30, 33, 96)
XS = []
for i in range(NWIN):
    x = int(round(2 ** (LO + (HI - LO) * i / NWIN))); x -= x % 2
    XS.append(x)
TOP = max(XS) + 4 * H + 2
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
for wi, x in enumerate(XS):
    tw = time.time()
    r = window_r(x)
    if SMOKE and wi == 0:
        ip = isp[: x + H + 2].cpu().numpy().astype(bool)
        for i0 in (0, 2, 1000):
            Nn = x + i0
            brute = int(np.sum(ip[2 : Nn - 1] & ip[Nn - 2 : 1 : -1]))
            assert brute == r[i0], (Nn, brute, r[i0])
        print("brute check ok", flush=True)
    N, S = singular_series(x)
    ev = N % 2 == 0
    N, S, r = N[ev], S[ev], r[ev]
    grid = np.linspace(N[0], N[-1], 41)
    Ig = np.array([Iint(float(g)) for g in grid])
    rel = r / (S * np.interp(N, grid, Ig)) - 1
    Ms = [int(N[0]), int(N[len(N) // 2]), int(N[-1])]
    preds = [rel_pred(M) for M in Ms]
    row = dict(i=wi, x=int(x), mean=float(rel.mean()), sem=float(rel.std(ddof=1) / math.sqrt(len(rel))), sd=float(rel.std(ddof=1)),
               pred=float(np.mean(preds)), pred_ends=[float(v) for v in preds], sqrt_x=math.sqrt(x), secs=time.time() - tw)
    rows.append(row)
    print(json.dumps({k: row[k] for k in ("i", "x", "mean", "pred", "sd")} | {"secs": round(row["secs"], 1)}), flush=True)
    json.dump(dict(rows=rows, H=H, smoke=SMOKE), open(os.path.join(OUT, "o5_smoke.json" if SMOKE else "o5_summary.json"), "w"), indent=1, allow_nan=False)

obs = np.array([r["mean"] * r["sqrt_x"] for r in rows]); prd = np.array([r["pred"] * r["sqrt_x"] for r in rows])
res = dict(mean_abs_obs=float(np.abs(obs).mean()), mean_abs_resid=float(np.abs(obs - prd).mean()),
           corr=float(np.corrcoef(obs, prd)[0, 1]), slope=float(np.polyfit(prd, obs, 1)[0]),
           mean_obs=float(obs.mean()), mean_pred=float(prd.mean()))
res["P1"] = bool(res["mean_abs_resid"] < 0.5)
res["P2"] = bool(res["corr"] > 0.5 and 0.7 <= res["slope"] <= 1.3)
print(json.dumps(res), flush=True)
json.dump(dict(rows=rows, H=H, smoke=SMOKE, checks=res), open(os.path.join(OUT, "o5_smoke.json" if SMOKE else "o5_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
