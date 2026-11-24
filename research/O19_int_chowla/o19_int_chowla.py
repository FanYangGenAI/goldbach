r"""O19 (preregistered): the integer Chowla-Goldbach fibre sum  Lam(N) = sum_{1<=n<N} lambda(n) lambda(N-n)
(Liouville lambda), integer analogue of O15.  lambda from a CPU numba Omega sieve, convolution on the RTX 4090 for every N
in windows near 1e8 and 1e9.  Reports Lam/sqrt(N) scale, mean, Gaussianity, cross-N autocorrelation and locality.
Step 1 (system python with numba): python lambda_sieve.py TOP OUTFILE ; step 2: .venv\Scripts\python.exe research/O19_int_chowla/o19_int_chowla.py [--smoke]"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv
dev = torch.device("cuda")
H = 1 << (14 if SMOKE else 21)
XS = [10**5, 10**6] if SMOKE else [10**8, 10**9]
TOP = max(XS) + 4 * H + 2
C2 = 0.6601618158468695739278121100145
t0 = time.time()


LAMFILE = os.environ.get("O19_LAMBDA", os.path.join(OUT, "lambda_smoke.npy" if SMOKE else "lambda.npy"))
lam_cpu = np.load(LAMFILE, mmap_mode="r")
assert len(lam_cpu) >= TOP + 1, (len(lam_cpu), TOP)
lam_cpu = np.ascontiguousarray(lam_cpu[: TOP + 1])
lam = torch.from_numpy(lam_cpu).to(dev)
del lam_cpu
small = np.ones(int(math.isqrt(TOP)) + 2, dtype=bool); small[:2] = False
for i in range(2, int(len(small) ** 0.5) + 1):
    if small[i]: small[i * i :: i] = False
base = np.nonzero(small)[0]
print(f"lambda sieve to {TOP:,}: {time.time()-t0:.0f}s", flush=True)


def ind(lo, n):
    out = torch.zeros(n, dtype=torch.float64, device=dev)
    a = max(lo, 1); b = min(lo + n, TOP)
    if b > a: out[a - lo : b - lo] = lam[a:b].to(torch.float64)
    return out


def window_conv(x):
    L = 4 * H
    acc = torch.zeros(H, dtype=torch.float64, device=dev)
    for k in range((x + H) // H + 2):
        if k * H > x + H: break
        a = ind(k * H, H); b = ind(x - (k + 1) * H + 1, 2 * H)
        acc += torch.fft.irfft(torch.fft.rfft(a, L) * torch.fft.rfft(b, L), L)[H - 1 : 2 * H - 1]
    return torch.round(acc).cpu().numpy()


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
    tw = time.time()
    Lam = window_conv(x)
    N = np.arange(x, x + H, dtype=np.int64)
    if SMOKE:
        lc = lam.cpu().numpy().astype(np.int64)
        for i0 in (0, 1, 777):
            Nn = int(N[i0]); n = np.arange(1, Nn)
            assert int((lc[n] * lc[Nn - n]).sum()) == Lam[i0]
        print("brute check ok", flush=True)
    y = Lam / np.sqrt(N)
    rec = dict(x=int(x), n=int(H))
    for name, m in (("all", np.ones(H, bool)), ("even", N % 2 == 0), ("odd", N % 2 == 1)):
        v = y[m]; vc = (v - v.mean()) / v.std()
        rec[name] = dict(mean=float(v.mean()), sd=float(v.std()), skew=float((vc**3).mean()), exkurt=float((vc**4).mean() - 3),
                         maxabs=float(np.abs(vc).max()))
    ev = N % 2 == 0
    ye = y[ev]; yc = (ye - ye.mean()) / ye.std()
    rec["autocorr_even"] = {str(L): float(np.mean(yc[:-L] * yc[L:])) for L in (1, 2, 3, 5, 15, 105)}
    F = features(N[ev]); Ne = N[ev]
    S = np.full(len(Ne), 2 * C2)
    for p in base[1:]:
        p = int(p)
        if p > 10**5: break
        m = Ne % p == 0
        S[m] *= (p - 1) / (p - 2)
    Xf = np.column_stack([F, np.log(S)])
    rec["R2_local_even"] = r2(ye, Xf); rec["R2_each_even"] = [r2(ye, Xf[:, [j]]) for j in range(Xf.shape[1])]
    rec["secs"] = time.time() - tw
    rows.append(rec)
    print(json.dumps(rec), flush=True)
    json.dump(dict(rows=rows, H=H, smoke=SMOKE), open(os.path.join(OUT, "o19_smoke.json" if SMOKE else "o19_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
