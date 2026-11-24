"""O11 (GPU, preregistered): the O6 refined local model of chi at N ~ 1e10, with exact PP/PS/SP/SS counts for every
even N in a window computed by FFT on the RTX 4090.  For N < z^3 (z = floor(N^{1/3})) a z-rough n in (z, N-z) has at most
two prime factors, so S = rough - prime.  One z per window (windows chosen so that z is constant on them).
See DESIGN.md.  Run:  .venv\\Scripts\\python.exe research/O11_chi_gpu/o11_chi_gpu.py [--smoke]"""
import json, math, os, sys, time
import numpy as np
import torch
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv
dev = torch.device("cuda")
H = 1 << (14 if SMOKE else 21)
SCALES = [10**6, 10**7] if SMOKE else [10**9, 3 * 10**9, 10**10]
K = 40 if SMOKE else 300
SEED = 2026112414
TOP = int(max(SCALES) * 1.0) + 4 * H + 2
t0 = time.time()


def icbrt(n):
    z = int(round(n ** (1.0 / 3.0)))
    while z**3 > n: z -= 1
    while (z + 1) ** 3 <= n: z += 1
    return z


NO = TOP // 2 + 1
prime_odd = torch.ones(NO, dtype=torch.uint8, device=dev); prime_odd[0] = 0
r_ = int(math.isqrt(TOP)) + 1
small = np.ones(r_ + 1, dtype=bool); small[:2] = False
for i in range(2, int(r_**0.5) + 1):
    if small[i]: small[i * i :: i] = False
base = np.nonzero(small)[0]
for p in base[1:]:
    p = int(p); prime_odd[(p * p - 1) // 2 :: p] = 0
torch.cuda.synchronize()
print(f"GPU odd prime sieve to {TOP:,}: {time.time()-t0:.1f}s", flush=True)


def rough_table(z):
    """odd-only indicator of z-rough integers > z."""
    R = torch.ones(NO, dtype=torch.uint8, device=dev)
    R[: (z - 1) // 2 + 1] = 0           # odd numbers <= z (and 1) excluded
    for p in base[1:]:
        p = int(p)
        if p > z: break
        R[(p - 1) // 2 :: p] = 0          # odd multiples of p (p itself already excluded since p <= z)
    return R


def ind(arr, lo, n):
    out = torch.zeros(n, dtype=torch.float64, device=dev)
    a = max(lo, 0); b = min(lo + n, TOP)
    if b <= a: return out
    s = a if a % 2 == 1 else a + 1
    if s >= b: return out
    i0 = (s - 1) // 2; cnt = (b - s + 1) // 2
    out[s - lo : b - lo : 2] = arr[i0 : i0 + cnt].to(torch.float64)
    return out


def window_conv(fa, fb, x):
    """sum_n fa(n) fb(N - n) for N = x .. x+H-1 (blocked FFT)."""
    L = 4 * H
    acc = torch.zeros(H, dtype=torch.float64, device=dev)
    for k in range((x + H) // H + 2):
        if k * H > x + H: break
        a = fa(k * H, H); b = fb(x - (k + 1) * H + 1, 2 * H)
        acc += torch.fft.irfft(torch.fft.rfft(a, L) * torch.fft.rfft(b, L), L)[H - 1 : 2 * H - 1]
    return torch.round(acc).cpu().numpy()


def odd_prime_factors(N):
    ps = []; m = N
    while m % 2 == 0: m //= 2
    p = 3
    while p * p <= m:
        if m % p == 0:
            ps.append(p)
            while m % p == 0: m //= p
        p += 2
    if m > 1: ps.append(m)
    return ps


def chi_from(pp, ps, sp, ss):
    A = pp + ps + sp + ss; B = -pp + ss + (ps - sp); C = pp - ps - sp + ss
    return C / A - (B / A) ** 2


def predict(N, z, big):
    """identical to O6."""
    lz = math.log(z)
    def r1(t): return 1 / math.log(t)
    def r2(t):
        L = math.log(t)
        return 0.0 if L <= 2 * lz else math.log((L - lz) / lz) / L
    def w(p, t):
        if t / p <= z: return 0.0
        d = r2(t)
        return 0.0 if d <= 0 else min(1.0, 1 / (p * math.log(t / p)) / d)
    def g(kind, t):
        v = 1.0
        for p in big:
            a, b = w(p, t), w(p, N - t)
            if kind == 1: v *= 1 - b
            elif kind == 2: v *= 1 - a
            elif kind == 3: v *= (p - 1) * a * b + (1 - a) * (1 - b)
        return v
    funcs = [(r1, r1, 0), (r1, r2, 1), (r2, r1, 2), (r2, r2, 3)]
    lo, hi = z + 1, N - z - 1
    pts = sorted({min(max(x, lo), hi) for p in big for x in (p * z, N - p * z)} | {N / 2})
    res_base, res_ref = [], []
    for f, h, kind in funcs:
        edges = [lo] + [x for x in pts if lo < x < hi] + [hi]
        Ib = Ir = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            Ib += integrate.quad(lambda t: f(t) * h(N - t), a, b, limit=400)[0]
            Ir += integrate.quad(lambda t: f(t) * h(N - t) * g(kind, t), a, b, limit=400)[0]
        res_base.append(Ib); res_ref.append(Ir)
    return chi_from(*res_base), chi_from(*res_ref)


def nullv(p):
    c = np.array([1, -1, -1, 1.0]); b = np.array([-1, 1, -1, 1.0])
    cb, bb = (p * c).sum(), (p * b).sum()
    S = np.array([[(p * c * c).sum() - cb * cb, (p * c * b).sum() - cb * bb], [(p * c * b).sum() - cb * bb, (p * b * b).sum() - bb * bb]])
    g = np.array([1.0, -2 * bb]); return float(g @ S @ g)


def centre(v, g):
    v = np.asarray(v, float).copy()
    for gg in (True, False):
        if (g == gg).any(): v[g == gg] -= v[g == gg].mean()
    return v


rng = np.random.default_rng(SEED)
summ = {}; rows = []
for X in SCALES:
    x = X - X % 2
    while icbrt(x) != icbrt(x + H): x += H          # z constant on the window
    z = icbrt(x)
    tw = time.time()
    R = rough_table(z)
    fP = lambda lo, n: ind(prime_odd, lo, n) * (lo + torch.arange(n, device=dev, dtype=torch.float64) > z)
    fS = lambda lo, n: ind(R, lo, n) - fP(lo, n)
    PP = window_conv(fP, fP, x); PS = window_conv(fP, fS, x); SP = window_conv(fS, fP, x); SS = window_conv(fS, fS, x)
    del R; torch.cuda.empty_cache()
    Ns = np.arange(x, x + H, dtype=np.int64); ev = Ns % 2 == 0
    Ns, PP, PS, SP, SS = Ns[ev], PP[ev], PS[ev], SP[ev], SS[ev]
    print(f"X={X:.0e}: z={z}, window {x}..{x+H}, conv {time.time()-tw:.0f}s", flush=True)
    if SMOKE:
        # brute check of the four counts for 3 N
        lim = x + H + 2
        sv = np.ones(lim + 1, dtype=bool); sv[:2] = False
        for i in range(2, int(lim**0.5) + 1):
            if sv[i]: sv[i * i :: i] = False
        rough = np.ones(lim + 1, dtype=bool); rough[: z + 1] = False
        for p in range(2, z + 1):
            if sv[p]: rough[p::p] = False
        for i in (0, 7, 100):
            N = int(Ns[i]); n = np.arange(z + 1, N - z)
            a, b = rough[n], rough[N - n]; pa, pb = sv[n], sv[N - n]
            m = a & b
            cnt = [int((m & pa & pb).sum()), int((m & pa & ~pb).sum()), int((m & ~pa & pb).sum()), int((m & ~pa & ~pb).sum())]
            assert cnt == [PP[i], PS[i], SP[i], SS[i]], (N, cnt, PP[i], PS[i], SP[i], SS[i])
        print("brute check ok", flush=True)
    A = PP + PS + SP + SS
    chi = (PP - PS - SP + SS) / A - ((-PP + SS + PS - SP) / A) ** 2
    P4 = np.column_stack([PP, PS, SP, SS]) / A[:, None]
    s = np.sqrt(np.array([nullv(p) for p in P4[:: max(1, len(A) // 20000)]]).mean() / A)   # null variance coefficient ~ constant
    D_window = float(((chi - chi.mean()) / s).std())
    idx = np.sort(rng.choice(len(Ns), size=K, replace=False))
    obs, pr, pb = [], [], []
    for i in idx:
        N = int(Ns[i]); big = [p for p in odd_prime_factors(N) if p > z]
        cb, cr = predict(N, z, big)
        obs.append(chi[i]); pr.append(cr); pb.append(cb)
        rows.append(dict(X=X, N=N, z=z, big=big, counts=[int(PP[i]), int(PS[i]), int(SP[i]), int(SS[i])], chi_obs=float(chi[i]), chi_ref=cr, chi_base=cb))
    obs, pr, pb = map(np.array, (obs, pr, pb))
    g = Ns[idx] % 3 == 0
    co, cp = centre(obs, g), centre(pr, g)
    sres = s[idx]
    summ[str(X)] = dict(z=z, x=int(x), rho=float(np.corrcoef(co, cp)[0, 1]), sd_ratio=float(cp.std(ddof=1) / co.std(ddof=1)),
                        R2_pred=float(1 - np.sum((co - cp) ** 2) / np.sum(co**2)), mean_obs=float(obs.mean()), mean_ref=float(pr.mean()),
                        mean_base=float(pb.mean()), D_window=D_window, D_sample_before=float(((obs - obs.mean()) / sres).std()),
                        D_sample_after=float((((obs - pr) - (obs - pr).mean()) / sres).std()), window_chi_mean=float(chi.mean()),
                        window_chi_sd=float(chi.std()), n_window=int(len(Ns)), secs=time.time() - tw)
    print(X, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in summ[str(X)].items()}), flush=True)
    json.dump(dict(summary=summ, rows=rows, smoke=SMOKE), open(os.path.join(OUT, "o11_smoke.json" if SMOKE else "o11_summary.json"), "w"), indent=1, allow_nan=False)
key = summ[str(max(SCALES))]
verdict = "support" if (key["rho"] > 0.5 and 0.5 <= key["sd_ratio"] <= 2 and key["R2_pred"] > 0.25) else ("reject" if (key["rho"] < 0.2 or key["sd_ratio"] < 0.1) else "partial")
print("verdict:", verdict, flush=True)
json.dump(dict(summary=summ, verdict=verdict, rows=rows, smoke=SMOKE), open(os.path.join(OUT, "o11_smoke.json" if SMOKE else "o11_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
