"""R13-2 (preregistered; follows the round-13 review): cross-scale structure of the Liouville fibre sum
C(N) = sum_{1<=n<N} lambda(n) lambda(N-n).  Model (review eq. 12 + two unproved assumptions): C(N) = sum_{m|N} C*(m),
primitive fibres orthogonal with Var C*(m) ~ 2 phi(m)  =>  Cov(C(N), C(M)) ~ 2 gcd(N, M), so
Corr(C(aN)/sqrt(aN), C(bN)/sqrt(bN)) ~ gcd(a, b) / sqrt(a b).  Usage: python r13_dilation.py [--smoke]"""
import json, math, os, sys, time
import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv
LO, HI = (20000, 40000) if SMOKE else (1_000_000, 2_000_000)
PAIRS = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (3, 4), (2, 6), (4, 6), (5, 7)]
t0 = time.time()


@nb.njit(cache=True)
def liouville(n):
    spf = np.zeros(n + 1, dtype=np.int32)
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            j = np.int64(i) * i
            while j <= n:
                if spf[j] == 0:
                    spf[j] = i
                j += i
    lam = np.zeros(n + 1, dtype=np.int8); lam[1] = 1
    for m in range(2, n + 1):
        lam[m] = -lam[m // spf[m]]
    return lam


TOP = 7 * HI + 2
lam = liouville(TOP).astype(np.float64); lam[0] = 0
L = 1 << int(math.ceil(math.log2(2 * TOP + 2)))
f = np.fft.rfft(lam, L)
C = np.rint(np.fft.irfft(f * f, L)[: TOP + 1])        # C[M] = sum_{n=0..M} lam(n) lam(M-n) = sum_{1<=n<M}
print(f"lambda + FFT to {TOP:,}: {time.time()-t0:.0f}s", flush=True)
# brute check
for M in (LO, LO + 1, 3 * LO + 7):
    n = np.arange(1, M)
    assert int((lam[n] * lam[M - n]).sum()) == int(C[M]), M
N = np.arange(LO, HI)
res = {}
for a, b in PAIRS:
    x = C[a * N] / np.sqrt(a * N); y = C[b * N] / np.sqrt(b * N)
    r = float(np.corrcoef(x, y)[0, 1]); pred = math.gcd(a, b) / math.sqrt(a * b)
    res[f"{a}N,{b}N"] = dict(corr=r, pred=pred, diff=r - pred, sd_x=float(x.std()), sd_y=float(y.std()))
    print(a, b, round(r, 4), round(pred, 4), flush=True)
# control: unrelated shift
x = C[N] / np.sqrt(N); y = C[N + 2] / np.sqrt(N + 2)
res["N,N+2"] = dict(corr=float(np.corrcoef(x, y)[0, 1]), pred=0.0)
res["D1"] = bool(all(abs(v["diff"]) < 0.03 for k, v in res.items() if "," in k and k != "N,N+2"))
res["D2"] = bool(abs(res["N,N+2"]["corr"]) < 0.03)
print(json.dumps({k: v for k, v in res.items() if k in ("D1", "D2")}), flush=True)
json.dump(dict(range=[LO, HI], results=res, smoke=SMOKE), open(os.path.join(OUT, "r13_dilation_smoke.json" if SMOKE else "r13_dilation.json"), "w"), indent=1)
print(f"done {time.time()-t0:.0f}s")
