"""M3-E42（预注册见 DESIGN.md 第 28 节）：祖先数分布的尾部形状。"""
import sys, json, math
from collections import defaultdict, Counter
import numpy as np
def ancs(N, q):
    cnt = defaultdict(int)
    for n in range(1, N):
        if q * n >= N: cnt[(q, n)] = 1
    allc = Counter()
    for a in range(q, 1, -1):
        items = [(x, c) for (aa, x), c in cnt.items() if aa == a]
        for x, c in items:
            if a < q: allc[c] += 1
            if a * x < N: continue
            for v in range(1, a):
                u = (x * v) % N
                if u * a < N or (N - u) * a < N: break
            if v > 1: cnt[(v, x)] += c; cnt[(v, (a * x) % N)] += c
        for x, _ in items: del cnt[(a, x)]
    return allc
def fit(allc):
    tot = sum(allc.values()); ts = sorted(allc); surv = []; acc = tot
    for t in ts: surv.append((t, acc / tot)); acc -= allc[t]
    med = next(t for t, s in surv if s <= 0.5)
    pts = [(t, s) for t, s in surv if t >= med and s * tot >= 5]
    T = np.array([t for t, _ in pts], float); S = np.log(np.array([s for _, s in pts]))
    A = np.vstack([T, np.ones_like(T)]).T; ce, re, *_ = np.linalg.lstsq(A, S, rcond=None)
    A2 = np.vstack([np.log(T), np.ones_like(T)]).T; cp, rp, *_ = np.linalg.lstsq(A2, S, rcond=None)
    rexp = float(np.mean((A @ ce - S) ** 2)); rpow = float(np.mean((A2 @ cp - S) ** 2))
    return dict(median=med, max=max(ts), exp_slope=round(float(ce[0]), 4), exp_resid=round(rexp, 4), pow_slope=round(float(cp[0]), 3), pow_resid=round(rpow, 4), npts=len(pts))
if __name__ == "__main__":
    for N in map(int, sys.argv[1].split(",")):
        for q in map(int, sys.argv[2].split(",")):
            print(json.dumps(dict(N=N, q=q, **fit(ancs(N, q)))), flush=True)
