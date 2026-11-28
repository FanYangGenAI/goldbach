"""M3-E10（预注册，DESIGN 第 9 节）：短证书系统的最优最大负载（LP）。"""
import sys, json, time, itertools
from collections import defaultdict
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from m3_e8_ratio import primes_upto
def run(N, q, cap=30):
    P = [p for p in primes_upto(N) if p != N]; idx = {p: i for i, p in enumerate(P)}; E = len(P)
    L = q * N + 1; spf = list(range(L + 1))
    for i in range(2, int(L ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, L + 1, i):
                if spf[j] == j: spf[j] = i
    def vec(n):
        v = 0
        while n > 1:
            p = spf[n]; c = 0
            while n % p == 0: n //= p; c ^= 1
            if c: v ^= 1 << idx[p]
        return v
    R = [vec(n) ^ vec(N - n) ^ (1 << E) for n in range(1, (N + 1) // 2)]
    pair = defaultdict(list)
    for i, j in itertools.combinations(range(len(R)), 2):
        pair[R[i] ^ R[j]].append((i, j))
    certs = []; unc = 0
    for n in range(1, N):
        v = vec(q * n) ^ vec((q * n) % N)
        cs = set()
        for (i, j) in pair.get(v, [])[:cap]: cs.add(frozenset((i, j)))
        if len(cs) < cap:
            for x, lst in pair.items():
                y = v ^ x
                if y in pair:
                    for a in lst[:3]:
                        for b in pair[y][:3]:
                            S = frozenset(a) ^ frozenset(b)
                            if len(S) in (2, 4): cs.add(S)
                    if len(cs) >= cap: break
        if not cs: unc += 1; continue
        certs.append(list(cs)[:cap])
    # LP 变量：每个 (n, c) 的权重，以及 t
    var = [(k, c) for k, cl in enumerate(certs) for c in cl]
    nv = len(var) + 1
    A_eq = lil_matrix((len(certs), nv)); b_eq = np.ones(len(certs))
    A_ub = lil_matrix((len(R), nv)); b_ub = np.zeros(len(R))
    for col, (k, c) in enumerate(var):
        A_eq[k, col] = 1
        for r in c: A_ub[r, col] += 1
    for r in range(len(R)): A_ub[r, nv - 1] = -1
    cvec = np.zeros(nv); cvec[-1] = 1
    res = linprog(cvec, A_ub=A_ub.tocsr(), b_ub=b_ub, A_eq=A_eq.tocsr(), b_eq=b_eq, bounds=[(0, None)] * nv, method="highs")
    return dict(N=N, q=q, rows=len(R), targets=N - 1, no_short_cert=unc, frac_no_short=round(unc / (N - 1), 4),
                lp_status=res.status, t_star=round(res.fun, 3) if res.status == 0 else None, q_plus_5=q + 5)
if __name__ == "__main__":
    for N in [int(x) for x in sys.argv[1].split(",")]:
        for q in (3, 7, 13):
            if q >= N: continue
            t0 = time.time(); r = run(N, q); r["sec"] = round(time.time() - t0, 1); print(json.dumps(r), flush=True)
