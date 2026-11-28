"""M3-E13（探索性，非预注册）：线性规划最优解中被赋予权重的反射路径长什么样（乘数的类型与大小）。"""
import sys, json
from collections import Counter
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from m3_e12_refl_arbitrary import divisors
def run(N, q, K, cap=200):
    pr = lambda m: min(m % N, N - m % N) - 1
    certs = []; meta = []
    for n in range(1, N):
        t0 = q * n
        if t0 < N: continue
        found = {}; frontier = [(t0, frozenset(), ())]
        for depth in range(K):
            nxt = []
            for t, S, ds in frontier:
                j = t // N
                for d in divisors(t):
                    if d <= j: continue
                    S2 = S ^ {pr(t // d)} ^ {pr(t)}; t2 = d * N - t; ds2 = ds + ((d, j),)
                    if t2 < N:
                        if S2 not in found: found[S2] = ds2
                    elif depth + 1 < K: nxt.append((t2, S2, ds2))
                if len(found) >= cap: break
            if len(found) >= cap: break
            frontier = nxt[:3000]
        if found:
            items = list(found.items())[:cap]
            certs.append([c for c, _ in items]); meta.append([(n, ds) for _, ds in items])
    R = (N - 1) // 2
    ri, ci, vals, er, ec = [], [], [], [], []; col = 0; colmeta = []
    for k, cl in enumerate(certs):
        for c, m in zip(cl, meta[k]):
            er.append(k); ec.append(col); colmeta.append(m)
            for r in c: ri.append(r); ci.append(col); vals.append(1.0)
            col += 1
    nv = col + 1
    for r in range(R): ri.append(r); ci.append(col); vals.append(-1.0)
    A_ub = coo_matrix((vals, (ri, ci)), shape=(R, nv)).tocsr(); A_eq = coo_matrix((np.ones(len(er)), (er, ec)), shape=(len(certs), nv)).tocsr()
    cv = np.zeros(nv); cv[-1] = 1
    res = linprog(cv, A_ub=A_ub, b_ub=np.zeros(R), A_eq=A_eq, b_eq=np.ones(len(certs)), bounds=(0, None), method="highs")
    x = res.x[:-1]
    L = Counter(); first = Counter(); dsize = Counter(); qdiv = Counter(); tot = 0.0
    for w, (n, ds) in zip(x, colmeta):
        if w < 1e-9: continue
        tot += w; L[len(ds)] += w
        d0, j0 = ds[0]
        first["free(d=j+1)" if d0 == j0 + 1 else ("q|d" if d0 % q == 0 else "q∤d")] += w
        for d, j in ds:
            dsize["d<=2q" if d <= 2 * q else ("d<=q^2" if d <= q * q else ("d<=N" if d <= N else "d>N"))] += w
            qdiv["q|d" if d % q == 0 else "q∤d"] += w
    norm = lambda c: {k: round(v / tot, 3) for k, v in sorted(c.items())}
    return dict(N=N, q=q, K=K, t_star=round(res.fun, 3), weighted_length=norm(L), first_step=norm(first), d_size=norm(dsize), q_divides_d=norm(qdiv))
if __name__ == "__main__":
    for N, q in ((401, 3), (401, 7), (401, 13)):
        print(json.dumps(run(N, q, 4)), flush=True)
