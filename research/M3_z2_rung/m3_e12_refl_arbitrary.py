"""M3-E12（预注册，DESIGN 第 11 节）：允许任意乘数的反射路径（长度 ≤ K）能否覆盖全部 T(qn)，
以及只用这类证书时线性规划的最优负载 t*_refl。完整纤维，素数 N。
反射一步：节点 t（t mod N ≠ 0，层级 j = ⌊t/N⌋），乘数 d | t、d > j，t′ = dN − t；所用的行是 t/d 与 t mod N 的配对。"""
import sys, json, time
from collections import defaultdict
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
def divisors(m):
    fs = {}; x = m; p = 2
    while p * p <= x:
        while x % p == 0: fs[p] = fs.get(p, 0) + 1; x //= p
        p += 1 if p == 2 else 2
    if x > 1: fs[x] = fs.get(x, 0) + 1
    ds = [1]
    for p, e in fs.items(): ds = [d * p ** i for d in ds for i in range(e + 1)]
    return ds
def run(N, q, K, cap=200, dmax=None):
    pr = lambda m: min(m % N, N - m % N) - 1   # 行下标：配对 {m, N−m}，m ∈ [1, (N−1)/2]
    certs = []; unc = 0
    for n in range(1, N):
        t0 = q * n
        if t0 < N: certs.append([frozenset()]); continue   # 平凡：T 成立，无需行
        found = set()
        frontier = [(t0, frozenset())]
        for depth in range(K):
            nxt = []
            for t, S in frontier:
                j = t // N
                for d in divisors(t):
                    if d <= j or (dmax and d > dmax): continue
                    S2 = S ^ {pr(t // d)} ^ {pr(t)}  # 更正（E14）：原为 S ^ {a, b}，a = b 时少抵消一次
                    t2 = d * N - t
                    if t2 < N:
                        found.add(S2)
                    elif depth + 1 < K:
                        nxt.append((t2, S2))
                if len(found) >= cap: break
            if len(found) >= cap: break
            frontier = nxt[:3000]
        if not found: unc += 1; continue
        certs.append(list(found)[:cap])
    R = (N - 1) // 2
    rows_i, cols_i, vals, eq_r, eq_c = [], [], [], [], []
    col = 0
    for k, cl in enumerate(certs):
        for c in cl:
            eq_r.append(k); eq_c.append(col)
            for r in c: rows_i.append(r); cols_i.append(col); vals.append(1.0)
            col += 1
    nv = col + 1
    for r in range(R): rows_i.append(r); cols_i.append(col); vals.append(-1.0)
    A_ub = coo_matrix((vals, (rows_i, cols_i)), shape=(R, nv)).tocsr()
    A_eq = coo_matrix((np.ones(len(eq_r)), (eq_r, eq_c)), shape=(len(certs), nv)).tocsr()
    cvec = np.zeros(nv); cvec[-1] = 1
    res = linprog(cvec, A_ub=A_ub, b_ub=np.zeros(R), A_eq=A_eq, b_eq=np.ones(len(certs)), bounds=(0, None), method="highs")
    return dict(N=N, q=q, K=K, dmax=dmax, uncovered=unc, frac_uncovered=round(unc / (N - 1), 4),
                t_star_refl=round(res.fun, 3) if res.status == 0 else None, q_plus_5=q + 5)
if __name__ == "__main__":
    Ns = [int(x) for x in sys.argv[1].split(",")]; qs = [int(x) for x in sys.argv[2].split(",")]; K = int(sys.argv[3])
    for N in Ns:
        for q in qs:
            t0 = time.time(); r = run(N, q, K); r["sec"] = round(time.time() - t0, 1); print(json.dumps(r), flush=True)
