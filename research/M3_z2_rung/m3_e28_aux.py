"""M3-E28（预注册见 DESIGN.md 第 22 节）：辅助乘数 m 产生的 Thue 树变体。"""
import sys, json
from math import isqrt
from collections import Counter
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from m3_e19c_thue_tree_nomemo import rows_of
def cert(N, q, n, m):
    out = Counter(); rows_of(q * m, n, N, out)
    if m > 1: rows_of(m, (q * n) % N, N, out)
    return frozenset(x for x, c in out.items() if c)
def run(N, q, M=8, lp=False):
    R = (N - 1) // 2; load = Counter(); fails = 0; certs = []
    for n in range(1, N):
        if q * n < N: continue
        r = (q * n) % N; cl = []
        for m in range(1, M + 1):
            S = cert(N, q, n, m)
            P = q * n * r
            for x in S: P *= x * (N - x)
            if len(S) % 2 or isqrt(P) ** 2 != P: fails += 1
            for x in S: load[x] += 1.0 / M
            cl.append(S)
        certs.append(list(set(cl)))
    out = dict(N=N, q=q, M=M, square_fail=fails, avg_load_max=round(max(load.values()), 2), avg_load_mean=round(sum(load.values()) / R, 2),
               argmax=[(k, round(v, 2)) for k, v in load.most_common(3)])
    if lp:
        cols = [(k, S) for k, cl in enumerate(certs) for S in cl]; nv = len(cols) + 1; ri, ci = [], []
        for j, (k, S) in enumerate(cols):
            for x in S: ri.append(x - 1); ci.append(j)
        A_ub = coo_matrix((np.r_[np.ones(len(ri)), -np.ones(R)], (np.r_[ri, np.arange(R)], np.r_[ci, np.full(R, nv - 1)])), shape=(R, nv)).tocsr()
        A_eq = coo_matrix((np.ones(len(cols)), ([k for k, _ in cols], np.arange(len(cols)))), shape=(len(certs), nv)).tocsr()
        c = np.zeros(nv); c[-1] = 1
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(R), A_eq=A_eq, b_eq=np.ones(len(certs)), bounds=(0, None), method="highs")
        out.update(lp_t_star=round(res.fun, 3), cand_mean=round(len(cols) / len(certs), 2))
    return out
if __name__ == "__main__":
    for N in map(int, sys.argv[1].split(",")):
        for q in map(int, sys.argv[2].split(",")):
            print(json.dumps(run(N, q, lp=(N <= 20000))), flush=True)
