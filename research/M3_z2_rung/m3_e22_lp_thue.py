"""M3-E22（预注册见 DESIGN.md 第 19 节）：在 Thue 树变体中用线性规划做负载均衡。"""
import sys, json, random
from math import isqrt
from collections import Counter
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from m3_e19c_thue_tree_nomemo import rows_of
from m3_e21_random_thue import rows_rand
def run(N, q, K=64, seed=7):
    rng = random.Random(seed); R = (N - 1) // 2; certs = []
    for n in range(1, N):
        if q * n < N: continue
        r = (q * n) % N; cs = set()
        out = Counter(); rows_of(q, n, N, out); cs.add(frozenset(x for x, c in out.items() if c))
        for _ in range(K - 1):
            out = Counter(); rows_rand(q, n, N, out, rng); cs.add(frozenset(x for x, c in out.items() if c))
        for S in cs:
            P = q * n * r
            for x in S: P *= x * (N - x)
            assert len(S) % 2 == 0 and isqrt(P) ** 2 == P
        certs.append(list(cs))
    cols = [(k, S) for k, cl in enumerate(certs) for S in cl]; nv = len(cols) + 1
    ri, ci = [], []
    for j, (k, S) in enumerate(cols):
        for x in S: ri.append(x - 1); ci.append(j)
    A_ub = coo_matrix((np.r_[np.ones(len(ri)), -np.ones(R)], (np.r_[ri, np.arange(R)], np.r_[ci, np.full(R, nv - 1)])), shape=(R, nv)).tocsr()
    A_eq = coo_matrix((np.ones(len(cols)), ([k for k, _ in cols], np.arange(len(cols)))), shape=(len(certs), nv)).tocsr()
    c = np.zeros(nv); c[-1] = 1
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(R), A_eq=A_eq, b_eq=np.ones(len(certs)), bounds=(0, None), method="highs")
    return dict(N=N, q=q, targets=len(certs), cand_mean=round(len(cols) / len(certs), 1), t_star=round(res.fun, 3), status=res.status)
if __name__ == "__main__":
    for q in map(int, sys.argv[2].split(",")):
        print(json.dumps(run(int(sys.argv[1]), q)), flush=True)
