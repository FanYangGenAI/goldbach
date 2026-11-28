"""M3-E14（预注册见 DESIGN.md 第 12 节）：线性规划界的有理数复核，以及平方类独立检验。"""
import sys, json
from fractions import Fraction
from math import isqrt
from collections import defaultdict
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from m3_e12_refl_arbitrary import divisors
def gen(N, q, K, cap=200):
    """与 E12/E13 相同的路径生成，但用多重集记录行（同时保留 E12 的集合字面量写法以便对照）。"""
    pr = lambda m: min(m % N, N - m % N)  # 返回代表元 x ∈ [1, (N-1)/2]
    out = {}
    for n in range(1, N):
        t0 = q * n
        if t0 < N: continue
        found = {}; frontier = [(t0, frozenset(), frozenset())]
        for depth in range(K):
            nxt = []
            for t, S, Sold in frontier:
                j = t // N
                for d in divisors(t):
                    if d <= j: continue
                    a, b = pr(t // d), pr(t)
                    S2 = S ^ {a} ^ {b}            # 正确：a = b 时两次抵消
                    So2 = Sold ^ {a, b}            # E12 写法
                    t2 = d * N - t
                    if t2 < N:
                        if S2 not in found: found[S2] = So2
                    elif depth + 1 < K: nxt.append((t2, S2, So2))
                if len(found) >= cap: break
            if len(found) >= cap: break
            frontier = nxt[:3000]
        if found: out[n] = list(found.items())[:cap]
    return out
def square_ok(N, q, n, S):
    r = (q * n) % N; P = q * n * r
    for x in S: P *= x * (N - x)
    return len(S) % 2 == 0 and isqrt(P) ** 2 == P
def run(N, q, K):
    C = gen(N, q, K); R = (N - 1) // 2
    bad = sum(1 for n, cl in C.items() for S, _ in cl if not square_ok(N, q, n, S))
    diff = sum(1 for n, cl in C.items() for S, So in cl if S != So)
    uncovered = sum(1 for n in range(1, N) if q * n >= N and n not in C)
    cols = [(n, S) for n, cl in C.items() for S, _ in cl]
    ri, ci = [], []; er = []
    for k, (n, S) in enumerate(cols):
        for x in S: ri.append(x - 1); ci.append(k)
    nv = len(cols) + 1
    rows_n = {n: i for i, n in enumerate(C)}
    A_ub = coo_matrix((np.r_[np.ones(len(ri)), -np.ones(R)], (np.r_[ri, np.arange(R)], np.r_[ci, np.full(R, nv - 1)])), shape=(R, nv)).tocsr()
    A_eq = coo_matrix((np.ones(len(cols)), ([rows_n[n] for n, _ in cols], np.arange(len(cols)))), shape=(len(C), nv)).tocsr()
    cv = np.zeros(nv); cv[-1] = 1
    res = linprog(cv, A_ub=A_ub, b_ub=np.zeros(R), A_eq=A_eq, b_eq=np.ones(len(C)), bounds=(0, None), method="highs")
    x = res.x[:-1]
    W = defaultdict(list)
    for (n, S), w in zip(cols, x):
        f = Fraction(round(max(w, 0.0) * 10 ** 6), 10 ** 6)
        if f > 0: W[n].append((S, f))
    load = defaultdict(Fraction)
    for n, lst in W.items():
        tot = sum(f for _, f in lst)
        for S, f in lst:
            assert square_ok(N, q, n, S)
            for xr in S: load[xr] += f / tot
    t_exact = max(load.values())
    return dict(N=N, q=q, K=K, certs=len(cols), square_fail=bad, e12_set_diff=diff, uncovered=uncovered,
                t_float=round(res.fun, 4), t_exact=float(t_exact), t_exact_frac=str(t_exact.limit_denominator(10 ** 6)), n_used=len(W))
if __name__ == "__main__":
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    for spec in sys.argv[1].split(";"):
        N, q = map(int, spec.split(","))
        print(json.dumps(run(N, q, K)), flush=True)
