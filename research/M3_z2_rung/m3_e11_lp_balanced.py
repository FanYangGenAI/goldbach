"""M3-E11（预注册，DESIGN 第 10 节）：用"随机起点 + 全量扫描"选取候选证书（≤ 4 行），求最优最大负载 t*，
并用对偶解找出瓶颈行。"""
import sys, json, time, itertools, random
from collections import defaultdict
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from m3_e8_ratio import primes_upto
def factor(m):
    f = []; d = 2
    while d * d <= m:
        while m % d == 0: f.append(d); m //= d
        d += 1
    if m > 1: f.append(m)
    return "·".join(map(str, f)) if f else "1"
def run(N, q, cap=600, per=10, seed=0):
    rng = random.Random(seed)
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
    R = [vec(m) ^ vec(N - m) ^ (1 << E) for m in range(1, (N + 1) // 2)]
    pair = defaultdict(list)
    for i, j in itertools.combinations(range(len(R)), 2):
        pair[R[i] ^ R[j]].append((i, j))
    keys = list(pair.keys()); rng.shuffle(keys); K = len(keys)
    for k in keys: rng.shuffle(pair[k])
    certs = []; unc = 0
    for n in range(1, N):
        v = vec(q * n) ^ vec((q * n) % N)
        cs = set(frozenset(p) for p in pair.get(v, [])[:cap])
        off = rng.randrange(K); t = 0
        while len(cs) < cap and t < K:
            x = keys[(off + t) % K]; t += 1
            y = v ^ x
            if y in pair:
                for a in pair[x][:per]:
                    for b in pair[y][:per]:
                        S = frozenset(a) ^ frozenset(b)
                        if len(S) in (2, 4): cs.add(S)
        if not cs: unc += 1; continue
        certs.append(list(cs)[:cap])
    rows_i, cols_i, vals = [], [], []
    eq_r, eq_c = [], []
    col = 0
    for k, cl in enumerate(certs):
        for c in cl:
            eq_r.append(k); eq_c.append(col)
            for r in c: rows_i.append(r); cols_i.append(col); vals.append(1.0)
            col += 1
    nv = col + 1
    for r in range(len(R)): rows_i.append(r); cols_i.append(col); vals.append(-1.0)
    A_ub = coo_matrix((vals, (rows_i, cols_i)), shape=(len(R), nv)).tocsr()
    A_eq = coo_matrix((np.ones(len(eq_r)), (eq_r, eq_c)), shape=(len(certs), nv)).tocsr()
    cvec = np.zeros(nv); cvec[-1] = 1
    res = linprog(cvec, A_ub=A_ub, b_ub=np.zeros(len(R)), A_eq=A_eq, b_eq=np.ones(len(certs)), bounds=(0, None), method="highs")
    out = dict(N=N, q=q, rows=len(R), no_short_cert=unc, n_cert_vars=col, t_star=round(res.fun, 3) if res.status == 0 else None, q_plus_5=q + 5)
    if res.status == 0:
        y = -np.array(res.ineqlin.marginals)  # 对偶：行上的权重
        top = np.argsort(-y)[:6]
        out["dual_top_rows"] = [dict(m=int(i) + 1, weight=round(float(y[i]), 4), m_fac=factor(int(i) + 1), Nm_fac=factor(N - int(i) - 1)) for i in top]
        out["dual_mass_on_q_rows"] = round(float(sum(y[i] for i in range(len(R)) if ((i + 1) % q == 0) or ((N - i - 1) % q == 0))), 4)
    return out
if __name__ == "__main__":
    Ns = [int(x) for x in sys.argv[1].split(",")]; qs = [int(x) for x in sys.argv[2].split(",")]
    for N in Ns:
        for q in qs:
            t0 = time.time(); r = run(N, q, seed=N + q); r["sec"] = round(time.time() - t0, 1); print(json.dumps(r), flush=True)
