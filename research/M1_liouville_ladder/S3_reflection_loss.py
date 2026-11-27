"""S3 探针（探索性）：素数 p 的完整纤维上，用反射边 T(t) ⇔ T(dp − t)（d | t，t/d < p）做近似递推的损失。
每条边用到两个刚性实例：x = t/d 与 s = t mod p（按配对 {m, p−m} 记为 min(m, p−m)）。
若 T(t) 沿 BFS 树到平凡节点（t < p）的路径上没有坏实例，则 T(t) 成立。于是
  |E_q| ≤ #{n<p 未被覆盖} + Σ_{m∈E} mult_q(m)，mult_q(m) = 路径用到实例 m 的 n 的个数。
统计：覆盖率、路径长度、max/mean 多重度（与 Mangerel 的 2^{q²} 比较）。"""
import sys, json
from collections import deque, defaultdict
def run(p, qs=(2, 3, 5, 7, 11, 13), cap_mult=60):
    CAP = cap_mult * p
    spf = list(range(CAP + 1))
    for i in range(2, int(CAP ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, CAP + 1, i):
                if spf[j] == j: spf[j] = i
    def divisors(t):
        fs = {}
        while t > 1:
            q = spf[t]; fs[q] = fs.get(q, 0) + 1; t //= q
        ds = [1]
        for q, e in fs.items(): ds = [d * q ** i for d in ds for i in range(e + 1)]
        return ds
    parent = {}; depth = {}
    dq = deque()
    for t in range(1, p): depth[t] = 0; dq.append(t)
    while dq:
        u = dq.popleft()
        for d in divisors(u):
            if d < 2 or d * p <= u: continue
            v = d * p - u
            if v <= CAP and v % p and v not in depth:
                depth[v] = depth[u] + 1; parent[v] = (u, d); dq.append(v)
    pair = lambda m: min(m, p - m)
    out = {}
    for q in qs:
        mult = defaultdict(int); cov = 0; maxlen = 0; lens = []
        for n in range(1, p):
            t = q * n
            if t < p: cov += 1; continue
            if t % p == 0 or t not in depth: continue
            cov += 1; inst = set(); v = t
            while depth[v] > 0:
                u, d = parent[v]
                # 边 u—v（v = d p − u）：实例 x = u/d、s = u mod p
                inst.add(pair(u // d)); inst.add(pair(u % p)) if u % p else None
                v = u
            lens.append(depth[t])
            for m in inst: mult[m] += 1
        vals = list(mult.values()) or [0]
        out[q] = dict(coverage=cov / (p - 1), max_path=max(lens) if lens else 0,
                      mean_path=sum(lens) / len(lens) if lens else 0,
                      max_mult=max(vals), mean_mult=sum(vals) / max(len(vals), 1), mangerel_2_q2=2 ** (q * q))
    return dict(p=p, by_q=out)
for p in [int(a) for a in sys.argv[1:]]:
    r = run(p); print(json.dumps(r), flush=True)
