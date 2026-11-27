"""M2-E3 补充：逐个 n 比较层级图与 BFS（不是只比较比例）。"""
import numpy as np
from collections import deque
from m2_e1 import spf_table
from m2_level_graph import covered_level_graph
def bfs_cov(p, q, Z):
    CAP = q * (Z + 1) * p + p; spf = spf_table(CAP)
    def divs(t):
        fs = {}; x = t
        while x > 1:
            r = int(spf[x]); fs[r] = fs.get(r, 0) + 1; x //= r
        ds = [1]
        for r, e in fs.items(): ds = [d * r ** i for d in ds for i in range(e + 1)]
        return [d for d in ds if d >= 2 and (d <= Z or (d % q == 0 and d // q <= Z))]
    seen = np.zeros(CAP + 1, dtype=bool); seen[1:p] = True; dq = deque(range(1, p))
    while dq:
        u = dq.popleft()
        for d in divs(u):
            if d * p <= u: continue
            v = d * p - u
            if v <= CAP and v % p and not seen[v]: seen[v] = True; dq.append(v)
    return [bool(seen[q * n]) for n in range(1, p)]
tot = mism = 0
for p, q, Z in [(1009, 13, 26), (1009, 23, 46), (1009, 41, 82), (2003, 23, 69), (2003, 41, 123)]:
    a = bfs_cov(p, q, Z); b = covered_level_graph(p, q, Z)
    m = sum(x != y for x, y in zip(a, b)); tot += len(a); mism += m
    print(p, q, Z, "n checked:", len(a), "mismatches:", m, flush=True)
print("TOTAL n checked", tot, "mismatches", mism)
