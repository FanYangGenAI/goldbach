"""按路径长度与"字型"统计覆盖来源：每一步的乘数分为 F（= q，免费）、Q（= q·a，a≥2）、G（与 q 互素的一般乘数），
最后一步总是"结束步"（d = 层级 + 1）。目的：找出主导的字型，为引理 R 的证明设计路径族。"""
import sys, json
import numpy as np
from collections import deque, Counter
import m2_e1 as M
def run(p, q, Z):
    CAP = q * (Z + 1) * p + p
    spf = M.spf_table(CAP)
    def allowed_divs(t):
        fs = {}; x = t
        while x > 1:
            r = int(spf[x]); fs[r] = fs.get(r, 0) + 1; x //= r
        ds = [1]
        for r, e in fs.items(): ds = [d * r ** i for d in ds for i in range(e + 1)]
        return sorted(d for d in ds if d >= 2 and (d <= Z or (d % q == 0 and d // q <= Z)))
    depth = np.full(CAP + 1, -1, dtype=np.int8); par_u = np.zeros(CAP + 1, dtype=np.int64); par_d = np.zeros(CAP + 1, dtype=np.int32)
    dq = deque(range(1, p)); depth[1:p] = 0
    while dq:
        u = dq.popleft()
        for d in allowed_divs(u):
            if d * p <= u: continue
            v = d * p - u
            if v <= CAP and v % p and depth[v] < 0:
                depth[v] = depth[u] + 1; par_u[v] = u; par_d[v] = d; dq.append(v)
    lens = Counter(); types = Counter(); lev = Counter()
    for n in range(1, p):
        t = q * n
        if t < p: lens[0] += 1; continue
        if depth[t] < 0: lens["none"] += 1; continue
        v = t; w = []
        while depth[v] > 0:
            u = int(par_u[v]); d = int(par_d[v])
            w.append("F" if d == q else ("Q" if d % q == 0 else "G")); v = u
        # BFS 树里路径方向：从 t 出发，第一个记录的是离 t 最近的一条边
        lens[len(w)] += 1; types["".join(w)] += 1
    tot = p - 1
    return dict(p=p, q=q, Z=Z, length_dist={str(k): round(v / tot, 4) for k, v in sorted(lens.items(), key=lambda kv: str(kv[0]))},
                top_types=[(k, round(v / tot, 4)) for k, v in types.most_common(12)])
for q in (13, 41):
    print(json.dumps(run(1009, q, 6 * q)), flush=True)
