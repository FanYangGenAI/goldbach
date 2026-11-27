"""S3 探针 2：只允许"小乘数"反射（d ≤ Z，或 d = q·a 且 a ≤ Z）时的覆盖率与多重度——为证明做准备：
小乘数下"未覆盖"应只来自"相关数都没有 ≤ Z 的小因子"，是筛法可估计的事件。"""
import sys, json
from collections import deque, defaultdict
def run(p, q, Z, cap_mult):
    CAP = cap_mult * p
    spf = list(range(CAP + 1))
    for i in range(2, int(CAP ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, CAP + 1, i):
                if spf[j] == j: spf[j] = i
    def small_divs(t):
        # t 的所有 ≤ Z 的因子 d ≥ 2，以及 q·a（a ≤ Z, q·a | t）
        out = set()
        for d in range(2, Z + 1):
            if t % d == 0: out.add(d)
            if t % (q * d) == 0: out.add(q * d)
        if t % q == 0: out.add(q)
        return out
    depth = {}; parent = {}; dq = deque()
    for t in range(1, p): depth[t] = 0; dq.append(t)
    while dq:
        u = dq.popleft()
        for d in small_divs(u):
            if d * p <= u: continue
            v = d * p - u
            if v <= CAP and v % p and v not in depth:
                depth[v] = depth[u] + 1; parent[v] = (u, d); dq.append(v)
    pair = lambda m: min(m, p - m)
    mult = defaultdict(int); cov = 0; lens = []
    for n in range(1, p):
        t = q * n
        if t < p: cov += 1; continue
        if t not in depth: continue
        cov += 1; inst = set(); v = t
        while depth[v] > 0:
            u, d = parent[v]; inst.add(pair(u // d))
            if u % p: inst.add(pair(u % p))
            v = u
        lens.append(depth[t])
        for m in inst: mult[m] += 1
    vals = list(mult.values()) or [0]
    return dict(p=p, q=q, Z=Z, coverage=round(cov / (p - 1), 5), max_path=max(lens), mean_path=round(sum(lens) / len(lens), 3), max_mult=max(vals), mean_mult=round(sum(vals) / len(vals), 3))
for p in [4099, 16411]:
    for q in [5, 11, 23]:
        for Z in [3, 10, 30]:
            print(json.dumps(run(p, q, Z, cap_mult=q * (Z + 1) + 2)), flush=True)
