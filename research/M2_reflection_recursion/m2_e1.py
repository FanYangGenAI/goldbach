"""M2-E1（预注册，见 DESIGN.md）：受限反射路径的覆盖率 u_q 与多重度 M_q。
允许的反射乘数：d ≤ Z，或 d = q·a（a ≤ Z），且 d | 当前节点、d > 当前层级（d·p > 节点）。"""
import sys, json, time
import numpy as np
from collections import deque, defaultdict
def spf_table(n):
    spf = np.zeros(n + 1, dtype=np.int32)
    spf[1] = 1
    for i in range(2, int(n ** .5) + 1):
        if spf[i] == 0:
            blk = spf[i * i::i]; blk[blk == 0] = i
    idx = np.nonzero(spf == 0)[0]; spf[idx] = idx
    return spf
def run(p, q, Z):
    CAP = q * (Z + 1) * p + p
    spf = spf_table(CAP)
    def allowed_divs(t):
        fs = {}
        x = t
        while x > 1:
            r = int(spf[x]); fs[r] = fs.get(r, 0) + 1; x //= r
        ds = [1]
        for r, e in fs.items(): ds = [d * r ** i for d in ds for i in range(e + 1)]
        return [d for d in ds if d >= 2 and (d <= Z or (d % q == 0 and d // q <= Z))]
    depth = np.full(CAP + 1, -1, dtype=np.int8)
    par_u = np.zeros(CAP + 1, dtype=np.int64); par_d = np.zeros(CAP + 1, dtype=np.int32)
    dq = deque(range(1, p)); depth[1:p] = 0
    while dq:
        u = dq.popleft()
        for d in allowed_divs(u):
            if d * p <= u: continue
            v = d * p - u
            if v <= CAP and v % p and depth[v] < 0:
                depth[v] = depth[u] + 1; par_u[v] = u; par_d[v] = d; dq.append(v)
    pair = lambda m: min(m, p - m)
    mult = defaultdict(int); cov = 0; tot = 0; lens = []
    for n in range(1, p):
        t = q * n; tot += 1
        if t < p: cov += 1; continue
        if depth[t] < 0: continue
        cov += 1; inst = set(); v = t
        while depth[v] > 0:
            u = int(par_u[v]); d = int(par_d[v])
            inst.add(pair(u // d))
            if u % p: inst.add(pair(u % p))
            v = u
        lens.append(int(depth[t]))
        for m in inst: mult[m] += 1
    vals = list(mult.values()) or [0]
    return dict(p=p, q=q, Z=Z, u_q=round(1 - cov / tot, 5), max_path=max(lens) if lens else 0,
                mean_path=round(sum(lens) / max(len(lens), 1), 3), M_q=max(vals), q_squared=q * q)
if __name__ == "__main__":
    out = []
    for p in (1009, 2003):
        for q in (5, 7, 11, 13, 17, 23, 31, 41):
            for zf in (2, 3):
                t0 = time.time(); r = run(p, q, zf * q); r["sec"] = round(time.time() - t0, 1)
                print(json.dumps(r), flush=True); out.append(r)
    json.dump(out, open("m2_e1_results.json", "w"), indent=1)
