"""M2-E3：层级图约化（reduction）的数值核对。

断言（证明见 RESULTS.md 第 8 节）：固定素数 p、素数 q、n ∈ [1, p)。记 j0 = ⌊qn/p⌋，
a = qn − j0·p，并对每个乘数 d（gcd(d, p) = 1）令 e_d = (−a·p^{-1}) mod d。则 M2-E1 中
"qn 有一条受限反射路径到 [1, p)"当且仅当在图
    H = { 边 {e_d, e_d − d} : d ∈ D }（顶点为整数）
中，顶点 j0 与顶点 0 连通（顶点 −1 经 d = 1 的边与 0 等价，这里把 0、−1 都当作终点）。

本脚本对每个 n 构造 H，用并查集判定连通，与 m2_e1.run 的 BFS 覆盖率逐个 n 比较。"""
import sys, json
from m2_e1 import run as bfs_run

def allowed_D(q, Z):
    D = set(range(2, Z + 1))
    D |= {q * a for a in range(1, Z + 1)}
    return sorted(D)

def covered_level_graph(p, q, Z):
    D = allowed_D(q, Z)
    inv = {d: pow(p, -1, d) if d > 1 else 0 for d in D}
    cov = []
    for n in range(1, p):
        t = q * n; j0 = t // p; a = t - j0 * p
        if j0 == 0:
            cov.append(True); continue
        par = {}
        def find(x):
            while par.get(x, x) != x:
                par[x] = par.get(par[x], par[x]); x = par[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry: par[rx] = ry
        union(0, -1)
        for d in D:
            e = (-a * inv[d]) % d
            union(e, e - d)
        cov.append(find(j0) == find(0))
    return cov

if __name__ == "__main__":
    out = []
    for p in (1009, 2003):
        for q in (13, 23, 41):
            for zf in (2, 3, 6):
                Z = zf * q
                cov = covered_level_graph(p, q, Z)
                u = 1 - sum(cov) / len(cov)
                r = bfs_run(p, q, Z)
                rec = dict(p=p, q=q, Z=Z, u_level_graph=round(u, 5), u_bfs=r["u_q"],
                           agree=round(u, 5) == r["u_q"])
                print(json.dumps(rec), flush=True); out.append(rec)
    json.dump(out, open("m2_e3_reduction_check.json", "w"), indent=1)
