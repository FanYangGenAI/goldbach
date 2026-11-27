"""M2-E12（预注册，DESIGN 第 10 节）：整张层级图的连通块结构（引理 U 的数值前提）。
plain 模型，Z = round(q^{1+η})，所有乘数 d ≤ Z。记录：大小 ≥ 30 的块数；最大块是否含 0；
第二大块的大小；非孤立顶点中最大块所占比例。"""
import sys, json, math, random, time
from m2_e4_random_model import bits_needed
def run(q, eta, S, seed):
    Z = round(q ** (1 + eta)); B = bits_needed(Z, q); rng = random.Random(seed); t0 = time.time()
    off = Z  # 顶点 v ∈ [−Z, Z) ↦ v + Z
    one_big = 0; giant_has0 = 0; second = []; frac = []
    for _ in range(S):
        y = rng.getrandbits(B)
        par = list(range(2 * Z + 1)); deg = [0] * (2 * Z + 1)
        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: par[ra] = rb
        for d in range(1, Z + 1):
            e = y % d; a, b = e + off, e - d + off
            union(a, b); deg[a] += 1; deg[b] += 1
        size = {}
        for v in range(2 * Z + 1):
            if deg[v]: r = find(v); size[r] = size.get(r, 0) + 1
        sz = sorted(size.values(), reverse=True)
        big = [s for s in sz if s >= 30]
        one_big += (len(big) == 1)
        giant_root = max(size, key=size.get)
        giant_has0 += (find(off) == giant_root)
        second.append(sz[1] if len(sz) > 1 else 0)
        frac.append(sz[0] / sum(sz))
    second.sort()
    return dict(q=q, eta=eta, Z=Z, S=S, frac_one_big=round(one_big / S, 4), frac_giant_has0=round(giant_has0 / S, 4),
                second_median=second[len(second) // 2], second_max=second[-1],
                giant_frac_of_nonisolated=round(sum(frac) / S, 4), sec=round(time.time() - t0, 1))
if __name__ == "__main__":
    q, S, seed = int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4]); eta = float(sys.argv[2])
    print(json.dumps(run(q, eta, S, seed)), flush=True)
