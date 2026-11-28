"""M3-E33（预注册见 DESIGN.md 第 26 节）：独立均匀点模型 vs 实际 Thue 树。"""
import sys, json, random
from collections import Counter
def vfun(a, th):
    for v in range(1, a):
        f = (v * th) % 1.0
        if f < 1 / a: return v, 1
        if f > 1 - 1 / a: return v, -1
    return a - 1, 1
def model_tree(q, rng):
    # 返回（非平凡节点数，ε=-1 节点数，深度）
    def rec(a, th):
        if a * th < 1 or a == 1: return 0, 0, 0
        v, e = vfun(a, th)
        s1, n1, d1 = rec(v, rng.random()); s2, n2, d2 = rec(v, rng.random())
        return 1 + s1 + s2, (e == -1) + n1 + n2, 1 + max(d1, d2)
    return rec(q, rng.uniform(1 / q, 1))
def actual_tree(N, q, n):
    def rec(a, x):
        if a * x < N or a == 1: return 0, 0, 0
        for v in range(1, a):
            u = (x * v) % N
            if u * a < N: e = 1; break
            if (N - u) * a < N: e = -1; break
        s1, n1, d1 = rec(v, x); s2, n2, d2 = rec(v, (a * x) % N)
        return 1 + s1 + s2, (e == -1) + n1 + n2, 1 + max(d1, d2)
    return rec(q, n)
def summarize(samples):
    S = [s for s, _, _ in samples]; G = [g for _, g, _ in samples]; D = Counter(d for _, _, d in samples)
    tot = len(samples); tail = {}; acc = 0
    for k in sorted(D, reverse=True): acc += D[k]; tail[k] = acc / tot
    return dict(size_mean=round(sum(S) / tot, 3), neg_mean=round(sum(G) / tot, 3), tail={k: round(tail[k], 6) for k in sorted(tail)})
if __name__ == "__main__":
    N = 262139; rng = random.Random(5)
    for q in map(int, sys.argv[1].split(",")):
        act = summarize([actual_tree(N, q, n) for n in range(1, N) if q * n >= N])
        mod = summarize([model_tree(q, rng) for _ in range(200000)])
        print(json.dumps(dict(q=q, actual=act, model=mod)), flush=True)
