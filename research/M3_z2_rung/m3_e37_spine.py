"""M3-E37：检验"脊柱 + 毛毛虫"结构：A 型链（同一点、乘数沿渐近分母下降）上节点的 B 型子节点要么平凡，要么 v = 1（只用两行即终止）。
同时统计脊柱长度（从根出发只走 B 型子节点的链）的分布。"""
import sys, json
from collections import Counter
def thue(a, x, N):
    for v in range(1, a):
        u = (x * v) % N
        if u * a < N: return v, 1
        if (N - u) * a < N: return v, -1
    raise RuntimeError
def check(N, q):
    viol = 0; checked = 0; spine = Counter(); cat_max = 0
    for n in range(1, N):
        if q * n < N: continue
        # 沿脊柱走
        a, x, L = q, n, 0
        while a > 1 and a * x >= N:
            v, e = thue(a, x, N); L += 1
            # 毛毛虫：A 型子节点 (v, x) 及其 A 链
            b, clen = v, 0
            while b > 1 and b * x >= N:
                w, e2 = thue(b, x, N); clen += 1
                # B 型子节点 (w, b*x mod N)：应当平凡，或者 thue 给出 v = 1
                z = (b * x) % N
                if w > 1 and w * z >= N:
                    checked += 1
                    w2, _ = thue(w, z, N)
                    if w2 != 1: viol += 1
                b = w
            cat_max = max(cat_max, clen)
            a, x = v, (a * x) % N
        spine[L] += 1
    tot = sum(spine.values()); tail = {}; acc = 0
    for k in sorted(spine, reverse=True): acc += spine[k]; tail[k] = round(acc / tot, 6)
    return dict(N=N, q=q, caterpillar_B_children_checked=checked, violations=viol, caterpillar_max_len=cat_max,
                spine_mean=round(sum(k * c for k, c in spine.items()) / tot, 3), spine_tail={k: tail[k] for k in sorted(tail)})
if __name__ == "__main__":
    for N in map(int, sys.argv[1].split(",")):
        for q in map(int, sys.argv[2].split(",")):
            print(json.dumps(check(N, q)), flush=True)
