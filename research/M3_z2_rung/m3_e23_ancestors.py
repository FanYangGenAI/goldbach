"""M3-E23（探索性）：Thue 树中各实例 (a, x) 的祖先计数（含重数），以及行负载的来源分解。"""
import sys, json
from collections import defaultdict, Counter
def run(N, q):
    cnt = defaultdict(int)
    for n in range(1, N):
        if q * n >= N: cnt[(q, n)] = 1
    rowload = Counter(); by_level = {}
    for a in range(q, 1, -1):
        items = [(x, c) for (aa, x), c in cnt.items() if aa == a]
        mx = 0
        for x, c in items:
            mx = max(mx, c)
            if a * x < N: continue
            for v in range(1, a):
                u = (x * v) % N
                if u * a < N: eps, y = 1, u; break
                if (N - u) * a < N: eps, y = -1, N - u; break
            if v > 1:
                cnt[(v, x)] += c; cnt[(v, (a * x) % N)] += c
            if eps == -1:
                for m in (y, a * y): rowload[min(m % N, N - m % N)] += c
        if items: by_level[a] = dict(instances=len(items), max_anc=mx)
    return dict(N=N, q=q, rowload_max=max(rowload.values()), top=rowload.most_common(3),
                max_anc_by_level={a: d["max_anc"] for a, d in sorted(by_level.items())})
if __name__ == "__main__":
    for N in map(int, sys.argv[1].split(",")):
        for q in map(int, sys.argv[2].split(",")):
            print(json.dumps(run(N, q)), flush=True)
