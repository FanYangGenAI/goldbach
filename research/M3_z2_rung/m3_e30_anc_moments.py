"""M3-E30（预注册见 DESIGN.md 第 24 节）：Thue 树（最小 v、不分裂）中祖先数的矩。"""
import sys, json
from collections import defaultdict
def run(N, q):
    cnt = defaultdict(int)
    for n in range(1, N):
        if q * n >= N: cnt[(q, n)] = 1
    A = {1: 0, 2: 0, 3: 0}; inst = 0; mx = 0
    for a in range(q, 1, -1):
        items = [(x, c) for (aa, x), c in cnt.items() if aa == a]
        for x, c in items:
            inst += 1; mx = max(mx, c)
            for k in A: A[k] += c ** k
            if a * x < N: continue
            for v in range(1, a):
                u = (x * v) % N
                if u * a < N or (N - u) * a < N: break
            if v > 1:
                cnt[(v, x)] += c; cnt[(v, (a * x) % N)] += c
        for x, c in items: del cnt[(a, x)]
    return dict(N=N, q=q, instances_over_N=round(inst / N, 3), A1_over_N=round(A[1] / N, 3), A2_over_N=round(A[2] / N, 3), A3_over_N=round(A[3] / N, 3), max_anc=mx)
if __name__ == "__main__":
    for N in map(int, sys.argv[1].split(",")):
        for q in map(int, sys.argv[2].split(",")):
            print(json.dumps(run(N, q)), flush=True)
