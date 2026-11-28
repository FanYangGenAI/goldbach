"""M3-E39（预注册见 DESIGN.md 第 27 节）：Thue 树节点的入度分布。"""
import sys, json, math
from collections import Counter
def thue(a, x, N):
    for v in range(1, a):
        u = (x * v) % N
        if u * a < N or (N - u) * a < N: return v
    raise RuntimeError
def indeg(b, z, q, N, memo):
    key = (b, z)
    if key in memo: return memo[key]
    d = 0
    for a in range(b + 1, q + 1):
        if a * z >= N and thue(a, z, N) == b: d += 1                      # A 型父节点 (a, z)
        w = (pow(a, N - 2, N) * z) % N
        if a * w >= N and thue(a, w, N) == b: d += 1                      # B 型父节点 (a, a⁻¹z)
    memo[key] = d; return d
def run(N, q, sample_step=1):
    D = math.log(q) ** 3; memo = {}; hist = Counter(); bad_per_tree = []; tot_nodes = 0
    for n in range(1, N, sample_step):
        if q * n < N: continue
        stack = [(q, n)]; bad = 0
        while stack:
            a, x = stack.pop()
            if a == 1 or a * x < N: continue
            tot_nodes += 1
            if a < q:
                d = indeg(a, x, q, N, memo); hist[min(d, 60)] += 1
                if d > D: bad += 1
            v = thue(a, x, N)
            stack.append((v, x)); stack.append((v, (a * x) % N))
        bad_per_tree.append(bad)
    nonroot = sum(hist.values())
    frac = sum(c for d, c in hist.items() if d > D) / nonroot
    return dict(N=N, q=q, D=round(D, 1), nonroot_nodes=nonroot, frac_deg_gt_D=round(frac, 5),
                mean_bad_per_tree=round(sum(bad_per_tree) / len(bad_per_tree), 5),
                frac_trees_with_bad=round(sum(1 for b in bad_per_tree if b) / len(bad_per_tree), 5),
                deg_mean=round(sum(d * c for d, c in hist.items()) / nonroot, 3), deg_max_bucket=max(hist), hist={d: hist[d] for d in sorted(hist)})
if __name__ == "__main__":
    N = int(sys.argv[1]); step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    for q in map(int, sys.argv[2].split(",")):
        print(json.dumps(run(N, q, step)), flush=True)
