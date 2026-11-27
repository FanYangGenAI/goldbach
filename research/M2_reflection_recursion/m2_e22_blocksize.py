"""M2-E22（预注册，DESIGN 第 20 节）：候选块的大小与"块连到终点 2 缩放簇"的关系。
块 = 候选在 2q 倍数乘数的边下的分支（同 M2-E19b/E20）；记录每个块的顶点数与是否成功。"""
import sys, json, math, random, time
from collections import defaultdict, deque
from m2_e4_random_model import bits_needed
from m2_e18_candidates import bfs
def run(q, eta, S, seed):
    Z = round(q ** (1 + eta)); B = bits_needed(Z, q); rng = random.Random(seed); t0 = time.time()
    by = defaultdict(lambda: [0, 0])  # 大小档 -> [块数, 失败数]
    edges = [1, 2, 3, 4, 8, 16, 10**9]
    for _ in range(S):
        y = rng.getrandbits(B); j0 = y % q
        if j0 in (0, q - 1): continue
        adj = defaultdict(list); adj[0].append((-1, 1)); adj[-1].append((0, 1))
        for d in range(2, Z + 1):
            e = y % d; adj[e].append((e - d, d)); adj[e - d].append((e, d))
        Rc = bfs(j0, adj, lambda d: d % q == 0)
        start, r = (0, 0) if y % 2 == 0 else (-1, 1)
        T = bfs(start, adj, lambda d: d % 2 == 0)
        cands = {z for z in Rc if z % 2 == r}; left = set(cands)
        while left:
            z = left.pop(); comp = bfs(z, adj, lambda d: d % (2 * q) == 0) & cands; left -= comp
            k = len(comp); b = next(i for i in range(len(edges) - 1) if k < edges[i + 1])
            by[b][0] += 1; by[b][1] += (z not in T)
    labels = ["1", "2", "3", "4-7", "8-15", "16+"]
    return dict(q=q, eta=eta, Z=Z, S=S, by_size={labels[b]: dict(blocks=v[0], fail=round(v[1] / v[0], 4)) for b, v in sorted(by.items())},
                sec=round(time.time() - t0, 1))
if __name__ == "__main__":
    q, S, seed = int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4]); eta = float(sys.argv[2])
    print(json.dumps(run(q, eta, S, seed)), flush=True)
