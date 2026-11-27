"""M2-E15（预注册，DESIGN 第 13 节）：起点的 R 簇（q 缩放）与终点的 g 缩放簇是否直接相交。
R = 从 j0 只走 q | d 的边可达的顶点；T_g = 从 0 只走 g | d 的边可达的顶点（g | y）。
meet：存在 g（2 ≤ g ≤ q^η，g | y，q ∤ g）使 R ∩ T_g ≠ ∅。"""
import sys, json, math, random, time
from collections import defaultdict, deque
from m2_e4_random_model import bits_needed
def bfs(start, adj, ok):
    seen = {start}; dq = deque([start])
    while dq:
        v = dq.popleft()
        for w, d in adj[v]:
            if ok(d) and w not in seen: seen.add(w); dq.append(w)
    return seen
def run(q, eta, S, seed):
    Z = round(q ** (1 + eta)); B = bits_needed(Z, q); rng = random.Random(seed); t0 = time.time()
    Gm = max(2, int(q ** eta)); n = meet = cov = meet_and_cov = 0; nmeet_vertices = []
    for _ in range(S):
        y = rng.getrandbits(B); j0 = y % q
        if j0 in (0, q - 1): continue
        n += 1
        adj = defaultdict(list); adj[0].append((-1, 1)); adj[-1].append((0, 1))
        for d in range(2, Z + 1):
            e = y % d; adj[e].append((e - d, d)); adj[e - d].append((e, d))
        R = bfs(j0, adj, lambda d: d % q == 0)
        hits = set()
        for g in range(2, Gm + 1):
            if y % g or g % q == 0: continue
            T = bfs(0, adj, lambda d, g=g: d % g == 0)
            hits |= (R & T)
        m = bool(hits); meet += m; nmeet_vertices.append(len(hits))
        # 覆盖（全图）
        c = j0 in bfs(0, adj, lambda d: True) or j0 in bfs(-1, adj, lambda d: True)
        cov += c; meet_and_cov += (m and c)
    nmeet_vertices.sort()
    return dict(q=q, eta=eta, Z=Z, Gmax=Gm, n=n, P_meet=round(meet / n, 4), P_cov=round(cov / n, 4),
                P_meet_given_cov=round(meet_and_cov / max(1, cov), 4),
                meet_vertices_median=nmeet_vertices[len(nmeet_vertices) // 2], sec=round(time.time() - t0, 1))
if __name__ == "__main__":
    q, S, seed = int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4]); eta = float(sys.argv[2])
    print(json.dumps(run(q, eta, S, seed)), flush=True)
