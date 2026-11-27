"""M2-E18（预注册，DESIGN 第 16 节）：2 缩放相交的候选数 K 与"全部失败"是否近似独立。
候选 = R 簇中与终点 2 缩放簇同余类（偶数，或 −1 一侧时 ≡ −1 mod 2）的顶点；成功 = 该候选在 T_2（或 T′_2）中。"""
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
    tot = succ = 0; byk = defaultdict(lambda: [0, 0]); n = 0
    for _ in range(S):
        y = rng.getrandbits(B); j0 = y % q
        if j0 in (0, q - 1): continue
        n += 1
        adj = defaultdict(list); adj[0].append((-1, 1)); adj[-1].append((0, 1))
        for d in range(2, Z + 1):
            e = y % d; adj[e].append((e - d, d)); adj[e - d].append((e, d))
        R = bfs(j0, adj, lambda d: d % q == 0)
        start, r = (0, 0) if y % 2 == 0 else (-1, 1)
        T = bfs(start, adj, lambda d: d % 2 == 0)
        cands = [z for z in R if z % 2 == r]
        ok = [z in T for z in cands]
        k = len(cands); tot += k; succ += sum(ok)
        kk = min(k, 8); byk[kk][0] += 1; byk[kk][1] += (not any(ok))
    p = succ / max(1, tot)
    return dict(q=q, eta=eta, Z=Z, n=n, per_candidate_success=round(p, 4),
                allfail_by_K={k: dict(n=v[0], obs=round(v[1] / v[0], 4), indep=round((1 - p) ** k, 4)) for k, v in sorted(byk.items())},
                sec=round(time.time() - t0, 1))
if __name__ == "__main__":
    q, S, seed = int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4]); eta = float(sys.argv[2])
    print(json.dumps(run(q, eta, S, seed)), flush=True)
