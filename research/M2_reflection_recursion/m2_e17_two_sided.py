"""M2-E17（预注册，DESIGN 第 15 节）：起点 R 簇与终点两侧的缩放簇相交。
终点一侧：T_g（从 0 出发、只走 g | d 的边，g | y）与 T′_g（从 −1 出发、只走 g | d 的边，g | y + 1）。
g 跑遍 [2, G]，G = max(2, ⌊q^{η/2}⌋)，q ∤ g。y 与 y + 1 必有一个是偶数，所以 g = 2 总可用。"""
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
    G = max(2, int(q ** (eta / 2))); n = meet = meet2 = 0
    for _ in range(S):
        y = rng.getrandbits(B); j0 = y % q
        if j0 in (0, q - 1): continue
        n += 1
        adj = defaultdict(list); adj[0].append((-1, 1)); adj[-1].append((0, 1))
        for d in range(2, Z + 1):
            e = y % d; adj[e].append((e - d, d)); adj[e - d].append((e, d))
        R = bfs(j0, adj, lambda d: d % q == 0)
        hit = hit2 = False
        for g in range(2, G + 1):
            if g % q == 0: continue
            for start, ok in ((0, y % g == 0), (-1, (y + 1) % g == 0)):
                if not ok: continue
                T = bfs(start, adj, lambda d, g=g: d % g == 0)
                if R & T:
                    hit = True
                    if g == 2: hit2 = True
        meet += hit; meet2 += hit2
    return dict(q=q, eta=eta, Z=Z, G=G, n=n, P_meet=round(meet / n, 4), P_meet_g2=round(meet2 / n, 4), sec=round(time.time() - t0, 1))
if __name__ == "__main__":
    q, S, seed = int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4]); eta = float(sys.argv[2])
    print(json.dumps(run(q, eta, S, seed)), flush=True)
