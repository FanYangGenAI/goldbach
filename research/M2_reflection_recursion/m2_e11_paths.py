"""M2-E11（预注册，DESIGN 第 9 节）：逃离之后的路径结构（描述性）。
plain 模型，Z = round(q^{1+η})。对 j0 ∉ {0, q−1} 且被覆盖的样本，取 BFS 最短路 j0 → {0, −1}，记录：
  esc_step：第一个 ≢ j0 (mod q) 的顶点在路径中的位置；esc_prime：该步乘数是否为素数；
  after：逃离后到终点的步数；small_hit：逃离后、终点之前是否经过 |v| ≤ √q 的顶点；
另外记录 0 的连通块对小顶点 [−V, V)（V = ⌊√q⌋）的覆盖比例 core_frac。"""
import sys, json, math, random, time
from collections import defaultdict, deque
from m2_e4_random_model import bits_needed
def isprime(n):
    if n < 2: return False
    i = 2
    while i * i <= n:
        if n % i == 0: return False
        i += 1
    return True
def run(q, eta, S, seed):
    Z = round(q ** (1 + eta)); B = bits_needed(Z, q); rng = random.Random(seed); V = math.isqrt(q)
    PR = [isprime(d) for d in range(Z + 1)]
    recs = []; cores = []; t0 = time.time()
    for _ in range(S):
        y = rng.getrandbits(B); j0 = y % q
        if j0 in (0, q - 1): continue
        adj = defaultdict(list); adj[0].append((-1, 1)); adj[-1].append((0, 1))
        for d in range(2, Z + 1):
            e = y % d; adj[e].append((e - d, d)); adj[e - d].append((e, d))
        par = {0: None, -1: None}; dq = deque([0, -1])
        while dq:
            v = dq.popleft()
            for w, d in adj[v]:
                if w not in par: par[w] = (v, d); dq.append(w)
        cores.append(sum(1 for v in range(-V, V) if v in par) / (2 * V))
        if j0 not in par: continue
        path = [j0]; mods = []
        while par[path[-1]] is not None:
            v, d = par[path[-1]]; path.append(v); mods.append(d)
        # path: j0 → … → 终点；mods[i] 是 path[i] 与 path[i+1] 之间的乘数
        k = next(i for i, v in enumerate(path) if (v - j0) % q)
        esc_prime = PR[mods[k - 1]]
        after = len(path) - 1 - k
        small = any(abs(v) <= V for v in path[k:-1])
        recs.append((k, esc_prime, after, small, len(path) - 1))
    n = len(recs)
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
    return dict(q=q, eta=eta, Z=Z, S_covered=n, sec=round(time.time() - t0, 1),
                esc_step_median=med([r[0] for r in recs]),
                esc_prime_frac=round(sum(r[1] for r in recs) / max(1, n), 3),
                after_median=med([r[2] for r in recs]), after_mean=round(sum(r[2] for r in recs) / max(1, n), 3),
                small_hit_frac=round(sum(r[3] for r in recs) / max(1, n), 3),
                len_mean=round(sum(r[4] for r in recs) / max(1, n), 3),
                core_frac_mean=round(sum(cores) / max(1, len(cores)), 4), core_frac_min=round(min(cores), 4))
if __name__ == "__main__":
    q, S, seed = int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4]); eta = float(sys.argv[2])
    print(json.dumps(run(q, eta, S, seed)), flush=True)
