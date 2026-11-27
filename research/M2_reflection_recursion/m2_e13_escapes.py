"""M2-E13（预注册，DESIGN 第 11 节）：逃离顶点的个数与"各自能否连到 0"（引理 S 的数值前提）。
R 簇 = 从 j0 只走 q | d 的边可达的顶点；E = 经 q ∤ d 的边与 R 簇相邻、且不在 R 簇中的顶点。
在删去 R 簇之后的图中，从 {0, −1} 做 BFS；逃离顶点 x"成功"当且仅当 x 被到达。"""
import sys, json, math, random, time
from collections import defaultdict, deque, Counter
from m2_e4_random_model import bits_needed
def run(q, eta, S, seed):
    Z = round(q ** (1 + eta)); B = bits_needed(Z, q); rng = random.Random(seed); t0 = time.time()
    sizes = []; per_succ = 0; per_tot = 0; byk = defaultdict(lambda: [0, 0])  # k -> [样本数, 全失败数]
    for _ in range(S):
        y = rng.getrandbits(B); j0 = y % q
        if j0 in (0, q - 1): continue
        adj = defaultdict(list); adj[0].append((-1, 1)); adj[-1].append((0, 1))
        for d in range(2, Z + 1):
            e = y % d; adj[e].append((e - d, d)); adj[e - d].append((e, d))
        R = {j0}; dq = deque([j0])
        while dq:
            v = dq.popleft()
            for w, d in adj[v]:
                if d % q == 0 and w not in R: R.add(w); dq.append(w)
        E = {w for v in R for w, d in adj[v] if d % q and w not in R}
        reach = {0, -1}; dq = deque([0, -1])
        while dq:
            v = dq.popleft()
            for w, d in adj[v]:
                if w not in reach and w not in R: reach.add(w); dq.append(w)
        succ = [x in reach for x in E]
        k = len(E); sizes.append(k)
        per_succ += sum(succ); per_tot += k
        kk = min(k, 6); byk[kk][0] += 1; byk[kk][1] += (not any(succ))
    sizes.sort(); f = 1 - per_succ / max(1, per_tot)
    return dict(q=q, eta=eta, Z=Z, n=len(sizes), E_median=sizes[len(sizes) // 2], E_mean=round(sum(sizes) / len(sizes), 2),
                per_escape_fail=round(f, 4),
                allfail_by_k={k: dict(n=v[0], obs=round(v[1] / v[0], 4), indep_pred=round(f ** k, 4)) for k, v in sorted(byk.items())},
                sec=round(time.time() - t0, 1))
if __name__ == "__main__":
    q, S, seed = int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4]); eta = float(sys.argv[2])
    print(json.dumps(run(q, eta, S, seed)), flush=True)
