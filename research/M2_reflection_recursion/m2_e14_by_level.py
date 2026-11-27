"""M2-E14（预注册，DESIGN 第 12 节）：逃离顶点的成功率按落点层级 t = log(|x|+1)/log Z 分箱。
成功 = 在删去 R 簇后的图中能连到 0 或 −1（同 M2-E13）。"""
import sys, json, math, random, time
from collections import defaultdict, deque
from m2_e4_random_model import bits_needed
BINS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.01]
def run(q, eta, S, seed):
    Z = round(q ** (1 + eta)); B = bits_needed(Z, q); rng = random.Random(seed); t0 = time.time()
    tot = [0] * (len(BINS) - 1); ok = [0] * (len(BINS) - 1)
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
        for x in E:
            t = math.log(abs(x) + 1) / math.log(Z)
            k = next(j for j in range(len(BINS) - 1) if t < BINS[j + 1])
            tot[k] += 1; ok[k] += (x in reach)
    return dict(q=q, eta=eta, Z=Z, S=S, bins=BINS, n=tot,
                success=[round(o / t, 4) if t else None for o, t in zip(ok, tot)], sec=round(time.time() - t0, 1))
if __name__ == "__main__":
    q, S, seed = int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4]); eta = float(sys.argv[2])
    print(json.dumps(run(q, eta, S, seed)), flush=True)
