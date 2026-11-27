"""M2-E19（预注册，DESIGN 第 17 节）：给定小模数余数（y mod q·L，L = lcm(1..⌈q^η⌉)，从而 j0、H_s、候选集合都固定）后，
候选之间"是否在 2 缩放终点簇中"的事件是否近似独立。y 的奇偶也被 L 固定（L 为偶数），所以终点一侧固定。"""
import sys, json, math, random, time
from collections import defaultdict, deque
from m2_e4_random_model import bits_needed
from m2_e18_candidates import bfs
def run(q, eta, NC, R, seed):
    Z = round(q ** (1 + eta)); B = bits_needed(Z, q); rng = random.Random(seed); t0 = time.time()
    L = 1
    for k in range(2, math.ceil(q ** eta) + 1): L = L * k // math.gcd(L, k)
    M = q * L
    obs = pred = 0.0; used = 0
    for _ in range(NC):
        c0 = rng.randrange(M)
        j0 = c0 % q
        if j0 in (0, q - 1): continue
        succ = defaultdict(int); allfail = 0; cand_ref = None; nsamp = 0
        for _ in range(R):
            y = c0 + M * rng.getrandbits(B)
            adj = defaultdict(list); adj[0].append((-1, 1)); adj[-1].append((0, 1))
            for d in range(2, Z + 1):
                e = y % d; adj[e].append((e - d, d)); adj[e - d].append((e, d))
            Rc = bfs(j0, adj, lambda d: d % q == 0)
            start, r = (0, 0) if y % 2 == 0 else (-1, 1)
            T = bfs(start, adj, lambda d: d % 2 == 0)
            cands = sorted(z for z in Rc if z % 2 == r)
            if cand_ref is None: cand_ref = cands
            ok = [z in T for z in cands]
            for z, o in zip(cands, ok): succ[z] += o
            allfail += (not any(ok)); nsamp += 1
        if not cand_ref: continue
        used += 1
        p = {z: succ[z] / nsamp for z in cand_ref}
        prod = 1.0
        for z in cand_ref: prod *= (1 - p[z])
        obs += allfail / nsamp; pred += prod
    return dict(q=q, eta=eta, Z=Z, M=M, classes=used, R=R, mean_allfail_obs=round(obs / used, 4),
                mean_allfail_indep_pred=round(pred / used, 4), sec=round(time.time() - t0, 1))
if __name__ == "__main__":
    q, NC, R, seed = int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); eta = float(sys.argv[2])
    print(json.dumps(run(q, eta, NC, R, seed)), flush=True)
