"""M2-E19b（探索性，非预注册）：候选按 2q 倍数乘数的边分块（同一块内成员同进同出），
检验"全部失败"是否与"按块独立"的预测一致。"""
import sys, json, math, random
from collections import defaultdict, deque
from m2_e4_random_model import bits_needed
from m2_e18_candidates import bfs
q, eta, S, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
Z = round(q ** (1 + eta)); B = bits_needed(Z, q); rng = random.Random(seed)
nb = []; tot_b = succ_b = 0; byb = defaultdict(lambda: [0, 0]); constant_blocks = 0; blocks_seen = 0
for _ in range(S):
    y = rng.getrandbits(B); j0 = y % q
    if j0 in (0, q - 1): continue
    adj = defaultdict(list); adj[0].append((-1, 1)); adj[-1].append((0, 1))
    for d in range(2, Z + 1):
        e = y % d; adj[e].append((e - d, d)); adj[e - d].append((e, d))
    Rc = bfs(j0, adj, lambda d: d % q == 0)
    start, r = (0, 0) if y % 2 == 0 else (-1, 1)
    T = bfs(start, adj, lambda d: d % 2 == 0)
    cands = {z for z in Rc if z % 2 == r}
    if not cands: continue
    left = set(cands); blocks = []
    while left:
        z = left.pop(); comp = bfs(z, adj, lambda d: d % (2 * q) == 0) & cands
        left -= comp; blocks.append(comp)
    res = []
    for b in blocks:
        ins = [z in T for z in b]; blocks_seen += 1; constant_blocks += (all(ins) or not any(ins)); res.append(ins[0])
    k = len(blocks); nb.append(k); tot_b += k; succ_b += sum(res)
    kk = min(k, 5); byb[kk][0] += 1; byb[kk][1] += (not any(res))
p = succ_b / tot_b
nb.sort()
print(json.dumps(dict(q=q, eta=eta, blocks_median=nb[len(nb) // 2], frac_blocks_constant=round(constant_blocks / blocks_seen, 4),
      per_block_success=round(p, 4),
      allfail_by_blocks={k: dict(n=v[0], obs=round(v[1] / v[0], 4), indep=round((1 - p) ** k, 4)) for k, v in sorted(byb.items())})))
