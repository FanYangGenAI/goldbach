"""M2-E18b（探索性，非预注册）：全部候选失败时，终点 2 缩放簇 T_2 的大小，以及 y/2（或 (y+1)/2）的最小素因子。"""
import sys, json, math, random
from collections import defaultdict, deque
from m2_e4_random_model import bits_needed
from m2_e18_candidates import bfs
q, eta, S, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
Z = round(q ** (1 + eta)); B = bits_needed(Z, q); rng = random.Random(seed)
fail_T = []; ok_T = []; fail_spf = []; ok_spf = []
def spf(m, lim=200):
    for r in range(2, lim):
        if m % r == 0: return r
    return lim
for _ in range(S):
    y = rng.getrandbits(B); j0 = y % q
    if j0 in (0, q - 1): continue
    adj = defaultdict(list); adj[0].append((-1, 1)); adj[-1].append((0, 1))
    for d in range(2, Z + 1):
        e = y % d; adj[e].append((e - d, d)); adj[e - d].append((e, d))
    R = bfs(j0, adj, lambda d: d % q == 0)
    if y % 2 == 0: start, r, half = 0, 0, y // 2
    else: start, r, half = -1, 1, (y + 1) // 2
    T = bfs(start, adj, lambda d: d % 2 == 0)
    c = [z for z in R if z % 2 == r]
    if not c: continue
    if any(z in T for z in c): ok_T.append(len(T)); ok_spf.append(spf(half))
    else: fail_T.append(len(T)); fail_spf.append(spf(half))
med = lambda a: sorted(a)[len(a) // 2] if a else None
print(json.dumps(dict(q=q, eta=eta, n_fail=len(fail_T), n_ok=len(ok_T), T2_size_median_fail=med(fail_T), T2_size_median_ok=med(ok_T),
      half_spf_median_fail=med(fail_spf), half_spf_median_ok=med(ok_spf),
      frac_fail_with_T2_le_10=round(sum(t <= 10 for t in fail_T) / max(1, len(fail_T)), 3))))
