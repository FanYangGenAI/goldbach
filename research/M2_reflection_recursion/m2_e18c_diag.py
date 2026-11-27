"""M2-E18c（探索性，非预注册）：全部候选失败的样本中，y mod 4、j0 的奇偶、候选数、候选的一半 z/2 在 H_{y/2} 中的层级分布。"""
import sys, json, math, random
from collections import defaultdict, Counter
from m2_e4_random_model import bits_needed
from m2_e18_candidates import bfs
q, eta, S, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
Z = round(q ** (1 + eta)); B = bits_needed(Z, q); rng = random.Random(seed)
F = Counter(); O = Counter(); fK = []; oK = []; f_lv = []; o_lv = []; f_iso = 0; f_c = 0
for _ in range(S):
    y = rng.getrandbits(B); j0 = y % q
    if j0 in (0, q - 1): continue
    adj = defaultdict(list); adj[0].append((-1, 1)); adj[-1].append((0, 1))
    for d in range(2, Z + 1):
        e = y % d; adj[e].append((e - d, d)); adj[e - d].append((e, d))
    R = bfs(j0, adj, lambda d: d % q == 0)
    start, r = (0, 0) if y % 2 == 0 else (-1, 1)
    T = bfs(start, adj, lambda d: d % 2 == 0)
    c = [z for z in R if z % 2 == r]
    if not c: continue
    key = (y % 4, j0 % 2)
    lv = [round(math.log(abs(z) + 1) / math.log(Z), 2) for z in c]
    if any(z in T for z in c): O[key] += 1; oK.append(len(c)); o_lv += lv
    else:
        F[key] += 1; fK.append(len(c)); f_lv += lv
        # 失败候选在全图中是否有任何非 2 倍数乘数之外的边（是否孤立于 2 缩放结构）
        for z in c:
            f_c += 1; f_iso += all(d % 2 for _, d in adj[z] if d % q)
med = lambda a: sorted(a)[len(a) // 2] if a else None
print(json.dumps(dict(fail_by_ymod4_j0par={str(k): v for k, v in sorted(F.items())}, ok_by_ymod4_j0par={str(k): v for k, v in sorted(O.items())},
      K_median_fail=med(fK), K_median_ok=med(oK), cand_level_median_fail=med(f_lv), cand_level_median_ok=med(o_lv),
      frac_fail_cands_with_no_even_nonq_edge=round(f_iso / max(1, f_c), 3))))
