"""M2-E12b（探索性，非预注册）：巨块之外的非孤立顶点按层级分布在哪里（以 log|v|/log Z 分箱）。"""
import sys, json, math, random
from m2_e4_random_model import bits_needed
q, eta, S, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
Z = round(q ** (1 + eta)); B = bits_needed(Z, q); rng = random.Random(seed); off = Z
bins = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.01]
out_cnt = [0] * (len(bins) - 1); all_cnt = [0] * (len(bins) - 1)
for _ in range(S):
    y = rng.getrandbits(B); par = list(range(2 * Z + 1)); deg = [0] * (2 * Z + 1)
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for d in range(1, Z + 1):
        e = y % d; a, b = e + off, e - d + off
        ra, rb = find(a), find(b)
        if ra != rb: par[ra] = rb
        deg[a] += 1; deg[b] += 1
    g0 = find(off)
    for i in range(2 * Z + 1):
        if not deg[i]: continue
        v = i - off; lv = math.log(abs(v) + 1) / math.log(Z)
        k = next(j for j in range(len(bins) - 1) if lv < bins[j + 1])
        all_cnt[k] += 1; out_cnt[k] += (find(i) != g0)
print(json.dumps(dict(q=q, eta=eta, S=S, bins=bins, nonisolated=all_cnt,
      frac_outside_giant=[round(o / a, 4) if a else None for o, a in zip(out_cnt, all_cnt)])))
