"""M3-E20b：核对 F* 的覆盖函数是每一行实际负载的上界（检验理论与代码）。"""
import sys, json, bisect
from fractions import Fraction as Fr
from collections import Counter
from m3_e20_Fq_exact import build
from m3_e19c_thue_tree_nomemo import rows_of
def cov_at(recs, phi):
    c = 0
    for r0, r1, s, cc in recs:
        p0, p1 = sorted((s * r0 + cc, s * r1 + cc))
        for a0, a1 in ((p0, p1), (1 - p1, 1 - p0)):   # φ 或 1−φ 落在像中
            if a0 < phi < a1: c += 1
    return c
N, q = int(sys.argv[1]), int(sys.argv[2])
recs = build(q)
# 实际负载（不做 XOR 抵消，按出现次数计，以便与记录数对应）
occ = Counter()
for n in range(1, N):
    if q * n < N: continue
    out = Counter(); rows_of(q, n, N, out)
    for x, c in out.items():
        if c: occ[x] += 1
worst = 0; viol = 0; samples = []
for rho in sorted(occ, key=lambda r: -occ[r])[:60] + list(range(1, 30)):
    cv = cov_at(recs, Fr(rho, N))
    if occ[rho] > cv: viol += 1
    samples.append((rho, occ[rho], cv))
print(json.dumps(dict(N=N, q=q, violations=viol, top=samples[:8], small_rows=samples[60:70])))
