"""M3-E20c（探索性）：按斜率加权的覆盖 Σ_{记录 ∋ φ} 1/|s|，以及斜率为 ±1 的记录的最大覆盖。"""
import sys, json
from fractions import Fraction as Fr
from m3_e20_Fq_exact import build
for q in map(int, sys.argv[1].split(",")):
    recs = build(q); ev = []; ev1 = []
    for r0, r1, s, c in recs:
        p0, p1 = sorted((s * r0 + c, s * r1 + c))
        segs = []
        if p1 <= Fr(1, 2): segs.append((p0, p1))
        elif p0 >= Fr(1, 2): segs.append((1 - p1, 1 - p0))
        else: segs += [(p0, Fr(1, 2)), (1 - p1, Fr(1, 2))]
        for a0, a1 in segs:
            if a1 > a0:
                ev += [(a0, Fr(1, abs(s))), (a1, -Fr(1, abs(s)))]
                if abs(s) == 1: ev1 += [(a0, 1), (a1, -1)]
    def sweep(E):
        E.sort(key=lambda e: (e[0], e[1])); cur = best = 0
        for x, d in E:
            cur += d; best = max(best, cur)
        return best
    from collections import Counter
    sl = Counter(abs(s) for *_, s, c in recs)
    print(json.dumps(dict(q=q, records=len(recs), weighted_max=round(float(sweep(ev)), 2), slope1_max=sweep(ev1),
                          slopes_small=sorted(sl.items())[:6])), flush=True)
