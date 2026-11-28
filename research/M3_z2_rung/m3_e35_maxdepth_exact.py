"""M3-E35（探索性）：连续极限下 Thue 树的精确最大深度 D(q) = sup_θ depth(q, θ)，以及精确的最大树大小。
对一切素数 N > q，所有整数点都落在片段内部，所以实际树的深度和大小都 ≤ 这两个量（与 N 无关）。"""
import sys, json
from fractions import Fraction as Fr
from math import floor
from m3_e20_Fq_exact import pieces
def explore(q):
    # 每个栈元素：(a, lo, hi, al, be, depth, path_id)。对每个根片段分别求深度和大小，取最大值。
    # 做法：先递归地对一个根区间求 (最大深度, 最大大小)，需要按根 θ 的划分细化，所以返回的是根区间的分段函数。
    def rec(a, lo, hi, al, be):
        """返回 [(lo,hi, depth, size)]：根区间 (lo,hi) 上 I(a, θ') 的子树深度与大小（分段常数）。"""
        if a == 1: return [(lo, hi, 0, 0)]
        out = []
        for r0, r1, tm in pieces(a, lo, hi, al, be):
            if a * tm < 1: out.append((r0, r1, 0, 0)); continue
            for v in range(1, a):
                fv = v * tm - floor(v * tm)
                if fv * a < 1 or (1 - fv) * a < 1: break
            m = floor(a * tm)
            A = rec(v, r0, r1, al, be); B = rec(v, r0, r1, a * al, a * be - m)
            # 合并两组分段
            pts = sorted(set([r0, r1] + [p for s in A for p in s[:2]] + [p for s in B for p in s[:2]]))
            for i in range(len(pts) - 1):
                u0, u1 = pts[i], pts[i + 1]; mid = (u0 + u1) / 2
                da = next(s for s in A if s[0] <= mid <= s[1]); db = next(s for s in B if s[0] <= mid <= s[1])
                out.append((u0, u1, 1 + max(da[2], db[2]), 1 + da[3] + db[3]))
        return out
    segs = rec(q, Fr(1, q), Fr(1), 1, Fr(0))
    D = max(s[2] for s in segs); S = max(s[3] for s in segs)
    meas = {}
    for s in segs: meas[s[2]] = meas.get(s[2], 0) + (s[1] - s[0])
    tot = 1 - Fr(1, q)
    tail = {}; acc = Fr(0)
    for d in sorted(meas, reverse=True): acc += meas[d]; tail[d] = float(acc / tot)
    return dict(q=q, segments=len(segs), max_depth=D, max_size=S, tail={k: round(tail[k], 7) for k in sorted(tail)})
if __name__ == "__main__":
    for q in map(int, sys.argv[1].split(",")):
        print(json.dumps(explore(q)), flush=True)
