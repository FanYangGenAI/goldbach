"""M3-E20：用 θ = x/N 的分段结构精确计算 Thue 树负载的 N 无关上界 F*(q)。
记录 = (根区间 I, 行位置 φ(θ) = sθ + c，s 为非零整数)。对一切素数 N > q，每条记录对每一行至多贡献 1，
所以最大负载 ≤ F*(q) = max_φ #{记录 : φ ∈ 像}。"""
import sys, json
from fractions import Fraction as Fr
from math import floor
def pieces(a, lo, hi, al, be):
    """在根区间 (lo,hi) 上，节点点 θ' = al·θ + be（al 为非零整数）。按 Thue 选择把根区间切开。"""
    # θ' 的像区间
    t0, t1 = sorted((al * lo + be, al * hi + be))
    bps = set()
    for v in range(1, a + 1):
        for k in range(floor(v * t0) - 1, floor(v * t1) + 2):
            for off in (Fr(0), Fr(1, a), -Fr(1, a)):
                t = (k + off) / v
                if t0 < t < t1: bps.add(t)
    pts = [t0] + sorted(bps) + [t1]
    out = []
    for i in range(len(pts) - 1):
        u0, u1 = pts[i], pts[i + 1]
        # 换回根坐标
        r0, r1 = sorted(((u0 - be) / al, (u1 - be) / al))
        out.append((r0, r1, (u0 + u1) / 2))
    return out
def build(q, maxrec=10 ** 7):
    recs = []
    stack = [(q, Fr(1, q), Fr(1), 1, Fr(0))]  # (a, 根区间 lo, hi, al, be)：θ' = al θ + be
    while stack:
        a, lo, hi, al, be = stack.pop()
        if a == 1: continue
        for r0, r1, tm in pieces(a, lo, hi, al, be):
            if a * tm < 1: continue  # 平凡
            # Thue 选择
            for v in range(1, a):
                fv = v * tm - floor(v * tm)
                if fv * a < 1: eps, k = 1, floor(v * tm); break
                if (1 - fv) * a < 1: eps, k = -1, floor(v * tm); break
            else: raise RuntimeError
            m = floor(a * tm)
            stack.append((v, r0, r1, al, be))                 # I(v, x')
            stack.append((v, r0, r1, a * al, a * be - m))     # I(v, a x' mod N)
            if eps == -1:
                # y 的位置 φ_y = (k+1) − vθ'，ay 的位置 a·φ_y；θ' = al θ + be
                for mult in (1, a):
                    s = -mult * v * al; c = mult * ((k + 1) - v * be)
                    recs.append((r0, r1, s, c))
            if len(recs) > maxrec: raise RuntimeError("too many")
    return recs
def coverage(recs):
    ev = []
    for r0, r1, s, c in recs:
        p0, p1 = sorted((s * r0 + c, s * r1 + c))   # φ ∈ (0,1) 上的开区间
        # 折叠 φ → min(φ, 1−φ)
        segs = []
        if p1 <= Fr(1, 2): segs.append((p0, p1))
        elif p0 >= Fr(1, 2): segs.append((1 - p1, 1 - p0))
        else: segs += [(p0, Fr(1, 2)), (1 - p1, Fr(1, 2))]
        for a0, a1 in segs:
            if a1 > a0: ev.append((a0, 1)); ev.append((a1, -1))
    ev.sort(key=lambda e: (e[0], e[1]))  # 开区间：同一点先减后加
    cur = best = 0; arg = None
    for x, d in ev:
        cur += d
        if cur > best: best, arg = cur, x
    return best, arg
if __name__ == "__main__":
    for q in map(int, sys.argv[1].split(",")):
        recs = build(q); F, arg = coverage(recs)
        print(json.dumps(dict(q=q, records=len(recs), Fstar=F, at=str(arg))), flush=True)
