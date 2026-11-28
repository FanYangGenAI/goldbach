"""M3-E32（探索性）：二阶矩（按重数计）的连续极限 σ₂(q)，用记录对的格点计数计算，并与数值比较。
行 ρ = s·x + c·N（s、c 为整数）。两条记录的公共行数 ≈ N·|J₁∩J₂|/lcm(|s₁|,|s₂|)·[c₁ ≡ c₂ mod gcd]，相容条件与 N 无关。"""
import sys, json
from math import gcd
from fractions import Fraction as Fr
from collections import Counter
from m3_e20_Fq_exact import build
def parts(q):
    P = []
    for r0, r1, s, c in build(q):
        p0, p1 = sorted((s * r0 + c, s * r1 + c))
        h = Fr(1, 2)
        if p0 < h: P.append((float(p0), float(min(p1, h)), s, c))
        if p1 > h: P.append((float(1 - p1), float(1 - max(p0, h)), -s, 1 - c))
    return P
def sigma2(q):
    P = parts(q); tot = 0.0
    for i, (a0, a1, s1, c1) in enumerate(P):
        for (b0, b1, s2, c2) in P:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi <= lo: continue
            g = gcd(abs(s1), abs(s2))
            if (c1 - c2) % g: continue
            tot += (hi - lo) * g / (abs(s1) * abs(s2))
    return 2 * tot, len(P)
def numeric(N, q):
    L = Counter()
    def rec(a, x):
        if a * x < N or a == 1: return
        for v in range(1, a):
            u = (x * v) % N
            if u * a < N: eps, y = 1, u; break
            if (N - u) * a < N: eps, y = -1, N - u; break
        rec(v, x); rec(v, (a * x) % N)
        if eps == -1:
            for m in (y, a * y): L[min(m % N, N - m % N)] += 1
    for n in range(1, N):
        if q * n >= N: rec(q, n)
    R = (N - 1) // 2
    return sum(v * v for v in L.values()) / R
if __name__ == "__main__":
    for q in map(int, sys.argv[1].split(",")):
        s2, npart = sigma2(q)
        print(json.dumps(dict(q=q, parts=npart, sigma2_continuum=round(s2, 4), m2_mult_N65521=round(numeric(65521, q), 4), m2_mult_N262139=round(numeric(262139, q), 4))), flush=True)
