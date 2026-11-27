"""T1 检查：带粗糙限制的 Plancherel 恒等式
(1/N) Σ_a |S_z(da/N) − λ(d) S_z(a/N)|² = 4·#{n∈R∩φ_d⁻¹(R): λ(dn)λ(φ_d n) = −1} + |R Δ φ_d(R)|
其中 R = {1≤n<N : n 与 P_z 互素}，φ_d(n) = dn mod N。并检查 P_z | N 时 φ_d(R) = R。"""
import numpy as np
from math import gcd
def lam_table(M):
    lam = np.ones(M + 1, dtype=np.int64)
    for n in range(2, M + 1):
        m, c, p = n, 0, 2
        while p * p <= m:
            while m % p == 0: m //= p; c += 1
            p += 1
        if m > 1: c += 1
        lam[n] = (-1) ** c
    return lam
lam = lam_table(7000)
def check(N, d, z):
    Pz = [p for p in range(2, z + 1) if all(p % q for q in range(2, p))]
    R = np.array([all(n % p for p in Pz) for n in range(N)]); R[0] = False
    n = np.arange(N); phi = (d * n) % N
    lamv = lam[:N]
    S = lambda a: np.sum(np.where(R, lamv, 0) * np.exp(2j * np.pi * a * n / N))
    lhs = np.mean([abs(S(d * a) - lam[d] * S(a)) ** 2 for a in range(N)])
    inv = np.empty(N, dtype=np.int64); inv[phi] = n
    phiR = np.zeros(N, bool); phiR[phi[R]] = True
    both = R & R[phi]                       # n∈R 且 φ_d(n)∈R
    bad = int(np.sum(both & (lam[d] * lamv * lamv[phi] == -1)))
    sym = int(np.sum(R ^ phiR))
    return lhs, 4 * bad + sym, sym
for N, d, z in [(997, 7, 1), (997, 7, 3), (1009, 11, 5), (30 * 31, 7, 5), (2 * 3 * 5 * 7 * 11, 13, 7), (2 * 3 * 5 * 7 * 11 * 3, 13, 11)]:
    if gcd(d, N) != 1: continue
    lhs, rhs, sym = check(N, d, z)
    print(f"N={N} d={d} z={z}: LHS={lhs:.6f} RHS={rhs} |RΔφ_d(R)|={sym}")
