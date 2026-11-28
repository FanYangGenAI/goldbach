"""独立的整数检验器（不依赖线性规划）。读入 certs/cert_N{N}_q{q}.json，检查：
(1) 每张证书 S：|S| 为偶数，且 qn·r·∏_{x∈S} x(N−x) 是完全平方数（引理 Q）；
(2) 每个 qn ≥ N 的 n ∈ [1, N−1] 都有证书，且权重为正、和恰为 1；
(3) 精确计算每行负载的最大值 T。
结论：对一切完全积性 g: ℕ → {±1} 和任一 η，|𝓔_q(g)| ≤ T·#{x ∈ [1,(N−1)/2] : g(x)g(N−x) ≠ η}。"""
import sys, json
from fractions import Fraction
from math import isqrt
def is_prime(n): return n > 1 and all(n % p for p in range(2, isqrt(n) + 1))
def verify(path):
    D = json.load(open(path)); N, q = D["N"], D["q"]; C = D["certs"]
    assert is_prime(N) and is_prime(q) and q < N
    load = {}
    for n in range(1, N):
        if q * n < N: continue
        lst = C[str(n)]; r = (q * n) % N; tot = Fraction(0)
        for S, a, b in lst:
            w = Fraction(a, b); assert w > 0; tot += w
            assert len(S) % 2 == 0 and len(set(S)) == len(S) and all(1 <= x <= (N - 1) // 2 for x in S)
            P = q * n * r
            for x in S: P *= x * (N - x)
            assert isqrt(P) ** 2 == P, (n, S)
        assert tot == 1, (n, tot)
        for S, a, b in lst:
            for x in S: load[x] = load.get(x, Fraction(0)) + Fraction(a, b)
    T = max(load.values()); return N, q, T
if __name__ == "__main__":
    for p in sys.argv[1:]:
        N, q, T = verify(p); print(f"N={N} q={q}: VERIFIED  T = {T} ≈ {float(T):.6f}")
