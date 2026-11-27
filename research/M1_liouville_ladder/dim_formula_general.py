"""M1-E10：精确刚性解空间维数公式的检验（一般偶数 N）。
猜想：dim = c(N) + G1(N) + G3(N) + f(N)，其中
  G1 = #{p 素数，N/3 < p ≤ N/2，N−p 素数}（两个都 > N/3 的哥德巴赫对，与 N 互素）
  G3 = #{p 素数，N/5 < p < N/3，N−p 与 N−3p 都是素数且 N−3p > N/3}
  f  = 只出现在"自配对"或成对抵消位置的、整除 N 的素数个数（这里只统计 p | N 且 p > N/3 的情形：n = p 与 N−p 同含 p）
  c  = 刚性实特征张成的维数（小常数）。输出 dim − (G1+G3) 以观察 c+f 是否为小常数。"""
import sys, json
from rigidity_gf2_nullspace import nullspace
def primes_upto(n):
    s = bytearray([1]) * (n + 1); s[0] = s[1] = 0
    for i in range(2, int(n ** .5) + 1):
        if s[i]: s[i*i::i] = bytearray(len(s[i*i::i]))
    return [p for p in range(2, n + 1) if s[p]]
def dim_rigid(N):
    ps = [p for p in primes_upto(N) if p > 2]; idx = {p: i for i, p in enumerate(ps)}; E = len(ps)
    spf = list(range(N + 1))
    for i in range(2, int(N ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j: spf[j] = i
    def vec(n):
        v = 0
        while n > 1:
            q = spf[n]; c = 0
            while n % q == 0: n //= q; c ^= 1
            if c and q != 2: v ^= 1 << idx[q]
        return v
    rows = [vec(n) ^ vec(N - n) ^ (1 << E) for n in range(1, N // 2 + 1, 2)]
    return len(nullspace(rows, E + 1)), set(ps)
for N in [int(a) for a in sys.argv[1:]]:
    d, P = dim_rigid(N)
    G1 = sum(1 for p in P if N / 3 < p <= N / 2 and (N - p) in P and N % p)
    G3 = sum(1 for p in P if N / 5 < p < N / 3 and (N - p) in P and (N - 3 * p) in P and (N - 3 * p) > N / 3 and N % p)
    print(json.dumps(dict(N=N, dim=d, G1=G1, G3=G3, residual=d - G1 - G3)), flush=True)
