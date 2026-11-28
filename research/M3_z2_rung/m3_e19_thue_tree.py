"""M3-E19（预注册见 DESIGN.md 第 16 节）：Thue 树证书。"""
import sys, json
from math import isqrt
from functools import lru_cache
from collections import Counter
sys.setrecursionlimit(10000)
def run(N, q, check=True):
    R = (N - 1) // 2
    pr = lambda m: min(m % N, N - m % N)
    @lru_cache(maxsize=None)
    def inst(a, x):
        """返回实例 I(a, x) 的行集合（按 GF(2) 重数取对称差），用 Python 整数作位掩码。"""
        if a * x < N or a == 1: return 0
        for v in range(1, a):
            u = (x * v) % N
            if u * a < N: eps, y = 1, u; break
            if (N - u) * a < N: eps, y = -1, N - u; break
        else: raise RuntimeError("Thue 失败")
        r = (a * x) % N
        m = inst(v, x) ^ inst(v, r)
        if eps == -1: m ^= (1 << pr(y)) ^ (1 << pr(a * y))
        return m
    load = Counter(); sizes = []; fails = 0
    for n in range(1, N):
        if q * n < N: continue
        m = inst(q, n); S = [i for i in range(1, R + 1) if m >> i & 1]
        sizes.append(len(S))
        for i in S: load[i] += 1
        if check:
            r = (q * n) % N; P = q * n * r
            for x in S: P *= x * (N - x)
            if len(S) % 2 or isqrt(P) ** 2 != P: fails += 1
    mx = max(load.values()) if load else 0
    return dict(N=N, q=q, square_fail=fails, cert_mean=round(sum(sizes) / len(sizes), 2), cert_max=max(sizes),
                load_max=mx, load_mean=round(sum(load.values()) / R, 2), argmax=load.most_common(3))
if __name__ == "__main__":
    for N in map(int, sys.argv[1].split(",")):
        for q in map(int, sys.argv[2].split(",")):
            print(json.dumps(run(N, q, check=(N <= 5000))), flush=True)
