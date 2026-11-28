"""M3-E25（预注册见 DESIGN.md 第 20 节）：合数乘数精确分裂的 Thue 树。"""
import sys, json
from math import isqrt
from collections import Counter
def spf(a):
    p = 2
    while p * p <= a:
        if a % p == 0: return p
        p += 1
    return a
def rows2(a, x, N, out):
    if a * x < N or a == 1: return
    p = spf(a)
    if p < a:  # 合数：精确分裂
        a2 = a // p
        rows2(a2, x, N, out); rows2(p, (a2 * x) % N, N, out); return
    for v in range(1, a):
        u = (x * v) % N
        if u * a < N: eps, y = 1, u; break
        if (N - u) * a < N: eps, y = -1, N - u; break
    r = (a * x) % N
    rows2(v, x, N, out); rows2(v, r, N, out)
    if eps == -1:
        for m in (y, a * y): out[min(m % N, N - m % N)] ^= 1
def run(N, q):
    load = Counter(); sizes = []; fails = 0
    for n in range(1, N):
        if q * n < N: continue
        out = Counter(); rows2(q, n, N, out); S = [x for x, c in out.items() if c]; sizes.append(len(S))
        r = (q * n) % N; P = q * n * r
        for x in S: P *= x * (N - x)
        if len(S) % 2 or isqrt(P) ** 2 != P: fails += 1
        for x in S: load[x] += 1
    R = (N - 1) // 2
    return dict(N=N, q=q, square_fail=fails, cert_mean=round(sum(sizes) / len(sizes), 2), cert_max=max(sizes),
                load_max=max(load.values()), load_mean=round(sum(load.values()) / R, 2), argmax=load.most_common(3))
if __name__ == "__main__":
    for N in map(int, sys.argv[1].split(",")):
        for q in map(int, sys.argv[2].split(",")):
            print(json.dumps(run(N, q)), flush=True)
