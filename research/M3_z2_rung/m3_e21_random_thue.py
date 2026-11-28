"""M3-E21（预注册见 DESIGN.md 第 18 节）：随机化 Thue 树的分数证书。"""
import sys, json, random
from math import isqrt
from collections import Counter
def rows_rand(a, x, N, out, rng):
    if a * x < N or a == 1: return
    opts = []
    for v in range(1, a):
        u = (x * v) % N
        if u * a < N: opts.append((v, 1, u))
        elif (N - u) * a < N: opts.append((v, -1, N - u))
    v, eps, y = rng.choice(opts)
    r = (a * x) % N
    rows_rand(v, x, N, out, rng); rows_rand(v, r, N, out, rng)
    if eps == -1:
        for m in (y, a * y): out[min(m % N, N - m % N)] ^= 1
def run(N, q, T=16, seed=1):
    rng = random.Random(seed); load = Counter(); fails = 0; sizes = []
    for n in range(1, N):
        if q * n < N: continue
        r = (q * n) % N
        for _ in range(T):
            out = Counter(); rows_rand(q, n, N, out, rng)
            S = [x for x, c in out.items() if c]; sizes.append(len(S))
            P = q * n * r
            for x in S: P *= x * (N - x)
            if len(S) % 2 or isqrt(P) ** 2 != P: fails += 1
            for x in S: load[x] += 1.0 / T
    R = (N - 1) // 2
    return dict(N=N, q=q, T=T, square_fail=fails, cert_mean=round(sum(sizes) / len(sizes), 2),
                load_max=round(max(load.values()), 2), load_mean=round(sum(load.values()) / R, 2),
                argmax=[(k, round(v, 2)) for k, v in load.most_common(3)])
if __name__ == "__main__":
    for N in map(int, sys.argv[1].split(",")):
        for q in map(int, sys.argv[2].split(",")):
            print(json.dumps(run(N, q)), flush=True)
