"""M3-E16（探索性，非预注册）：在已验证的证书系统中，按行的类型（q | x(N−x) 与否、x 的大小）统计负载。"""
import json, sys, glob
from fractions import Fraction
for path in sorted(glob.glob("certs/*.json")):
    D = json.load(open(path)); N, q = D["N"], D["q"]; load = {}
    for n, lst in D["certs"].items():
        for S, a, b in lst:
            for x in S: load[x] = load.get(x, 0.0) + a / b
    R = (N - 1) // 2
    qrows = [x for x in range(1, R + 1) if (x * (N - x)) % q == 0]
    other = [x for x in range(1, R + 1) if (x * (N - x)) % q]
    mx = lambda L: max((load.get(x, 0.0) for x in L), default=0)
    avg = lambda L: sum(load.get(x, 0.0) for x in L) / max(len(L), 1)
    small = [x for x in other if x <= 2 * q]
    print(json.dumps(dict(N=N, q=q, T=round(max(load.values()), 3), q_rows=len(qrows), max_q=round(mx(qrows), 3), avg_q=round(avg(qrows), 3),
          max_other=round(mx(other), 3), avg_other=round(avg(other), 3), max_other_small_x=round(mx(small), 3),
          argmax_other=max(other, key=lambda x: load.get(x, 0.0)))))
# 对照：g = χ_N 但翻转 g(q)（严格的下界例子），计算 ρ = |𝓔_q| / #坏行，并给出 T − ρ
def g_val(m, N, q):
    s = 1 if pow(m, (N - 1) // 2, N) == 1 else -1   # χ_N(m)
    while m % q == 0: m //= q; s = -s                 # 每个因子 q 额外乘 −1（翻转 g(q)）
    return s
for path in sorted(glob.glob("certs/*.json")):
    D = json.load(open(path)); N, q = D["N"], D["q"]; R = (N - 1) // 2
    vals = [g_val(x, N, q) * g_val(N - x, N, q) for x in range(1, R + 1)]
    eta = 1 if sum(vals) >= 0 else -1; bad = sum(v != eta for v in vals)
    Eq = sum(g_val(q * n, N, q) != g_val((q * n) % N, N, q) for n in range(1, N))
    load = {}
    for n, lst in D["certs"].items():
        for S, a, b in lst:
            for x in S: load[x] = load.get(x, 0.0) + a / b
    T = max(load.values())
    print(json.dumps(dict(N=N, q=q, flip_bad_rows=bad, flip_Eq=Eq, rho_flip=round(Eq / bad, 3), T=round(T, 3), gap=round(T - Eq / bad, 3))))
