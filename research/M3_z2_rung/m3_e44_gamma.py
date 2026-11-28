"""M3-E44（预注册见 DESIGN.md 第 29 节）：松弛″细化路径的碰撞量 Γ_j。"""
import sys, json
from math import gcd
from fractions import Fraction
def all_paths(q, J):
    # 每条细化路径记为 (乘数序列, 数字串)；用数字自动机生成：第 i+1 个乘数 b 取遍 [2, a_i − 1]，
    # 数字 c_{i+1} = (−b c_i − e) mod a_i，要求 c_{i+1} ∈ [1, b − 2]
    out = {0: [((q,), (c,)) for c in range(1, q - 1)]}
    for j in range(1, J + 1):
        nxt = []
        for seq, cs in out[j - 1]:
            a = seq[-1]; c = cs[-1]
            for b in range(4, a):
                for e in (0, 1):
                    cn = (-b * c - e) % a
                    if 1 <= cn <= b - 2: nxt.append((seq + (b,), cs + (cn,)))
        out[j] = nxt
    return out
def MP(seq, cs):
    M = 0; P = 1
    for a, c in zip(seq, cs): M = M * a + c; P *= a
    return M, P
if __name__ == "__main__":
    J = int(sys.argv[2])
    for q in map(int, sys.argv[1].split(",")):
        paths = all_paths(q, J)
        for j in range(1, J + 1):
            items = [MP(s, c) for s, c in paths[j]]
            mu = sum(1.0 / P for M, P in items)
            # 按 P 分组，加速同序列配对（贡献为 0）
            G = 0.0
            for i, (M, P) in enumerate(items):
                G += 1.0 / P                      # 对角项
                for (M2, P2) in items[i + 1:]:
                    if P2 == P and M2 != M: continue
                    g = gcd(P, P2)
                    if (M - M2) % g == 0: G += 2.0 * g / (P * P2)
            print(json.dumps(dict(q=q, j=j, paths=len(items), sum_mu=round(mu, 4), Gamma=round(G, 4), ratio=round(G / (mu * mu + mu), 3))), flush=True)
