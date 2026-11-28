"""M3-E29（预注册见 DESIGN.md 第 23 节）：Thue 树负载的二阶矩。"""
import sys, json
from collections import Counter
from m3_e25_split import rows2
if __name__ == "__main__":
    for N in map(int, sys.argv[1].split(",")):
        for q in map(int, sys.argv[2].split(",")):
            L = Counter()
            for n in range(1, N):
                if q * n < N: continue
                out = Counter(); rows2(q, n, N, out)
                for x, c in out.items():
                    if c: L[x] += 1
            R = (N - 1) // 2
            m1 = sum(L.values()) / R; m2 = sum(v * v for v in L.values()) / R
            tail = {T: round(sum(v for v in L.values() if v > T) / sum(L.values()), 5) for T in (q, 2 * q, 3 * q)}
            print(json.dumps(dict(N=N, q=q, m1=round(m1, 3), m2=round(m2, 2), m2_over_m1sq=round(m2 / m1 ** 2, 3), max=max(L.values()), tail_mass_frac=tail)), flush=True)
