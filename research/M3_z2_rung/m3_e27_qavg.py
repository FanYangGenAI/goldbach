"""M3-E27（预注册见 DESIGN.md 第 21 节）：对 q 取平均的 Thue 树负载。"""
import sys, json
from collections import Counter
from m3_e25_split import rows2
from m3_e8_ratio import primes_upto
def loads(N, q):
    L = Counter()
    for n in range(1, N):
        if q * n < N: continue
        out = Counter(); rows2(q, n, N, out)
        for x, c in out.items():
            if c: L[x] += 1
    return L
if __name__ == "__main__":
    for N in map(int, sys.argv[1].split(",")):
        for lo in (16, 32):
            Q = [p for p in primes_upto(2 * lo) if p > lo]
            tot = Counter(); summax = 0; R = (N - 1) // 2
            for q in Q:
                L = loads(N, q); summax += max(L.values()); tot.update(L)
            K = max(tot.values())
            print(json.dumps(dict(N=N, Q=Q, K=K, K_over_Q=round(K / len(Q), 2), sum_of_max=summax, ratio=round(K / summax, 3),
                                  mean_total=round(sum(tot.values()) / R, 2), argmax=tot.most_common(3))), flush=True)
