"""M1-E7（探索性）：反射图闭包 vs GF(2) 闭包。
严格的事实（见 RESULTS 第 5 节）：对奇数 t 的任一奇因子 d，若 t/d < N，则由 t/d 与 d(N−t/d) 处的刚性，
  T(t) ⇔ T(dN − t)。       （反射边）
T(t) 对 t < N 平凡成立。于是从 {t < N} 出发沿反射边可达的 t，其 T(t) 都由刚性推出（且近刚性时缺陷 ≤ 2×路径长×|E|）。
问题：反射图闭包是否等于 GF(2) 闭包（即刚性能推出的全部 T）？"""
import sys, json
from collections import deque
def run(k, cap_mult=40):
    N = 1 << k; CAP = cap_mult * N
    spf = list(range(CAP + 1))
    for i in range(2, int(CAP ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, CAP + 1, i):
                if spf[j] == j: spf[j] = i
    def odd_divisors(t):
        fs = {}
        while t > 1:
            q = spf[t]; fs[q] = fs.get(q, 0) + 1; t //= q
        ds = [1]
        for q, e in fs.items():
            if q == 2: continue
            ds = [d * q ** i for d in ds for i in range(e + 1)]
        return ds
    implied = bytearray(CAP + 1); dist = {}
    dq = deque()
    for t in range(1, N, 2): implied[t] = 1; dist[t] = 0
    # 反向搜索：节点 u 被推出若存在邻居已被推出。用 BFS：从已知点扩展到所有邻居（边无向）
    # 需要能从 u 找到所有邻居：邻居 t' = dN − u，其中 d | u 且 u/d < N（u 的反射），或者 u = dN − t' 且 d | t'（反过来）。
    # 后者等价于 d | (dN − u) ⇔ d | u（因 d | dN）且 t'/d = N − u/d < N ⇔ u/d > 0。所以邻居集合 = {dN − u : d 奇, d | u, d ≥ 3, dN > u}，两种方向一致。
    for t in range(1, N, 2): dq.append(t)
    while dq:
        u = dq.popleft()
        for d in odd_divisors(u):
            if d < 3 or d * N <= u: continue
            v = d * N - u
            if v <= CAP and not implied[v]:
                implied[v] = 1; dist[v] = dist[u] + 1; dq.append(v)
    return N, implied, dist
if __name__ == "__main__":
    import T_closure  # 复用其 GF(2) 判定，重写一个轻量版
    from rigidity_gf2 import odd_primes_upto
    for k in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
        N, imp, dist = run(k)
        # GF(2) 闭包（只比较 (N,3N) 中所有素因子 < N 的 t）
        L = 3 * N
        primes = odd_primes_upto(N); idx = {p: i for i, p in enumerate(primes)}; E = len(primes)
        spf = list(range(L + 1))
        for i in range(2, int(L ** .5) + 1):
            if spf[i] == i:
                for j in range(i * i, L + 1, i):
                    if spf[j] == j: spf[j] = i
        def vec(n):
            v = 0
            while n > 1:
                q = spf[n]; c = 0
                while n % q == 0: n //= q; c ^= 1
                if q != 2 and c:
                    if q >= N: return None
                    v ^= 1 << idx[q]
            return v
        piv = {}
        for n in range(1, N // 2 + 1, 2):
            r = vec(n) ^ vec(N - n) ^ (1 << E)
            while r:
                h = r.bit_length() - 1
                if h in piv: r ^= piv[h]
                else: piv[h] = r; break
        def in_span(v):
            while v:
                h = v.bit_length() - 1
                if h not in piv: return False
                v ^= piv[h]
            return True
        both = gf_only = graph_only = neither = 0; maxd = 0
        for t in range(N + 1, L, 2):
            vt, vs = vec(t), vec(t % N)
            if vt is None or vs is None: continue
            g2 = in_span(vt ^ vs); gr = bool(imp[t])
            if g2 and gr: both += 1; maxd = max(maxd, dist[t])
            elif g2: gf_only += 1
            elif gr: graph_only += 1
            else: neither += 1
        print(json.dumps(dict(k=k, both=both, gf2_only=gf_only, graph_only=graph_only, neither=neither, max_path_len=maxd)), flush=True)
