"""M3-E1（预注册，DESIGN.md 第 1 节）：N = 2^k 奇数子纤维上 T(qn) 的 GF(2) 覆盖率与反射覆盖率。"""
import sys, json, time
sys.path.insert(0, "../M1_liouville_ladder")
from rigidity_gf2 import odd_primes_upto
from collections import defaultdict, deque
def run(k, qs=(3, 5, 7, 11, 13)):
    N = 1 << k; L = 13 * N + 1
    primes = odd_primes_upto(N); idx = {p: i for i, p in enumerate(primes)}; E = len(primes)
    spf = list(range(L + 1))
    for i in range(2, int(L ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, L + 1, i):
                if spf[j] == j: spf[j] = i
    def vec(n):
        v = 0
        while n > 1:
            p = spf[n]; c = 0
            while n % p == 0: n //= p; c ^= 1
            if p != 2 and c:
                if p >= N: return None
                v ^= 1 << idx[p]
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
    out = []
    for q in qs:
        # 反射覆盖（层级图，奇数乘数）：D = {奇数 d ≤ 6q} ∪ {q·a：a 奇，a ≤ 6q}
        Z = 6 * q
        D = sorted({d for d in range(3, Z + 1, 2)} | {q * a for a in range(1, Z + 1, 2)})
        inv = {d: pow(N, -1, d) for d in D}
        tot = gf = refl = 0
        for n in range(1, N, 2):
            t = q * n
            vt = vec(t); vs = vec(t % N)
            if vt is None or vs is None: continue
            tot += 1
            gf += in_span(vt ^ vs)
            j0 = t // N; a = t - j0 * N
            if j0 == 0: refl += 1; continue
            par = {}
            def find(x):
                while par.get(x, x) != x:
                    par[x] = par.get(par[x], par[x]); x = par[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx != ry: par[rx] = ry
            union(0, -1)
            for d in D:
                e = (-a * inv[d]) % d; union(e, e - d)
            refl += (find(j0) == find(0))
        out.append(dict(k=k, q=q, n_considered=tot, gf2_coverage=round(gf / tot, 4), reflection_coverage=round(refl / tot, 4)))
    return out
if __name__ == "__main__":
    for k in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
        t0 = time.time()
        for r in run(k): r["sec"] = round(time.time() - t0, 1); print(json.dumps(r), flush=True)
