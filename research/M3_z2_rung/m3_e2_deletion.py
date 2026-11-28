"""M3-E2（预注册，DESIGN 第 2 节）：删去刚性方程组中的若干行后，失去 T(qn) 的 n 有多少（GF(2) 版多重度）。"""
import sys, json, random, time
sys.path.insert(0, "../M1_liouville_ladder")
from rigidity_gf2 import odd_primes_upto
def setup(k, q):
    N = 1 << k; L = q * N + 1
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
            if p != 2 and c: v ^= 1 << idx[p]
        return v
    rows = [(n, vec(n) ^ vec(N - n) ^ (1 << E)) for n in range(1, N // 2 + 1, 2)]
    targets = [(n, vec(q * n) ^ vec((q * n) % N)) for n in range(1, N, 2)]
    return N, primes, rows, targets
def span_test(rows_used, targets):
    piv = {}
    for _, r in rows_used:
        while r:
            h = r.bit_length() - 1
            if h in piv: r ^= piv[h]
            else: piv[h] = r; break
    def ins(v):
        while v:
            h = v.bit_length() - 1
            if h not in piv: return False
            v ^= piv[h]
        return True
    return {n for n, v in targets if ins(v)}
def run(k, q=3, seed=0):
    rng = random.Random(seed)
    N, primes, rows, targets = setup(k, q)
    base = span_test(rows, targets)
    res = dict(k=k, q=q, N=N, rows=len(rows), base_derivable=len(base) / len(targets))
    for r in (1, 4, 16, 64):
        ks = []
        for _ in range(10):
            B = set(rng.sample(range(len(rows)), r))
            after = span_test([rw for i, rw in enumerate(rows) if i not in B], targets)
            ks.append(len(base - after))
        res[f"random_r{r}_killed_mean"] = round(sum(ks) / len(ks), 2); res[f"random_r{r}_ratio"] = round(sum(ks) / len(ks) / r, 3)
    import math
    ps = [p for p in primes if p > math.isqrt(N)][:3]
    st = []
    for p in ps:
        B = {i for i, (n, _) in enumerate(rows) if n % p == 0 or (N - n) % p == 0}
        after = span_test([rw for i, rw in enumerate(rows) if i not in B], targets)
        st.append(dict(p=p, rows_removed=len(B), killed=len(base - after), ratio=round(len(base - after) / max(1, len(B)), 3)))
    res["structured"] = st
    return res
if __name__ == "__main__":
    for k in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
        t0 = time.time(); r = run(k, seed=k); r["sec"] = round(time.time() - t0, 1); print(json.dumps(r), flush=True)
