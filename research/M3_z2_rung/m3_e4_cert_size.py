"""M3-E4（探索性，非预注册）：小 N 上"可由 GF(2) 推出、但反射推不出"的 T(qn) 的最小证书大小（中间相遇，至多 6 行）。"""
import sys, json, itertools
from collections import defaultdict, deque, Counter
from m3_e2_deletion import setup
def refl_cov(N, q, n, Z):
    D = sorted({d for d in range(3, Z + 1, 2)} | {q * a for a in range(1, Z + 1, 2)})
    t = q * n; j0 = t // N; a = t - j0 * N
    if j0 == 0: return True
    par = {}
    def find(x):
        while par.get(x, x) != x: x = par.get(x, x)
        return x
    for d in D:
        e = (-a * pow(N, -1, d)) % d; rx, ry = find(e), find(e - d)
        if rx != ry: par[rx] = ry
    rx, ry = find(-1), find(0)
    if rx != ry: par[rx] = ry
    return find(j0) == find(0)
k, q = int(sys.argv[1]), int(sys.argv[2])
N, primes, rows, targets = setup(k, q)
R = [r for _, r in rows]
half = defaultdict(list)
for s in range(0, 4):
    for comb in itertools.combinations(range(len(R)), s):
        x = 0
        for i in comb: x ^= R[i]
        if x not in half: half[x] = comb
sizes = Counter(); examples = []
for n, v in targets:
    if refl_cov(N, q, n, 6 * q): continue
    best = None
    for x, comb in half.items():
        y = v ^ x
        if y in half:
            s = len(comb) + len(half[y])
            if best is None or s < best[0]: best = (s, comb, half[y])
    if best is None: sizes[">6"] += 1
    else:
        sizes[best[0]] += 1
        if len(examples) < 4: examples.append(dict(n=n, size=best[0], rows=[rows[i][0] for i in set(best[1]) ^ set(best[2])]))
print(json.dumps(dict(k=k, q=q, N=N, not_reflection_covered=sum(sizes.values()), cert_size_dist={str(a): b for a, b in sorted(sizes.items(), key=lambda x: str(x[0]))}, examples=examples)))
