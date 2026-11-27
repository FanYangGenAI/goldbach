"""找几个"GF(2) 推出但反射图推不出"的 T(t)，给出一组推出它的刚性方程（n 值及因子分解），寻找新机制。"""
import sys
from rigidity_gf2 import odd_primes_upto
from reflection_graph import run as graph_run
k = int(sys.argv[1]); N = 1 << k; L = 3 * N
_, imp, _ = graph_run(k)
primes = odd_primes_upto(N); idx = {p: i for i, p in enumerate(primes)}; E = len(primes)
spf = list(range(L + 1))
for i in range(2, int(L ** .5) + 1):
    if spf[i] == i:
        for j in range(i * i, L + 1, i):
            if spf[j] == j: spf[j] = i
def fac(n):
    f = []
    while n > 1: q = spf[n]; f.append(q); n //= q
    return f
def vec(n):
    v = 0
    for q in fac(n):
        if q != 2:
            if q >= N: return None
            v ^= 1 << idx[q]
    return v
ns = list(range(1, N // 2 + 1, 2))
rows = [vec(n) ^ vec(N - n) ^ (1 << E) for n in ns]
# 按 n 从小到大加入，偏好用小的 n
piv = {}
for i, r in enumerate(rows):
    c = 1 << i
    while r:
        h = r.bit_length() - 1
        if h in piv: r ^= piv[h][0]; c ^= piv[h][1]
        else: piv[h] = (r, c); break
shown = 0
for t in range(N + 1, 2 * N, 2):
    vt, vs = vec(t), vec(t % N)
    if vt is None or vs is None or imp[t]: continue
    v = vt ^ vs; c = 0
    while v:
        h = v.bit_length() - 1
        if h not in piv: c = None; break
        v ^= piv[h][0]; c ^= piv[h][1]
    if c is None: continue
    S = [ns[i] for i in range(len(ns)) if c >> i & 1]
    if len(S) > 8: continue
    print(f"T({t}) [t mod N = {t % N}; t = {fac(t)}; t mod N = {fac(t % N)}] 由 {len(S)} 个刚性方程推出：")
    for n in S: print(f"    n={n} {fac(n)}  N−n={N-n} {fac(N-n)}")
    shown += 1
    if shown >= 6: break
