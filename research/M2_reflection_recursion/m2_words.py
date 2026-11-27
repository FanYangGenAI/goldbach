"""查看 BFS 路径的"乘数字"(d_1, d_2, ...)，统计最常见的字，并显示每个 n 的层级序列。"""
import sys, json
import numpy as np
from collections import deque, Counter
import m2_e1 as M
def words(p, q, Z):
    CAP = q * (Z + 1) * p + p
    spf = M.spf_table(CAP)
    def allowed_divs(t):
        fs = {}; x = t
        while x > 1:
            r = int(spf[x]); fs[r] = fs.get(r, 0) + 1; x //= r
        ds = [1]
        for r, e in fs.items(): ds = [d * r ** i for d in ds for i in range(e + 1)]
        return sorted(d for d in ds if d >= 2 and (d <= Z or (d % q == 0 and d // q <= Z)))
    depth = np.full(CAP + 1, -1, dtype=np.int8); par_u = np.zeros(CAP + 1, dtype=np.int64); par_d = np.zeros(CAP + 1, dtype=np.int32)
    dq = deque(range(1, p)); depth[1:p] = 0
    while dq:
        u = dq.popleft()
        for d in allowed_divs(u):
            if d * p <= u: continue
            v = d * p - u
            if v <= CAP and v % p and depth[v] < 0:
                depth[v] = depth[u] + 1; par_u[v] = u; par_d[v] = d; dq.append(v)
    W = Counter(); examples = []
    for n in range(1, p):
        t = q * n
        if t < p or depth[t] < 0: continue
        v = t; word = []; levels = [t // p]
        while depth[v] > 0:
            u = int(par_u[v]); d = int(par_d[v]); word.append(d); v = u; levels.append(v // p)
        # 路径是从 t 走向平凡节点：par 指向更靠近根的节点，所以 word 按 t→根 的顺序，乘数是"边的 d"
        W[tuple(word)] += 1
        if len(examples) < 8 and len(word) >= 3: examples.append(dict(n=n, word=word, levels=levels))
    return W, examples
p, q, Z = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
W, ex = words(p, q, Z)
print("不同的字的个数:", len(W))
for w, c in W.most_common(15): print(c, w)
for e in ex: print(e)
