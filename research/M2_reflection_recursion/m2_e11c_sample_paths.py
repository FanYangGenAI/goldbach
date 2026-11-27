"""M2-E11c（探索性，非预注册）：打印若干条最短路（层级、乘数及其分解），看逃离之后的机制。"""
import sys, math, random
from collections import defaultdict, deque
from m2_e4_random_model import bits_needed
def fac(n):
    f = []; d = 2
    while d * d <= n:
        while n % d == 0: f.append(d); n //= d
        d += 1
    if n > 1: f.append(n)
    return "·".join(map(str, f))
q, eta, S, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
Z = round(q ** (1 + eta)); B = bits_needed(Z, q); rng = random.Random(seed)
shown = 0
while shown < S:
    y = rng.getrandbits(B); j0 = y % q
    if j0 in (0, q - 1): continue
    adj = defaultdict(list); adj[0].append((-1, 1)); adj[-1].append((0, 1))
    for d in range(2, Z + 1):
        e = y % d; adj[e].append((e - d, d)); adj[e - d].append((e, d))
    par = {0: None, -1: None}; dq = deque([0, -1])
    while dq:
        v = dq.popleft()
        for w, d in adj[v]:
            if w not in par: par[w] = (v, d); dq.append(w)
    if j0 not in par: continue
    path = [j0]; out = [f"j0={j0}"]
    while par[path[-1]] is not None:
        v, d = par[path[-1]]; tag = "R" if (v - j0) % q == 0 else " "
        out.append(f"--{d}[{fac(d)}]--> {v}{tag}"); path.append(v)
    print(f"q={q} Z={Z}: " + " ".join(out)); shown += 1
