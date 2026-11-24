"""Brute-force cross-check of r(F) for q=3, n=3,4 (independent of the FFT/sieve path): irreducibility by trial division."""
import itertools, json, sys
import numpy as np
sys.argv = [sys.argv[0], "--smoke"]
q = 3


def trim(a):
    a = list(a)
    while a and a[-1] == 0: a.pop()
    return a


def pmod(a, b):
    a = trim(a); b = trim(b)
    inv = pow(b[-1], q - 2, q)
    while len(a) >= len(b):
        f = (a[-1] * inv) % q; s = len(a) - len(b)
        for i, x in enumerate(b): a[s + i] = (a[s + i] - f * x) % q
        a = trim(a)
    return a


def monics(d):
    for low in itertools.product(range(q), repeat=d):
        yield list(low) + [1]


def irreducible(a):
    a = trim(a); d = len(a) - 1
    if d < 1: return False
    for k in range(1, d // 2 + 1):
        for m in monics(k):
            if not pmod(a, m): return False
    return True


res = {}
for n in (3, 4):
    fails = []; rs = []
    for F in monics(n):
        r = 0
        for G in itertools.product(range(q), repeat=n):
            if not any(G): continue
            if not irreducible(list(G)): continue
            H = [(F[i] - (G[i] if i < n else 0)) % q for i in range(n + 1)]
            if irreducible(H): r += 1
        rs.append(r)
        if r == 0: fails.append(F)
    res[n] = dict(mean_r=float(np.mean(rs)), min_r=int(min(rs)), n_fail=len(fails), fails=fails)
    print(n, res[n])
json.dump(res, open("out/brute_check.json", "w"), indent=1)
