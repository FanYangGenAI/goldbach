"""R14: check sum_{beta,delta} r(t^3 + beta t^2 + delta t) = n1 * q (q^2 - 1) / 3 by brute force (small q)."""
def has_root(c0, c1, c2, q): return any((x**3 + c2 * x * x + c1 * x + c0) % q == 0 for x in range(q))
for q in (5, 7, 11, 13):
    sq = {(x * x) % q for x in range(1, q)}
    eta = lambda a: 0 if a % q == 0 else (1 if a % q in sq else -1)
    Lset = [(0, c, (c * b) % q) for c in range(1, q) for b in range(q)]          # (g2, g1, g0) linear
    Lset += [(c, d, e) for c in range(1, q) for d in range(q) for e in range(q) if eta(d * d - 4 * c * e) == -1]
    tot = 0
    for beta in range(q):
        for delta in range(q):
            for g2, g1, g0 in Lset:
                if not has_root(-g0, delta - g1, beta - g2, q): tot += 1
    n1 = (q - 1) * (q + 2) // 2
    print(q, tot, n1 * q * (q * q - 1) // 3, tot == n1 * q * (q * q - 1) // 3, flush=True)
