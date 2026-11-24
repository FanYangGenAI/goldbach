r"""R14: pieces of sum_{gamma in S} r(F_gamma), F_gamma = t(t-1)(t-gamma), S = F_q \ {0,1}; R_111 = q(q-1)/6 * sum."""
import sys, itertools
def run(q):
    sq = {(x * x) % q for x in range(1, q)}
    eta = lambda a: 0 if a % q == 0 else (1 if a % q in sq else -1)
    S = range(2, q)
    n1 = (q - 1) + (q * q - q) // 2
    L = (q - 1) * (q + (q * q - q) // 2)
    sT1 = sum((q - 3) * n1 + 3 * (q - 1) for g in S)
    W = 0; partb = 0; sT2 = 0
    for g in S:
        F = [x * (x - 1) * (x - g) % q for x in range(q)]
        a = sum(1 for x in range(q) for y in range(x + 1, q) if F[x] != F[y])
        b2 = 0
        for x in range(q):
            for y in range(x + 1, q):
                if F[x] and F[y]: b2 += q - eta(F[x] * F[y])
        sT2 += a + b2 // 2
        W += sum(1 for x in range(q) for y in range(q) if x != y and F[x] == F[y])
    # T3 sum and pieces
    sT3 = 0; Nk0 = 0; A = 0; Z0 = 0; Z1 = 0
    for X in itertools.combinations(range(q), 3):
        x, y, z = X
        e1 = (x + y + z) % q; e2 = (x * y + y * z + z * x) % q; e3 = x * y * z % q
        if (e1 - 1) % q not in (0, 1) and (e1 - 1 - e2) % q != 0: Nk0 += 1
        inS = all(v not in (0, 1) for v in X)
        if inS:
            D = 1
            for v in X: D = D * v * (v - 1) % q
            A += (q - eta(D)) // 2
            if eta(e2 * e2 - 4 * (e1 - 1) * e3) == -1: Z0 += 1
            if eta((1 - e2) ** 2 - 4 * (e1 - 2) * e3) == -1: Z1 += 1
        for g in S:
            k = (e1 - 1 - g) % q; m = (g - e2) % q
            ok = (m != 0) if k == 0 else (eta(m * m - 4 * k * e3) == -1)
            sT3 += ok
    tot = (q - 2) * L - sT1 + sT2 - sT3
    return dict(q=q, sT1=sT1, sT2=sT2, W=W, sT3=sT3, Nk0=Nk0, A=A, Z0=Z0, Z1=Z1, check_T3=sT3 == A - Z0 - Z1 + Nk0, sum_r=tot)
for q in [int(a) for a in sys.argv[1:]]:
    print(run(q), flush=True)
