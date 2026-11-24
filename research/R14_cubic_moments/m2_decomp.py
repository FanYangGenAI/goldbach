"""R14: decompose M2 = sum_h A_L(h) A_P(h) by affine classes of h (deg h < 3)."""
import sys, numpy as np
def run(q):
    sq = {(x * x) % q for x in range(1, q)}
    eta = lambda a: 0 if a % q == 0 else (1 if a % q in sq else -1)
    nu = next(a for a in range(2, q) if eta(a) == -1)
    Q = q**3
    idx = lambda c0, c1, c2: (c0 % q) + q * (c1 % q) + q * q * (c2 % q)
    Lv = np.zeros(Q); Pv = np.zeros(Q)
    for c in range(1, q):
        for b in range(q): Lv[idx(c * b, c, 0)] = 1
        for d in range(q):
            for e in range(q):
                if eta(d * d - 4 * c * e) == -1: Lv[idx(e, d, c)] = 1
    for c0 in range(q):
        for c1 in range(q):
            for c2 in range(q):
                if all((x**3 + c2 * x * x + c1 * x + c0) % q for x in range(q)): Pv[idx(c0, c1, c2)] = 1
    sh = (q, q, q)
    def acorr(v):
        F = np.fft.fftn(v.reshape(sh)); return np.rint(np.real(np.fft.ifftn(F * np.conj(F)))).reshape(-1)
    AL = acorr(Lv); AP = acorr(Pv)      # AL[h] = sum_G L(G) L(G+h)  (sign convention irrelevant: symmetric)
    M2 = float((AL * AP).sum())
    const = sum(AL[idx(h, 0, 0)] * AP[idx(h, 0, 0)] for h in range(1, q))
    sumAPconst = sum(AP[idx(h, 0, 0)] for h in range(1, q))
    lin = {a: (AL[idx(0, a, 0)], AP[idx(0, a, 0)]) for a in (1, nu)}
    quad = [(e, AL[idx(e, 0, 1)], AP[idx(e, 0, 1)]) for e in range(q)]
    return dict(q=q, M2=M2, h0=float(AL[0] * AP[0]), const=float(const), sumAPconst=float(sumAPconst), ALconst=float(AL[idx(1, 0, 0)]),
                lin={int(k): (float(v[0]), float(v[1])) for k, v in lin.items()},
                quadsum=float(sum(a * b for _, a, b in quad)), quad=[(e, float(a), float(b)) for e, a, b in quad])
for q in [int(a) for a in sys.argv[1:]]:
    d = run(q); print({k: v for k, v in d.items() if k != "quad"}); print("   quad", d["quad"])
