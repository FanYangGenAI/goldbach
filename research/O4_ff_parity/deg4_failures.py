"""O3 follow-up (post-hoc): why do the three degree-4 Goldbach failures in F_3[t] fail?
For each failing F and each admissible G (c * monic irreducible, deg G < 4), record why F - G is reducible:
'root' (has a root in F_3) or 'quad x quad' (no root, product of two irreducible quadratics)."""
import itertools, json, os

q = 3
HERE = os.path.dirname(os.path.abspath(__file__))


def ev(f, a): return sum(c * a**i for i, c in enumerate(f)) % q


def mul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b): out[i + j] = (out[i + j] + x * y) % q
    return out


def monic(k): return [list(low) + [1] for low in itertools.product(range(q), repeat=k)]


def is_irr(f):
    k = len(f) - 1
    if k == 1: return True
    if any(ev(f, a) == 0 for a in range(q)): return False
    if k <= 3: return True
    return tuple(f) not in QQ


IQ = [f for f in monic(2) if all(ev(f, a) for a in range(q))]
QQ = {tuple(mul(a, b)) for a in IQ for b in IQ}
Gs = []
for k in (1, 2, 3):
    for g in monic(k):
        if is_irr(g):
            for c in (1, 2): Gs.append([c * x % q for x in g])
fails = [[2, 0, 2, 0, 1], [2, 2, 2, 1, 1], [2, 1, 2, 2, 1]]
out = []
for F in fails:
    vals = [ev(F, a) for a in range(q)]
    reasons = {"root": 0, "quad x quad": 0, "irreducible": 0}; exceptional = []
    for G in Gs:
        H = [(F[i] - (G[i] if i < len(G) else 0)) % q for i in range(5)]
        if any(ev(H, a) == 0 for a in range(q)): reasons["root"] += 1
        elif tuple(H) in QQ: reasons["quad x quad"] += 1; exceptional.append(dict(G=G, F_minus_G=H))
        else: reasons["irreducible"] += 1
    rec = dict(F=F, values_on_F3=vals, n_G=len(Gs), reasons=reasons, exceptional=exceptional)
    out.append(rec); print(json.dumps(rec))
json.dump(out, open(os.path.join(HERE, "out", "deg4_failures.json"), "w"), indent=1)
