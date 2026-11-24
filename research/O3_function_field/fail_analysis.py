"""O3 post-hoc (exploratory): describe the Goldbach failures F (r(F)=0) found in o3_summary.json.
For each failure: polynomial, irreducible?, invariance under t -> t + a (a in F_q), factorization pattern,
and the rank of S(F) among all monic F of that degree.  Also scaling of rel_sd vs Poisson 1/sqrt(mean_r)."""
import json, os, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "out", "o3_summary.json")))


def poly_from_index(idx, q, n):
    c = []
    for _ in range(n):
        c.append(idx % q); idx //= q
    return c + [1]


def pmod(a, b, q):
    a = a[:]
    inv = pow(b[-1], q - 2, q)
    while len(a) >= len(b) and any(a):
        if a[-1] == 0: a.pop(); continue
        f = a[-1] * inv % q; s = len(a) - len(b)
        for i, x in enumerate(b): a[s + i] = (a[s + i] - f * x) % q
        a.pop()
    while a and a[-1] == 0: a.pop()
    return a


def pdiv(f, g, q):
    a = f[:]; quo = [0] * (len(f) - len(g) + 1)
    inv = pow(g[-1], q - 2, q)
    while len(a) >= len(g):
        co = a[-1] * inv % q; s = len(a) - len(g); quo[s] = co
        for i, x in enumerate(g): a[s + i] = (a[s + i] - co * x) % q
        a.pop()
    return quo, a


def factor_degrees(f, q):
    """degrees of monic irreducible factors (with multiplicity), by trial division (small degree only)."""
    degs = []
    while len(f) - 1 >= 1:
        n = len(f) - 1; hit = None
        for d in range(1, n // 2 + 1):
            for low in itertools.product(range(q), repeat=d):
                g = list(low) + [1]
                quo, rem = pdiv(f, g, q)
                if not any(rem): hit = (d, quo); break
            if hit: break
        if hit is None: degs.append(n); break
        degs.append(hit[0]); f = hit[1]
    return sorted(degs)


def shift(f, a, q):
    """f(t + a)"""
    n = len(f) - 1; out = [0] * (n + 1)
    from math import comb
    for k, c in enumerate(f):
        for j in range(k + 1):
            out[j] = (out[j] + c * comb(k, j) * pow(a, k - j, q)) % q
    return out


def fmt(f):
    terms = [(f"{c}" if i == 0 else (("" if c == 1 else f"{c}") + ("t" if i == 1 else f"t^{i}"))) for i, c in enumerate(f) if c]
    return " + ".join(terms)


out = []
for row in D["rows"]:
    q, n = row["q"], row["n"]
    rec = dict(q=q, n=n, n_fail=row["n_fail"], rel_sd=row["rel_sd"], poisson=row["mean_r"] ** -0.5,
               rel_sd_over_poisson=row["rel_sd"] * row["mean_r"] ** 0.5, fails=[])
    for idx in row["fail_examples"]:
        f = poly_from_index(idx, q, n)
        inv = [a for a in range(1, q) if shift(f, a, q) == f]
        rec["fails"].append(dict(poly=fmt(f), factor_degrees=factor_degrees(f, q), shift_invariant_by=inv))
    out.append(rec)
    print(q, n, "fail", row["n_fail"], "rel_sd*sqrt(mean)", round(rec["rel_sd_over_poisson"], 3), rec["fails"])
json.dump(out, open(os.path.join(HERE, "out", "fail_analysis.json"), "w"), indent=1, ensure_ascii=False)
