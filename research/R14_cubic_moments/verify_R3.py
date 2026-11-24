"""R14: brute-force verification of every lemma in proof_R3.md for odd primes 5 <= q < 110 (q != 3); direct r(t^3) for q <= 60.
L = nonzero constant multiples of monic irreducibles of degree 1, 2; r(F) = #{G in L : F - G monic irreducible cubic}."""
import json, os

def primes(n):
    return [p for p in range(5, n) if all(p % d for d in range(2, int(p**0.5) + 1))]

def has_root(c, q):          # monic cubic t^3 + c2 t^2 + c1 t + c0, c = (c0, c1, c2)
    c0, c1, c2 = c
    return any((x * x * x + c2 * x * x + c1 * x + c0) % q == 0 for x in range(q))

out = {}
for q in primes(110):
    sq = {(x * x) % q for x in range(1, q)}
    eta = lambda a: 0 if a % q == 0 else (1 if a % q in sq else -1)
    # irreducible monic cubics with h2 = 1, as a set of (h0, h1)
    roots_tab = {}
    irr1 = set()
    for h1 in range(q):
        for h0 in range(q):
            if not has_root((h0, h1, 1), q): irr1.add((h0, h1))
    I = len(irr1)
    S = sum(eta(h1 * h1 - 4 * h0) for h0, h1 in irr1)
    E0 = sum(1 for h0, h1 in irr1 if (h1 * h1 - 4 * h0) % q == 0)
    M31 = sum(1 for c in range(q) if not has_root((1, c, 3), q))
    # A: irreducible with h2 = 0, h1 != 0
    A = sum(1 for h1 in range(1, q) for h0 in range(q) if not has_root((h0, h1, 0), q))
    # r(t^3) directly: G = -(h2 t^2 + h1 t + h0) in L  <=>  (h2 = 0, h1 != 0) or (h2 != 0 and disc nonsquare)
    r = None if q > 60 else 0
    for h2 in (range(q) if q <= 60 else []):
        for h1 in range(q):
            for h0 in range(q):
                inL = (h2 == 0 and h1 != 0) or (h2 != 0 and eta(h1 * h1 - 4 * h2 * h0) == -1)
                if inL and not has_root((h0, h1, h2), q): r += 1
    one = q % 3 == 1
    pred = dict(I=(q * q - 1) // 3, S=0, E0=(q - 1) // 3 if one else (q + 1) // 3, M31=(q - 1) // 3 if one else (q + 1) // 3,
                A=(q - 1) ** 2 // 3 if one else (q * q - 1) // 3,
                r=(q - 1) ** 2 * (q + 2) // 6 if one else q * (q * q - 1) // 6)
    if r is None: pred.pop("r")
    got = dict(I=I, S=S, E0=E0, M31=M31, A=A) if r is None else dict(I=I, S=S, E0=E0, M31=M31, A=A, r=r)
    ok = got == pred
    out[q] = dict(got=got, pred=pred, ok=ok)
    print(q, q % 3, ok, got, flush=True)
    assert ok, (q, got, pred)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_R3.json"), "w"), indent=1)
print("all lemmas verified")
