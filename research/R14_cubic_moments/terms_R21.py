"""R14: inclusion-exclusion terms of r(F0), F0 = t^2 (t-1), for primes q; prints exact values to guide closed forms.
r(F0) = |L| - T1 + T2 - T3 with
 T1 = sum_x nL1(F0(x)),  T2 = sum_{x<y} nL2(x,y,F0(x),F0(y)),  T3 = #{3-subsets X : F0 - prod_{x in X}(t-x) in L}."""
import sys

def run(q):
    sq = {(x * x) % q for x in range(1, q)}
    eta = lambda a: 0 if a % q == 0 else (1 if a % q in sq else -1)
    inv = lambda a: pow(a, q - 2, q)
    F0 = lambda x: (x * x * (x - 1)) % q
    def inL(c2, c1, c0):
        c2 %= q; c1 %= q; c0 %= q
        if c2 == 0: return c1 != 0
        return eta(c1 * c1 - 4 * c2 * c0) == -1
    Lsize = (q - 1) * (q + (q * q - q) // 2)
    n1 = lambda v: (q - 1) if v % q == 0 else (q - 1) + (q * q - q) // 2
    T1 = sum(n1(F0(x)) for x in range(q))
    T2 = 0
    for x in range(q):
        for y in range(x + 1, q):
            u, v = F0(x), F0(y)
            cnt = 1 if u != v else 0                    # linear interpolant c(t+beta), c != 0
            if u and v:
                # quadratics c t^2 + d t + e through (x,u),(y,v), c != 0, irreducible
                for c in range(1, q):
                    # d,e from: c x^2 + d x + e = u, c y^2 + d y + e = v
                    d = ((v - u) - c * (y * y - x * x)) * inv(y - x) % q
                    e = (u - c * x * x - d * x) % q
                    if eta(d * d - 4 * c * e) == -1: cnt += 1
            T2 += cnt
    T3 = 0
    for x in range(q):
        for y in range(x + 1, q):
            for z in range(y + 1, q):
                e1 = x + y + z; e2 = x * y + y * z + z * x; e3 = x * y * z
                if inL(e1 - 1, -e2, e3): T3 += 1
    r = Lsize - T1 + T2 - T3
    return dict(q=q, L=Lsize, T1=T1, T2=T2, T3=T3, r=r,
                r_pred=(q * (q - 1) * (q + 2) // 6 if q % 3 == 1 else (q + 1) * (q * q - 2) // 6))

for q in [int(a) for a in sys.argv[1:]]:
    print(run(q), flush=True)
