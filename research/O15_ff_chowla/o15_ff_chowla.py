"""O15 (preregistered): Liouville two-point sums over the Goldbach fibre in F_q[t] (a Chowla-type quantity).
For every monic F of degree n: Lam(F) = sum_{G: deg G < n, G != 0} lambda(G) lambda(F - G), lambda = (-1)^Omega of monic(G)
(top side F - G monic of degree n).  Normalised l(F) = Lam(F) / (q^n - 1).  Reports mean, sd * q^{n/2}, and the share of
Var explained by the factorization type of F.  Usage: python o15_ff_chowla.py [--smoke]"""
import json, math, os, sys, time
import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv
SMALLQ = "--smallq" in sys.argv
if SMOKE:
    PLAN = [(3, [6, 8]), (5, [4, 5]), (11, [3, 4])]
else:
    PLAN = [(3, list(range(4, 16))), (5, list(range(3, 11))), (7, list(range(3, 9)))] + [(q, [3, 4, 5]) for q in (11, 13, 17, 19, 23, 29, 31)] +            [(q, [3, 4]) for q in (37, 41, 43)] + [(q, [3]) for q in (53, 61, 71, 83, 97)]
t0 = time.time()


def coeff_matrix(idx, q, k):
    """rows: monic coefficient vectors (len k+1) of the lower-coefficient indices idx."""
    idx = np.asarray(idx, dtype=np.int64).copy()
    C = np.zeros((len(idx), k + 1), dtype=np.int64)
    for i in range(k):
        C[:, i] = idx % q; idx //= q
    C[:, k] = 1
    return C


@nb.njit(cache=True)
def polymul_batch(A, B, q):
    n = A.shape[0]; la = A.shape[1]; lb = B.shape[1]
    out = np.zeros((n, la + lb - 1), dtype=np.int64)
    for r in range(n):
        for i in range(la):
            if A[r, i] == 0: continue
            for j in range(lb):
                out[r, i + j] = (out[r, i + j] + A[r, i] * B[r, j]) % q
    return out


@nb.njit(cache=True)
def mark_batch(arr, q, d, PE, ke, val, mode):
    """for every row pe of PE (monic, degree ke) and every monic g of degree d-ke, at index of pe*g:
    mode 0: arr += val ; mode 1: arr = val if arr == 0."""
    m = d - ke
    total = q ** m
    g = np.zeros(m + 1, dtype=np.int64)
    prod = np.zeros(d + 1, dtype=np.int64)
    for r in range(PE.shape[0]):
        for gi in range(total):
            x = gi
            for i in range(m):
                g[i] = x % q; x //= q
            g[m] = 1
            for i in range(d + 1): prod[i] = 0
            for a in range(ke + 1):
                if PE[r, a] == 0: continue
                for b in range(m + 1):
                    prod[a + b] = (prod[a + b] + PE[r, a] * g[b]) % q
            v = 0
            for i in range(d - 1, -1, -1):
                v = v * q + prod[i]
            if mode == 0:
                arr[v] += val
            elif arr[v] == 0:
                arr[v] = val


def tables(q, n):
    """om[d], mfd[d] (Omega and minimal irreducible-factor degree) for monic polys of degree d <= n; irr[k] index arrays."""
    om = [np.zeros(q**d, dtype=np.int16) for d in range(n + 1)]
    mfd = [np.zeros(q**d, dtype=np.int16) for d in range(n + 1)]
    irr = {}
    for k in range(1, n + 1):
        cand = np.nonzero(om[k] == 0)[0]
        irr[k] = cand
        C = coeff_matrix(cand, q, k)
        PE = C.copy(); e = 1
        while e * k <= n:
            for d in range(e * k, n + 1):
                mark_batch(om[d], q, d, PE, e * k, 1, 0)
                if e == 1: mark_batch(mfd[d], q, d, PE, k, k, 1)
            e += 1
            if e * k <= n: PE = polymul_batch(PE, C, q)
    return om, mfd, irr


def mobius(n):
    m, p, res = n, 2, 1
    while p * p <= m:
        if m % p == 0:
            m //= p
            if m % p == 0: return 0
            res = -res
        p += 1
    return -res if m > 1 else res


def n_irr(q, n):
    return sum(mobius(d) * q ** (n // d) for d in range(1, n + 1) if n % d == 0) // n


def lower_tables(q, n, om, mfd):
    """Omega and min factor degree of g (deg < n, coefficient vector index), via monic(g); g = 0 -> (-1, -1)."""
    Q = q**n
    O = np.full(Q, -1, dtype=np.int16); Mf = np.full(Q, -1, dtype=np.int16)
    inv = np.array([0] + [pow(c, q - 2, q) for c in range(1, q)])
    idx = np.arange(Q, dtype=np.int64)
    coeffs = np.zeros((Q, n), dtype=np.int64)
    for i in range(n):
        coeffs[:, i] = idx % q; idx //= q
    deg = np.full(Q, -1)
    for i in range(n):
        deg[coeffs[:, i] != 0] = i
    for d in range(0, n):
        sel = np.nonzero(deg == d)[0]
        if len(sel) == 0: continue
        invl = inv[coeffs[sel, d]]
        low = (coeffs[sel, :d] * invl[:, None]) % q
        mi = np.zeros(len(sel), dtype=np.int64)
        for i in range(d - 1, -1, -1):
            mi = mi * q + low[:, i]
        O[sel] = om[d][mi] if d > 0 else 0
        Mf[sel] = mfd[d][mi] if d > 0 else 0
    return O, Mf


def conv_group(a, b, q, n):
    shape = (q,) * n
    A = np.fft.fftn(a.reshape(shape)); B = np.fft.fftn(b.reshape(shape))
    return np.rint(np.real(np.fft.ifftn(A * B))).reshape(-1)


def null_var_coef(pPP, pPS, pSP, pSS):
    """per-pair covariance of (c, b) under i.i.d. class labels; returns g' Sigma g with g = (1, -2 bbar)."""
    p = np.array([pPP, pPS, pSP, pSS]); c = np.array([1, -1, -1, 1.0]); b = np.array([-1, 1, -1, 1.0])
    cb, bb = (p * c).sum(), (p * b).sum()
    S = np.array([[(p * c * c).sum() - cb * cb, (p * c * b).sum() - cb * bb], [(p * c * b).sum() - cb * bb, (p * b * b).sum() - bb * bb]])
    g = np.array([1.0, -2 * bb])
    return float(g @ S @ g)


def r2(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return float(1 - res.var() / y.var())



def type_code(q, n, irr):
    """integer code of the factorization type (multiset of irreducible-factor degrees, with multiplicity) of every monic F of degree n."""
    code = np.zeros(q**n, dtype=np.float64)
    for k in range(1, n + 1):
        C = coeff_matrix(irr[k], q, k); PE = C.copy(); e = 1
        while e * k <= n:
            mark_batch(code, q, n, PE, e * k, float((n + 1) ** (k - 1)), 0)
            e += 1
            if e * k <= n: PE = polymul_batch(PE, C, q)
    return code.astype(np.int64)


def decode_type(c, n):
    out = []
    for k in range(1, n + 1):
        m = c % (n + 1); c //= (n + 1)
        out += [k] * m
    return out



rows = []
for q, ns in PLAN:
    for n in ns:
        tq = time.time()
        om, mfd, irr = tables(q, n)
        Olow, _ = lower_tables(q, n, om, mfd)
        lam_top = np.where(om[n] % 2 == 0, 1.0, -1.0)
        lam_low = np.where(Olow < 0, 0.0, np.where(Olow % 2 == 0, 1.0, -1.0))
        Lam = conv_group(lam_low, lam_top, q, n)
        Q = q**n
        l = Lam / (Q - 1)
        tc = type_code(q, n, irr)
        within = 0.0; types = []
        for c in np.unique(tc):
            m = tc == c
            within += m.sum() * l[m].var()
            types.append(dict(type=decode_type(int(c), n), frac=float(m.mean()), mean_l_scaled=float(l[m].mean() * math.sqrt(Q))))
        within /= Q
        row = dict(q=q, n=n, mean_l=float(l.mean()), mean_l_scaled=float(l.mean() * math.sqrt(Q)), sd_l_scaled=float(l.std() * math.sqrt(Q)),
                   max_abs_l=float(np.abs(l).max()), R2_type=float(1 - within / l.var()) if l.var() > 0 else None,
                   mean_lam_top=float(lam_top.mean()), mean_lam_low=float(lam_low.sum() / (Q - 1)), types=types, secs=time.time() - tq)
        rows.append(row)
        print(json.dumps({k_: (round(v, 5) if isinstance(v, float) else v) for k_, v in row.items() if k_ != "types"}), flush=True)
        json.dump(dict(rows=rows, smoke=SMOKE), open(os.path.join(OUT, "o15_smoke.json" if SMOKE else "o15_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
