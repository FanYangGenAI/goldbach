"""O10 (exploratory): large-q behaviour of the Goldbach count r(F) in F_q[t] at fixed degree n.
Is r(F) (to leading order) a function of the factorization type of F?  Reports c^2, the share of Var(r)
explained by the factorization type, and by the HL prediction.  Usage: python o10_ff_largeq.py [--smoke]"""
import json, math, os, sys, time
import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv
SMALLQ = "--smallq" in sys.argv
if SMALLQ:
    PLAN = [(3, list(range(4, 15))), (5, [6, 7, 8, 9]), (7, [6, 7])]
elif SMOKE:
    PLAN = [(5, [3, 4]), (7, [3])]
else:
    PLAN = [(q, [3]) for q in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61)] +            [(q, [4]) for q in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43)] + [(q, [5]) for q in (5, 7, 11, 13, 17, 19, 23, 29, 31)]
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
        for k in range(1, n + 1):
            assert len(irr[k]) == n_irr(q, k), (q, k)
        Olow, _ = lower_tables(q, n, om, mfd)
        In = (om[n] == 1).astype(np.float64); Ilow = (Olow == 1).astype(np.float64)
        r = conv_group(Ilow, In, q, n)
        # HL prediction (as O3)
        Q = q**n
        logS = np.zeros(Q) + sum(len(irr[k]) * math.log(1 - 1 / (q**k - 1) ** 2) for k in range(1, n + 1))
        for k in range(1, n + 1):
            P_ = q**k
            mark_batch(logS, q, n, coeff_matrix(irr[k], q, k), k, math.log(P_ / (P_ - 1)) - math.log(1 - 1 / (P_ - 1) ** 2), 0)
        pred = len(irr[n]) * Ilow.sum() / Q * np.exp(logS)
        tc = type_code(q, n, irr)
        V = r.var(); within = 0.0; types = []
        e = r - pred; Ve = e.var(); within_e = 0.0
        for c in np.unique(tc):
            m = tc == c
            within += m.sum() * r[m].var(); within_e += m.sum() * e[m].var()
            types.append(dict(type=decode_type(int(c), n), frac=float(m.mean()), mean_rel=float(r[m].mean() / r.mean() - 1),
                              var_over_mean=float(r[m].var() / r.mean()), hl_resid_mean=float(e[m].mean() / math.sqrt(r.mean())),
                              hl_resid_var_over_mean=float(e[m].var() / r.mean())))
        within /= Q; within_e /= Q
        row = dict(q=q, n=n, mean_r=float(r.mean()), c2_raw=float(V / r.mean()), c2_hl=float((r - pred).var() / r.mean()),
                   R2_type=float(1 - within / V), R2_type_of_hl_resid=float(1 - within_e / Ve), c2_hl_within_type=float(within_e / r.mean()), R2_hl=float(1 - (r - pred).var() / V), c2_within_type=float(within / r.mean()),
                   types=types, secs=time.time() - tq)
        rows.append(row)
        print(json.dumps({k_: (round(v, 5) if isinstance(v, float) else v) for k_, v in row.items() if k_ != "types"}), flush=True)
        json.dump(dict(rows=rows, smoke=SMOKE), open(os.path.join(OUT, "o10_smoke.json" if SMOKE else ("o10_smallq.json" if SMALLQ else "o10_summary.json")), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
