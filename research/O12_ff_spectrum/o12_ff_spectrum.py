"""O12 (exploratory): Fourier level spectrum of the Goldbach problem in F_q[t].  Additive characters psi_k of the
lower-coefficient space are graded by level h(k) = min{i : k_i != 0} - 1 (Keating-Rudnick short-interval grading).
For each level: mean |P^|^2 / pi_n (KR heuristic (n-h-2)/n), the weight of |L^|^2, and the share of the energy of the
HL residual e = r - r_pred.  Usage: python o12_ff_spectrum.py [--smoke]"""
import json, math, os, sys, time
import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv
if SMOKE:
    PLAN = [(5, [4]), (3, [8])]
else:
    PLAN = [(3, list(range(6, 16))), (5, [4, 5, 6, 7, 8, 9]), (7, [4, 5, 6, 7]), (13, [4, 5]), (23, [4, 5]), (31, [4, 5]), (43, [4])]
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




rows = []
for q, ns in PLAN:
    for n in ns:
        tq = time.time()
        om, mfd, irr = tables(q, n)
        Olow, _ = lower_tables(q, n, om, mfd)
        In = (om[n] == 1).astype(np.float64); Ilow = (Olow == 1).astype(np.float64)
        shape = (q,) * n
        Ph = np.fft.fftn(In.reshape(shape)); Lh = np.fft.fftn(Ilow.reshape(shape))
        r = np.rint(np.real(np.fft.ifftn(Ph * Lh))).reshape(-1)
        Q = q**n
        logS = np.zeros(Q) + sum(len(irr[k]) * math.log(1 - 1 / (q**k - 1) ** 2) for k in range(1, n + 1))
        for k in range(1, n + 1):
            P_ = q**k
            mark_batch(logS, q, n, coeff_matrix(irr[k], q, k), k, math.log(P_ / (P_ - 1)) - math.log(1 - 1 / (P_ - 1) ** 2), 0)
        pred = len(irr[n]) * Ilow.sum() / Q * np.exp(logS)
        e = r - pred; e -= e.mean()
        Eh = np.abs(np.fft.fftn(e.reshape(shape))) ** 2
        # level of each character index: index digit i (coefficient a_i) is axis n-1-i in C order
        idx = np.indices(shape).reshape(n, -1)            # idx[axis]
        dig = idx[::-1]                                   # dig[i] = k_i
        lev = np.full(Q, n - 1)
        for i in range(n - 1, -1, -1):
            lev[dig[i] != 0] = i - 1
        P2 = (np.abs(Ph) ** 2).reshape(-1); L2 = (np.abs(Lh) ** 2).reshape(-1); E2 = Eh.reshape(-1)
        pin = len(irr[n]); Lsz = Ilow.sum()
        levels = []
        for h in range(-1, n - 1):
            m = lev == h
            levels.append(dict(h=h, n_chars=int(m.sum()), P2_over_pin=float(P2[m].mean() / pin), KR=(n - h - 2) / n,
                               L2_weight=float(L2[m].sum() / (Q * Lsz - Lsz**2)), resid_energy_share=float(E2[m].sum() / E2[lev < n - 1].sum())))
        alpha = Lsz / Q
        heur = (1 - alpha) * sum(L['KR'] * L['L2_weight'] for L in levels)
        heur_emp = (1 - alpha) * sum(L['P2_over_pin'] * L['L2_weight'] for L in levels)
        row = dict(q=q, n=n, c2_hl=float(e.var() / r.mean()), c2_raw=float(r.var() / r.mean()), c2_indep_KR=float(heur), c2_indep_emp=float(heur_emp), levels=levels, secs=time.time() - tq)
        rows.append(row)
        print(json.dumps(dict(q=q, n=n, c2_hl=round(row["c2_hl"], 4), c2_raw=round(row["c2_raw"], 3), indepKR=round(heur, 3), indep_emp=round(heur_emp, 3),
                              lv=[(L["h"], round(L["P2_over_pin"], 3), round(L["KR"], 3), round(L["L2_weight"], 3), round(L["resid_energy_share"], 3)) for L in levels])), flush=True)
        json.dump(dict(rows=rows, smoke=SMOKE), open(os.path.join(OUT, "o12_smoke.json" if SMOKE else "o12_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
