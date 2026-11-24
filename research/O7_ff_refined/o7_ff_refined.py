"""O7 (preregistered): refined local model of chi in F_q[t] (function-field analogue of O6), exhaustive over all
monic F.  Relative pair factors from irreducible pi of degree > d0 dividing F ('div'; primary), and also from all
pi of degree > d0 ('all').  Divisibility fractions w are exact counts.  See DESIGN.md.  Usage: python o7_ff_refined.py [--smoke]"""
import json, math, os, sys, time
import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv
if SMOKE:
    PLAN = [(3, [5, 6, 7, 8]), (5, [5, 6])]
else:
    PLAN = [(3, list(range(5, 16))), (5, list(range(5, 11))), (7, list(range(5, 9))), (11, [5, 6]), (13, [5, 6]),
            (17, [5, 6]), (19, [5]), (23, [5]), (29, [5]), (31, [5])]
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


def nullv_rows(P4):
    """null variance coefficient g'Sigma g for each row of class proportions P4 (n x 4)."""
    c = np.array([1, -1, -1, 1.0]); b = np.array([-1, 1, -1, 1.0])
    cb = P4 @ c; bb = P4 @ b
    scc = P4 @ (c * c) - cb**2; scb = P4 @ (c * b) - cb * bb; sbb = P4 @ (b * b) - bb**2
    g2 = -2 * bb
    return scc + 2 * g2 * scb + g2**2 * sbb


def chi_of(PP, PS, SP, SS):
    A = PP + PS + SP + SS
    return (PP - PS - SP + SS) / A - ((-PP + SS + PS - SP) / A) ** 2


def stats(obs, pred):
    co = obs - obs.mean(); cp = pred - pred.mean()
    return dict(rho=float(np.corrcoef(co, cp)[0, 1]) if cp.std() > 0 else 0.0, sd_ratio=float(cp.std() / co.std()),
                R2_pred=float(1 - np.sum((co - cp) ** 2) / np.sum(co**2)))


rows = []
for q, ns in PLAN:
    for n in ns:
        tq = time.time()
        om, mfd, irr = tables(q, n)
        for k in range(1, n + 1):
            assert len(irr[k]) == n_irr(q, k), (q, k)
        d0 = (n - 1) // 3
        Olow, Mlow = lower_tables(q, n, om, mfd)
        Ptop = ((om[n] == 1) & (mfd[n] > d0)).astype(np.float64); Stop = ((om[n] == 2) & (mfd[n] > d0)).astype(np.float64)
        Plow = ((Olow == 1) & (Mlow > d0)).astype(np.float64); Slow = ((Olow == 2) & (Mlow > d0)).astype(np.float64)
        PP = conv_group(Plow, Ptop, q, n); PS = conv_group(Plow, Stop, q, n)
        SP = conv_group(Slow, Ptop, q, n); SS = conv_group(Slow, Stop, q, n)
        A = PP + PS + SP + SS
        ok = A > 0
        Q = q**n
        chi = np.zeros(Q); chi[ok] = chi_of(PP[ok], PS[ok], SP[ok], SS[ok])
        Nk = {k: len(irr[k]) for k in range(1, n + 1)}
        nPl, nSl, nPt, nSt = Plow.sum(), Slow.sum(), Ptop.sum(), Stop.sum()
        base = np.array([nPl * nPt, nPl * nSt, nSl * nPt, nSl * nSt])
        # exact divisibility fractions w (by symmetry independent of which pi of degree k)
        logg_div = np.zeros((4, Q)); logg_all = np.zeros((4, Q))
        for k in range(d0 + 1, n + 1):
            wt = (Nk[n - k] if n - k > d0 else 0) / nSt if nSt > 0 else 0.0
            wl = sum((q - 1) * Nk[d - k] for d in range(k + d0 + 1, n) if d - k >= 1) / nSl if nSl > 0 else 0.0
            if wt == 0 and wl == 0: continue
            P_ = float(q**k); u = P_ - 1
            div = np.array([1.0, 1 - wt, 1 - wl, u * wl * wt + (1 - wl) * (1 - wt)])
            bPP = P_ * (u - 1) / u**2
            nd = np.array([bPP, P_ * ((u - 1) * (1 - wt) / u**2 + wt / u), P_ * ((u - 1) * (1 - wl) / u**2 + wl / u),
                           P_ * ((u - 1) * (1 - wl) * (1 - wt) / u**2 + (wl * (1 - wt) + wt * (1 - wl)) / u)]) / bPP
            mk = np.zeros(Q)
            mark_batch(mk, q, n, coeff_matrix(irr[k], q, k), k, 1.0, 0)   # number of distinct degree-k irreducible factors
            ld = np.log(np.maximum(div, 1e-300)); ln = np.log(nd)
            logg_div += ld[:, None] * mk[None, :]
            logg_all += ld[:, None] * mk[None, :] + ln[:, None] * (Nk[k] - mk)[None, :]
        res = dict(q=q, n=n, d0=d0, nF=Q, n_ok=int(ok.sum()), counts_low=[int(nPl), int(nSl)], counts_top=[int(nPt), int(nSt)])
        x = chi[ok]
        P4 = np.column_stack([PP[ok], PS[ok], SP[ok], SS[ok]]) / A[ok][:, None]
        s = np.sqrt(nullv_rows(P4) / A[ok])
        res["chi_mean"] = float(x.mean()); res["chi_sd"] = float(x.std())
        res["D_before"] = float(((x - x.mean()) / s).std())
        for name, lg in (("div", logg_div), ("all", logg_all)):
            Xp = base[:, None] * np.exp(lg)          # S(F) and q^-n cancel in chi
            pr = chi_of(*Xp)[ok]
            st = stats(x, pr)
            d = (x - pr) - (x - pr).mean()
            res[name] = dict(st, chi_pred_mean=float(pr.mean()), D_after=float((d / s).std()))
        res["secs"] = time.time() - tq
        rows.append(res)
        print(json.dumps(res), flush=True)
        json.dump(dict(rows=rows, smoke=SMOKE), open(os.path.join(OUT, "o7_smoke.json" if SMOKE else "o7_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
