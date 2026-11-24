"""O4 (preregistered): the signed parity two-point quantity chi (O1 / round-12 review) in F_q[t].
For every monic F of degree n, on pairs (G, F-G) with deg G < n, both sides 'rough' (every irreducible factor
of degree > d0 = floor((n-1)/3)), classify each side as P (irreducible) or S (two irreducible factors), count
PP, PS, SP, SS by FFT over (Z/q)^n, and form A, B, C, mu = B/A, chi = C/A - mu^2 exactly as in O1.
See DESIGN.md.  Usage: python o4_ff_parity.py [--smoke]"""
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


rows = []
for q, ns in PLAN:
    for n in ns:
        tq = time.time()
        om, mfd, irr = tables(q, n)
        for k in range(1, n + 1):
            assert len(irr[k]) == n_irr(q, k), (q, k)
        d0 = (n - 1) // 3              # rough: every irreducible factor has degree > d0 (avoids empty S_low when 3 | n)
        Olow, Mlow = lower_tables(q, n, om, mfd)
        Ptop = ((om[n] == 1) & (mfd[n] > d0)).astype(np.float64); Stop = ((om[n] == 2) & (mfd[n] > d0)).astype(np.float64)
        Plow = ((Olow == 1) & (Mlow > d0)).astype(np.float64); Slow = ((Olow == 2) & (Mlow > d0)).astype(np.float64)
        PP = conv_group(Plow, Ptop, q, n); PS = conv_group(Plow, Stop, q, n)
        SP = conv_group(Slow, Ptop, q, n); SS = conv_group(Slow, Stop, q, n)
        A = PP + PS + SP + SS; B = -PP + SS + (PS - SP); C = PP - PS - SP + SS
        ok = A > 0
        mu = np.where(ok, B / np.where(ok, A, 1), 0); chi = np.where(ok, C / np.where(ok, A, 1) - mu**2, 0)
        # local features of F: log S(F), number of distinct irreducible factors of degree <= d0 and > d0, irreducible
        Q = q**n
        logS = np.zeros(Q); wsm = np.zeros(Q); wbg = np.zeros(Q)
        base = sum(len(irr[k]) * math.log(1 - 1 / (q**k - 1) ** 2) for k in range(1, n + 1))
        logS += base
        for k in range(1, n + 1):
            P_ = q**k
            corr = math.log(P_ / (P_ - 1)) - math.log(1 - 1 / (P_ - 1) ** 2)
            Ck = coeff_matrix(irr[k], q, k)
            mark_batch(logS, q, n, Ck, k, corr, 0)
            mark_batch(wsm if k <= d0 else wbg, q, n, Ck, k, 1.0, 0)
        irrF = (om[n] == 1).astype(np.float64)
        # base local model: X_pred = S(F) * #X_low * #Y_top / q^n  -> chi_pred is F-independent
        nP_l, nS_l, nP_t, nS_t = Plow.sum(), Slow.sum(), Ptop.sum(), Stop.sum()
        w = np.array([nP_l * nP_t, nP_l * nS_t, nS_l * nP_t, nS_l * nS_t]); pcl = w / w.sum()
        Cp = pcl[0] - pcl[1] - pcl[2] + pcl[3]; Bp = -pcl[0] + pcl[3] + pcl[1] - pcl[2]
        chi_pred = Cp - Bp**2
        v1 = null_var_coef(*pcl)
        x = chi[ok]; a = A[ok]
        z = (x - x.mean()) / np.sqrt(v1 / a)
        Apred = np.exp(logS) * w.sum() / q**n
        row = dict(q=q, n=n, d0=d0, nF=Q, n_ok=int(ok.sum()), counts_low=[int(nP_l), int(nS_l)], counts_top=[int(nP_t), int(nS_t)],
                   mean_A=float(a.mean()), A_obs_over_pred=float((A[ok] / Apred[ok]).mean()), corr_A_S=float(np.corrcoef(A[ok], Apred[ok])[0, 1]),
                   chi_mean=float(x.mean()), chi_pred=float(chi_pred), chi_sd=float(x.std()), mu_mean=float(mu[ok].mean()),
                   mu_pred=float(Bp), D=float(z.std()),
                   corr_chi_logS=float(np.corrcoef(x, logS[ok])[0, 1]),
                   R2_local=r2(x, [logS[ok], wsm[ok], wbg[ok], irrF[ok]]),
                   R2_local_z=r2(z, [logS[ok], wsm[ok], wbg[ok], irrF[ok]]),
                   R2_each=[r2(x, [f[ok]]) if f[ok].std() > 0 else 0.0 for f in (logS, wsm, wbg, irrF)])
        row["secs"] = time.time() - tq
        rows.append(row)
        print(json.dumps({k_: (round(v, 6) if isinstance(v, float) else v) for k_, v in row.items()}), flush=True)
        if SMOKE and q == 3 and n == 6:
            # brute check of the four counts for a few F using om/mfd (independent pairwise loop)
            rng = np.random.default_rng(1)
            def cls(Ov, Mv):
                return 0 if (Ov == 1 and Mv > d0) else (1 if (Ov == 2 and Mv > d0) else -1)
            for v in rng.choice(Q, 5, replace=False):
                cnt = np.zeros(4)
                vd = [(v // q**i) % q for i in range(n)]
                for gi in range(Q):
                    gd = [(gi // q**i) % q for i in range(n)]
                    hd = [(vd[i] - gd[i]) % q for i in range(n)]
                    hi = sum(hd[i] * q**i for i in range(n))
                    cl, ct = cls(Olow[gi], Mlow[gi]), cls(om[n][hi], mfd[n][hi])
                    if cl >= 0 and ct >= 0: cnt[2 * cl + ct] += 1
                assert np.allclose(cnt, [PP[v], PS[v], SP[v], SS[v]]), (v, cnt, PP[v], PS[v], SP[v], SS[v])
            print("brute check of PP/PS/SP/SS ok", flush=True)
            import itertools
            def pdiv(f, g):
                a = f[:]; quo = [0] * (len(f) - len(g) + 1)
                while len(a) >= len(g):
                    co = a[-1]; s = len(a) - len(g); quo[s] = co
                    for i, xx in enumerate(g): a[s + i] = (a[s + i] - co * xx) % q
                    a.pop()
                return quo, a
            def fdeg(f):
                degs = []
                while len(f) - 1 >= 1:
                    nn = len(f) - 1; hit = None
                    for d in range(1, nn // 2 + 1):
                        for low in itertools.product(range(q), repeat=d):
                            quo, rem = pdiv(f, list(low) + [1])
                            if not any(rem): hit = (d, quo); break
                        if hit: break
                    if hit is None: degs.append(nn); break
                    degs.append(hit[0]); f = hit[1]
                return degs
            for v in rng.choice(Q, 60, replace=False):
                f = [int((v // q**i) % q) for i in range(n)] + [1]
                dg = fdeg(f)
                assert len(dg) == om[n][v] and min(dg) == mfd[n][v], (f, dg, om[n][v], mfd[n][v])
            print("trial-division check of Omega/min-degree ok", flush=True)
        json.dump(dict(rows=rows, smoke=SMOKE), open(os.path.join(OUT, "o4_smoke.json" if SMOKE else "o4_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
