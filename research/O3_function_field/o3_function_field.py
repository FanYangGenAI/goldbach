"""O3 (exploratory): Goldbach problem in F_q[t].  For every monic F of degree n, count
r(F) = #{G : deg G < n, G != 0, G = c * (monic irreducible), F - G monic irreducible of degree n},
compare with an HL-type singular-series prediction, list Goldbach failures (r(F) = 0), and compute
Omega-sector tables / entanglement-entropy analogues.  See DESIGN.md.  Usage: python o3_function_field.py [--smoke]"""
import json, math, os, sys, time
import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SMOKE = "--smoke" in sys.argv
PLAN = [(3, list(range(3, 9))), (5, list(range(3, 6)))] if SMOKE else [(3, list(range(3, 16))), (5, list(range(3, 11))), (7, list(range(3, 9)))]
t0 = time.time()


@nb.njit(cache=True)
def decode(idx, q, m, out):
    for i in range(m):
        out[i] = idx % q; idx //= q


@nb.njit(cache=True)
def encode(c, q, m):
    v = 0
    for i in range(m - 1, -1, -1):
        v = v * q + c[i]
    return v


@nb.njit(parallel=True, cache=True)
def mark(om_d, q, d, pe, ke):
    """for every monic g of degree d-ke: om_d[index of lower coeffs of pe*g] += 1 ; pe = monic coeff array (len ke+1)."""
    m = d - ke
    total = q ** m
    for gi in nb.prange(total):
        g = np.zeros(m + 1, dtype=np.int64)
        decode(gi, q, m, g); g[m] = 1
        prod = np.zeros(d + 1, dtype=np.int64)
        for a in range(ke + 1):
            if pe[a] == 0: continue
            for b in range(m + 1):
                prod[a + b] = (prod[a + b] + pe[a] * g[b]) % q
        om_d[encode(prod, q, d)] += 1


@nb.njit(cache=True)
def mark_add(arr, q, d, pe, ke, val):
    """arr[index of pe*g] += val for every monic g of degree d-ke (serial; used for divisor corrections)."""
    m = d - ke
    total = q ** m
    g = np.zeros(m + 1, dtype=np.int64)
    prod = np.zeros(d + 1, dtype=np.int64)
    for gi in range(total):
        decode(gi, q, m, g); g[m] = 1
        for i in range(d + 1): prod[i] = 0
        for a in range(ke + 1):
            if pe[a] == 0: continue
            for b in range(m + 1):
                prod[a + b] = (prod[a + b] + pe[a] * g[b]) % q
        arr[encode(prod, q, d)] += val


def polymul(a, b, q):
    out = np.zeros(len(a) + len(b) - 1, dtype=np.int64)
    for i, x in enumerate(a):
        if x: out[i : i + len(b)] = (out[i : i + len(b)] + x * b) % q
    return out


def omega_tables(q, n):
    om = [np.zeros(q**d, dtype=np.int8) for d in range(n + 1)]
    irr = {}
    for k in range(1, n + 1):
        cand = np.nonzero(om[k] == 0)[0]
        irr[k] = cand
        for ci in cand:
            c = np.zeros(k + 1, dtype=np.int64); decode(int(ci), q, k, c); c[k] = 1
            pe = c.copy(); e = 1
            while e * k <= n:
                for d in range(e * k, n + 1):
                    mark(om[d], q, d, pe, e * k)
                e += 1
                if e * k <= n: pe = polymul(pe, c, q)
    return om, irr


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


def lower_index_table(q, n, om):
    """for every g in F_q^n (deg < n, coefficient vector), Omega of g (=Omega of monic(g)), or -1 for g = 0."""
    Q = q**n
    out = np.full(Q, -1, dtype=np.int8)
    inv = [0] + [pow(c, q - 2, q) for c in range(1, q)]
    coeffs = np.zeros((Q, n), dtype=np.int64)
    idx = np.arange(Q, dtype=np.int64)
    for i in range(n):
        coeffs[:, i] = idx % q; idx //= q
    deg = np.full(Q, -1)
    for i in range(n):
        deg[coeffs[:, i] != 0] = i
    for d in range(0, n):
        sel = np.nonzero(deg == d)[0]
        if len(sel) == 0: continue
        lead = coeffs[sel, d]
        invl = np.array(inv)[lead]
        low = (coeffs[sel, :d] * invl[:, None]) % q
        mi = np.zeros(len(sel), dtype=np.int64)
        for i in range(d - 1, -1, -1):
            mi = mi * q + low[:, i]
        out[sel] = om[d][mi] if d > 0 else 0
    return out


def conv_group(a, b, q, n):
    """(a * b)(v) = sum_g a(g) b(v - g) over (Z/q)^n, arrays indexed by base-q digits (digit i = axis n-1-i in C order)."""
    shape = (q,) * n
    A = np.fft.fftn(a.reshape(shape)); B = np.fft.fftn(b.reshape(shape))
    return np.rint(np.real(np.fft.ifftn(A * B))).reshape(-1)


def singular_series(q, n, irr):
    """S(F) for every monic F of degree n: prod_{pi | F} |pi|/(|pi|-1) * prod_{pi not| F} (1 - 1/(|pi|-1)^2), pi up to degree n."""
    Q = q**n
    logS = np.zeros(Q)
    base = 0.0
    for k in range(1, n + 1):
        P = q**k
        base += len(irr[k]) * math.log(1 - 1 / (P - 1) ** 2)
    logS += base
    return logS


rows = []
for q, ns in PLAN:
    for n in ns:
        tq = time.time()
        om, irr = omega_tables(q, n)
        for k in range(1, n + 1):
            assert len(irr[k]) == n_irr(q, k), (q, k, len(irr[k]), n_irr(q, k))
        In = (om[n] == 1).astype(np.float64)                  # monic deg n irreducible (index = lower coeffs)
        Olow = lower_index_table(q, n, om)                    # Omega of g (deg < n), -1 for 0
        Ilow = (Olow == 1).astype(np.float64)
        r = conv_group(Ilow, In, q, n)                        # r[v_F]
        # singular series: base product + divisor corrections via sieve over irreducibles pi | F
        logS = singular_series(q, n, irr)
        for k in range(1, n + 1):
            P = q**k
            corr = math.log(P / (P - 1)) - math.log(1 - 1 / (P - 1) ** 2)
            for ci in irr[k]:
                c = np.zeros(k + 1, dtype=np.int64); decode(int(ci), q, k, c); c[k] = 1
                # monic F of degree n divisible by pi: F = pi * g, g monic deg n-k (each such F hit once)
                mark_add(logS, q, n, c, k, corr)
        S = np.exp(logS)
        pin = len(irr[n]); nlow = int(Ilow.sum())
        pred = pin * nlow / q**n * S
        rel = r / pred
        fails = np.nonzero(r == 0)[0]
        row = dict(q=q, n=n, nF=q**n, pi_n=pin, n_irr_lower=nlow, mean_r=float(r.mean()), min_r=float(r.min()),
                   n_fail=int(len(fails)), fail_examples=[int(x) for x in fails[:10]],
                   rel_mean=float(rel.mean()), rel_sd=float(rel.std()), rel_min=float(rel.min()), rel_max=float(rel.max()),
                   corr_r_S=float(np.corrcoef(r, S)[0, 1]) if r.std() > 0 else None)
        # sector tables for Omega 1..3 on both sides, and amplitude-encoding entanglement entropy for a sample of F
        if q**n <= 6 * 10**6:
            K = 4
            Tj = {}
            for j in range(1, K):
                aj = (Olow == j).astype(np.float64)
                for kk in range(1, K):
                    bk = (om[n] == kk).astype(np.float64)
                    Tj[(j, kk)] = conv_group(aj, bk, q, n)
            rng = np.random.default_rng(1000 * q + n)
            samp = rng.choice(q**n, size=min(200, q**n), replace=False)
            ents = []; c11 = []
            for v in samp:
                T = np.array([[Tj[(j, kk)][v] for kk in range(1, K)] for j in range(1, K)], float)
                if T.sum() == 0: continue
                P = T / T.sum()
                sv = np.linalg.svd(np.sqrt(P), compute_uv=False) ** 2; sv = sv[sv > 0]
                ents.append(float(-(sv * np.log(sv)).sum()))
                rr_ = P.sum(1); cc_ = P.sum(0)
                if rr_[0] > 0 and cc_[0] > 0: c11.append(float(P[0, 0] / (rr_[0] * cc_[0]) / S[v]))
            row["entropy_median"] = float(np.median(ents)) if ents else None
            row["C11_over_S_median"] = float(np.median(c11)) if c11 else None
        row["secs"] = time.time() - tq
        rows.append(row)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items() if k != "fail_examples"}), flush=True)
        json.dump(dict(rows=rows, smoke=SMOKE), open(os.path.join(OUT, "o3_smoke.json" if SMOKE else "o3_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
