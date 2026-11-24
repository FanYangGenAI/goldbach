"""O11 post-hoc (exploratory): how much of the residual chi_obs - chi_ref is the Goldbach noise of N/p?
For p | N, p > z, SS pairs sharing the factor p are exactly the pairs (p m, p m') with m + m' = N/p, m, m' primes > z,
i.e. r_z(N/p).  The refined model replaces this count by its local expectation S(N) int rho2 rho2 p w w'.  The 'oracle'
model uses the true r_z(N/p) instead (everything else unchanged).  Compare residual dispersion D and correlation."""
import json, math, os
import numpy as np
from scipy import integrate
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "out", "o11_summary.json")))
C2 = 0.6601618158468695739278121100145


def sieve(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if s[i]: s[i * i :: i] = False
    return s


def opf(N):
    ps = []; m = N
    while m % 2 == 0: m //= 2
    p = 3
    while p * p <= m:
        if m % p == 0:
            ps.append(p)
            while m % p == 0: m //= p
        p += 2
    if m > 1: ps.append(m)
    return ps


def chi_from(pp, ps, sp, ss):
    A = pp + ps + sp + ss; B = -pp + ss + (ps - sp); C = pp - ps - sp + ss
    return C / A - (B / A) ** 2


def nullv(p):
    c = np.array([1, -1, -1, 1.0]); b = np.array([-1, 1, -1, 1.0])
    cb, bb = (p * c).sum(), (p * b).sum()
    S = np.array([[(p * c * c).sum() - cb * cb, (p * c * b).sum() - cb * bb], [(p * c * b).sum() - cb * bb, (p * b * b).sum() - bb * bb]])
    g = np.array([1.0, -2 * bb]); return float(g @ S @ g)


def model_counts(N, z, big):
    """refined (O6) class integrals, plus the 'shared' SS part for each p (S(N) factor omitted: cancels in chi except
    against the oracle count, so it is applied explicitly)."""
    lz = math.log(z); S = 2 * C2
    for p in opf(N): S *= (p - 1) / (p - 2)
    def r1(t): return 1 / math.log(t)
    def r2(t):
        L = math.log(t)
        return 0.0 if L <= 2 * lz else math.log((L - lz) / lz) / L
    def w(p, t):
        if t / p <= z: return 0.0
        d = r2(t)
        return 0.0 if d <= 0 else min(1.0, 1 / (p * math.log(t / p)) / d)
    lo, hi = z + 1, N - z - 1
    pts = sorted({min(max(x, lo), hi) for p in big for x in (p * z, N - p * z)} | {N / 2})
    edges = [lo] + [x for x in pts if lo < x < hi] + [hi]
    def integ(f):
        return sum(integrate.quad(f, a, b, limit=400)[0] for a, b in zip(edges[:-1], edges[1:]))
    def g(kind, t):
        v = 1.0
        for p in big:
            a, b = w(p, t), w(p, N - t)
            if kind == 1: v *= 1 - b
            elif kind == 2: v *= 1 - a
            elif kind == 3: v *= (p - 1) * a * b + (1 - a) * (1 - b)
        return v
    PP = S * integ(lambda t: r1(t) * r1(N - t))
    PS = S * integ(lambda t: r1(t) * r2(N - t) * g(1, t))
    SP = S * integ(lambda t: r2(t) * r1(N - t) * g(2, t))
    SS = S * integ(lambda t: r2(t) * r2(N - t) * g(3, t))
    shared = {p: S * integ(lambda t, p=p: r2(t) * r2(N - t) * (p - 1) * w(p, t) * w(p, N - t)) for p in big}
    return PP, PS, SP, SS, shared


out = {}
for X in sorted({r["X"] for r in D["rows"]}):
    rs = [r for r in D["rows"] if r["X"] == X]
    lim = max(r["N"] // min(r["big"]) for r in rs if r["big"]) + 10 if any(r["big"] for r in rs) else 10
    isp = sieve(lim)
    obs, ref, orc, sres = [], [], [], []
    for r in rs:
        N, z, big = r["N"], r["z"], r["big"]
        PP, PS, SP, SS, shared = model_counts(N, z, big)
        SSo = SS
        for p in big:
            M = N // p
            if M <= 2 * z: true = 0
            else:
                m = np.arange(z + 1, M - z)
                true = int((isp[m] & isp[M - m]).sum())
            SSo += true - shared[p]
        obs.append(r["chi_obs"]); ref.append(chi_from(PP, PS, SP, SS)); orc.append(chi_from(PP, PS, SP, max(SSo, 0.0)))
        cnt = np.array(r["counts"], float); A = cnt.sum()
        sres.append(math.sqrt(nullv(cnt / A) / A))
    obs, ref, orc, sres = map(np.array, (obs, ref, orc, sres))
    def D_after(pred):
        d = (obs - pred) - (obs - pred).mean(); return float((d / sres).std())
    def R2(pred):
        co = obs - obs.mean(); cp = pred - pred.mean(); return float(1 - np.sum((co - cp) ** 2) / np.sum(co**2))
    out[str(X)] = dict(D_before=float(((obs - obs.mean()) / sres).std()), D_after_ref=D_after(ref), D_after_oracle=D_after(orc),
                       R2_ref=R2(ref), R2_oracle=R2(orc), corr_ref=float(np.corrcoef(obs, ref)[0, 1]), corr_oracle=float(np.corrcoef(obs, orc)[0, 1]))
    print(X, {k: round(v, 4) for k, v in out[str(X)].items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "out", "oracle_shared.json"), "w"), indent=1)
