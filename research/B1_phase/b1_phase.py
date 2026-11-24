"""B1 phase pilot (see DESIGN.md)."""
import json, math, os, time
import numpy as np
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
t0 = time.time()
M = 1_100_000
L = 2310 * 953
C2 = 0.6601618158468695739278121100145


def prior_Ns():
    used = set()
    def draw(seed, scales, size):
        rng = np.random.default_rng(seed)
        for N0 in scales:
            for N in 2 * rng.integers(N0 // 2, int(1.1 * N0) // 2, size=size):
                used.add(int(N))
    draw(11, [10**6, 10**7, 10**8], 30)
    draw(21, [10**6, 10**7, 10**8, 10**9], 10)
    draw(31, [10**6, 10**7, 10**8, 10**9], 10)
    draw(41, [10**6, 10**7, 10**8, 10**9], 100)
    draw(51, [10**6, 10**7, 10**8], 60)
    draw(61, [10**6, 10**7, 10**8, 10**9], 20)
    draw(71, [10**6, 10**7, 10**8, 10**9], 20)
    draw(20261125, [10**6, 10**7, 10**8], 30)
    return used


used = prior_Ns()
rng = np.random.default_rng(2026112402)
s3, s0 = [], []
while len(s3) < 40 or len(s0) < 40:
    N = int(2 * rng.integers(10**6 // 2, 11 * 10**5 // 2))
    if N in used or N in s3 or N in s0:
        continue
    if N % 3 == 0 and len(s3) < 40:
        s3.append(N)
    elif N % 3 != 0 and len(s0) < 40:
        s0.append(N)
Ns = np.array(s3 + s0)
grp = np.array([N % 3 == 0 for N in Ns])

s = np.ones(M + 1, bool); s[:2] = False
for i in range(2, int(M**0.5) + 1):
    if s[i]:
        s[i * i :: i] = False
f = np.zeros(L); f[: M + 1] = s
d = np.zeros(L); n = np.arange(2, M + 1); d[2 : M + 1] = 1 / np.log(n)
q = f - d
F = np.fft.fft(f); D = np.fft.fft(d); Q = np.fft.fft(q)
conv = lambda A, B: np.real(np.fft.ifft(A * B))
rr = conv(F, F); dq = conv(D, Q); dd = conv(D, D); qq = conv(Q, Q)
print("FFTs", f"{time.time()-t0:.1f}s", flush=True)


def sing(N):
    v = 2 * C2; m = N
    while m % 2 == 0:
        m //= 2
    p = 3
    while p * p <= m:
        if m % p == 0:
            v *= (p - 1) / (p - 2)
            while m % p == 0:
                m //= p
        p += 2
    if m > 1:
        v *= (m - 1) / (m - 2)
    return v


def b2(N):
    I, _ = integrate.quad(lambda t: 1 / (math.log(t) * math.log(N - t)), 2, N / 2, limit=400)
    return sing(N) * 2 * I


r_direct = np.array([int(np.sum(s[2 : N - 1] & s[N - 2 : 1 : -1])) for N in Ns])
b2v = np.array([b2(int(N)) for N in Ns])
e2 = r_direct - b2v
C2v = 2 * dq[Ns]; QQv = qq[Ns]; A2v = dd[Ns] - b2v
ident_err = float(np.max(np.abs(e2 - (C2v + QQv + A2v))))
fft_r_err = float(np.max(np.abs(rr[Ns] - r_direct)))
print("identity max err", ident_err, "fft r err", fft_r_err, flush=True)


def center(x):
    x = np.asarray(x, float).copy()
    for g_ in (True, False):
        x[grp == g_] -= x[grp == g_].mean()
    return x


def kept_mask(step):
    k = np.arange(L)
    mask = np.zeros(L, bool)
    for c in np.arange(0, L, step):
        mask[(c + np.arange(-16, 17)) % L] = True
    mask |= mask[(-k) % L]
    mask[0] = True; mask[L // 2] = True
    return mask


def analyse(mask, label, seed):
    Qk = Q * mask; Qu = Q * (~mask)
    W = np.real(np.fft.ifft(Qk * Qk)); V = np.real(np.fft.ifft(Qu * Qu))
    Wv = W[Ns]; Vv = V[Ns]
    Tobs = float(np.sum(center(Vv / Ns) ** 2))
    rngs = np.random.default_rng(seed)
    kk = np.arange(L)
    U = np.nonzero((~mask) & (kk < L // 2) & (kk > 0))[0]
    absQ = np.abs(Q[U])
    Ts = []
    for _ in range(199):
        ph = np.exp(1j * rngs.uniform(0, 2 * np.pi, len(U)))
        Qs = np.zeros(L, complex); Qs[U] = absQ * ph; Qs[(L - U) % L] = np.conj(absQ * ph)
        Vs = np.real(np.fft.ifft(Qs * Qs))[Ns]
        Ts.append(float(np.sum(center(Vs / Ns) ** 2)))
    Ts = np.array(Ts)
    rank = int(np.sum(Ts < Tobs))
    p_two = float(min(1.0, 2 * min((rank + 1) / 200, (199 - rank + 1) / 200)))
    dN = (Ns[:, None] - Ns[None, :]).ravel()
    cov = np.zeros(len(dN))
    for i0 in range(0, len(U), 20000):
        Ub = U[i0 : i0 + 20000]
        cov += np.cos(2 * np.pi * np.outer(dN, Ub) / L) @ (absQ[i0 : i0 + 20000] ** 4)
    cov = (2 / L**2 * cov).reshape(len(Ns), len(Ns)) / np.outer(Ns, Ns)
    Cm = np.eye(len(Ns))
    for g_ in (True, False):
        idx = np.nonzero(grp == g_)[0]
        Cm[np.ix_(idx, idx)] -= 1 / len(idx)
    ET_pred = float(np.trace(Cm @ cov @ Cm.T))
    res = dict(label=label, n_kept=int(mask.sum()), T_obs=Tobs, T_null_median=float(np.median(Ts)),
               T_null_q025=float(np.quantile(Ts, 0.025)), T_null_q975=float(np.quantile(Ts, 0.975)),
               rank=rank, p_two_sided=p_two, ET_pred_eq5=ET_pred,
               sd_W=float(np.std(center(Wv / Ns), ddof=1)), sd_V=float(np.std(center(Vv / Ns), ddof=1)))
    print(label, json.dumps({k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in res.items()}), f"{time.time()-t0:.0f}s", flush=True)
    return res


main = analyse(kept_mask(L // 210), "mod210", 2026112403)
sens = analyse(kept_mask(L // 2310), "mod2310", 2026112404)
comp = dict(sd_e2=float(np.std(center(e2 / Ns), ddof=1)), sd_C2=float(np.std(center(C2v / Ns), ddof=1)),
            sd_qq=float(np.std(center(QQv / Ns), ddof=1)), sd_A2=float(np.std(center(A2v / Ns), ddof=1)))
print("components (group-centred sd of /N):", comp)
verdict = "not rejected" if main["p_two_sided"] >= 0.05 else "rejected"
print("H0 (mod 210):", verdict)
json.dump(dict(Ns=[int(x) for x in Ns], identity_max_err=ident_err, fft_r_err=fft_r_err, components=comp, main=main,
               sensitivity_mod2310=sens, verdict=verdict, r=[int(x) for x in r_direct], b2=[float(x) for x in b2v],
               C2=[float(x) for x in C2v], qq=[float(x) for x in QQv], A2=[float(x) for x in A2v]),
          open(os.path.join(OUT, "b1_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
