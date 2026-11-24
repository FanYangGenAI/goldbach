"""B3: sieve-preserving (Granville-type pseudo-prime) null for the tail-removed reflection statistic. See DESIGN.md."""
import json, math, os, time
import numpy as np
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
t0 = time.time()
M = 1_100_000; L = 2310 * 953; BLOCK = 4096
C2 = 0.6601618158468695739278121100145


def prior_Ns():
    used = set()
    def draw(seed, scales, size):
        rng = np.random.default_rng(seed)
        for N0 in scales:
            for N in 2 * rng.integers(N0 // 2, int(1.1 * N0) // 2, size=size):
                used.add(int(N))
    for seed, sc, size in [(11, [10**6, 10**7, 10**8], 30), (21, [10**6, 10**7, 10**8, 10**9], 10), (31, [10**6, 10**7, 10**8, 10**9], 10),
                           (41, [10**6, 10**7, 10**8, 10**9], 100), (51, [10**6, 10**7, 10**8], 60), (61, [10**6, 10**7, 10**8, 10**9], 20),
                           (71, [10**6, 10**7, 10**8, 10**9], 20), (20261125, [10**6, 10**7, 10**8], 30)]:
        draw(seed, sc, size)
    for f_ in [os.path.join(HERE, "..", "B1_phase", "out", "b1_summary.json"), os.path.join(HERE, "..", "B2_rigidity", "out", "b2_summary.json")]:
        used |= set(int(x) for x in json.load(open(f_))["Ns"])
    return used


used = prior_Ns()
rng = np.random.default_rng(2026112408)
s3, s0 = [], []
while len(s3) < 40 or len(s0) < 40:
    N = int(2 * rng.integers(10**6 // 2, 11 * 10**5 // 2))
    if N in used or N in s3 or N in s0: continue
    if N % 3 == 0 and len(s3) < 40: s3.append(N)
    elif N % 3 != 0 and len(s0) < 40: s0.append(N)
Ns = np.array(s3 + s0); grp = np.array([N % 3 == 0 for N in Ns])

s = np.ones(M + 1, bool); s[:2] = False
for i in range(2, int(M**0.5) + 1):
    if s[i]: s[i * i :: i] = False
d = np.zeros(L); d[2 : M + 1] = 1 / np.log(np.arange(2, M + 1)); D = np.fft.fft(d)


def kept_mask(step):
    k = np.arange(L); mask = np.zeros(L, bool)
    for c in np.arange(0, L, step):
        mask[(c + np.arange(-16, 17)) % L] = True
    mask |= mask[(-k) % L]; mask[0] = True; mask[L // 2] = True
    return mask


MASK = kept_mask(L // 210); NOTM = ~MASK


def V_of(f_bool):
    f = np.zeros(L); f[: M + 1] = f_bool
    Q = np.fft.fft(f) - D
    Qu = Q * NOTM
    return np.real(np.fft.ifft(Qu * Qu))[Ns]


def oddprimes(N):
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


def sing(N):
    v = 2 * C2
    for p in oddprimes(N): v *= (p - 1) / (p - 2)
    return v


def b2(N):
    I, _ = integrate.quad(lambda t: 1 / (math.log(t) * math.log(N - t)), 2, N / 2, limit=400)
    return sing(N) * 2 * I


b2v = np.array([b2(int(N)) for N in Ns])


def ratio(N, pmax):
    a = 1.0; b = 1.0
    for p in oddprimes(int(N)):
        b *= (p - 1) / (p - 2)
        if p <= pmax: a *= (p - 1) / (p - 2)
    return a / b


def center(x):
    x = np.asarray(x, float).copy()
    for g_ in (True, False):
        x[grp == g_] -= x[grp == g_].mean()
    return x


tau_full = np.array([b2v[i] * (1 - ratio(N, 7)) for i, N in enumerate(Ns)])
V_real = V_of(s)
T_real = float(np.sum(center((V_real - tau_full) / Ns) ** 2))
print("T_real", T_real, f"{time.time()-t0:.0f}s", flush=True)


def pseudo_null(z, seed, reps=199):
    rr = np.random.default_rng(seed)
    small = np.array([p for p in range(2, z + 1) if s[p]])
    surv = np.ones(M + 1, bool); surv[:z + 1] = False
    for p in small: surv[p :: p] = False
    surv[: z + 1] = False
    sv = np.nonzero(surv)[0]
    blk_s = sv // BLOCK
    primes_big = np.nonzero(s)[0]; primes_big = primes_big[primes_big > z]
    need = np.bincount(primes_big // BLOCK, minlength=M // BLOCK + 1)
    have = np.bincount(blk_s, minlength=M // BLOCK + 1)
    assert np.all(need <= have)
    starts = np.concatenate([[0], np.cumsum(have)[:-1]])
    tau_z = np.array([b2v[i] * (ratio(N, z) - ratio(N, 7)) for i, N in enumerate(Ns)])
    Ts = []
    for _ in range(reps):
        key = blk_s + rr.random(len(sv))            # random order within block
        order = np.argsort(key, kind="stable")
        rank_in_block = np.arange(len(sv)) - starts[blk_s[order]]
        chosen = sv[order[rank_in_block < need[blk_s[order]]]]
        fb = np.zeros(M + 1, bool); fb[small] = True; fb[chosen] = True
        Vp = V_of(fb)
        Ts.append(float(np.sum(center((Vp - tau_z) / Ns) ** 2)))
    Ts = np.array(Ts)
    return dict(z=z, n_survivors=int(len(sv)), n_primes_big=int(len(primes_big)), T_median=float(np.median(Ts)),
                q025=float(np.quantile(Ts, 0.025)), q975=float(np.quantile(Ts, 0.975)), rank_of_real=int(np.sum(Ts < T_real)),
                ratio_real_to_median=float(T_real / np.median(Ts)), Ts=[float(x) for x in Ts])


res100 = pseudo_null(100, 2026112409)
print("z=100", {k: v for k, v in res100.items() if k != "Ts"}, f"{time.time()-t0:.0f}s", flush=True)
res30 = pseudo_null(30, 2026112410)
print("z=30", {k: v for k, v in res30.items() if k != "Ts"}, f"{time.time()-t0:.0f}s", flush=True)
if T_real < res100["q025"]:
    verdict = "rigidity beyond sieve (z=100)"
elif T_real > res100["q975"]:
    verdict = "real primes less rigid than pseudo-primes"
else:
    verdict = "explained by sieve structure (z=100)"
print("B3 verdict:", verdict)
json.dump(dict(Ns=[int(x) for x in Ns], T_real=T_real, z100=res100, z30=res30, verdict=verdict),
          open(os.path.join(OUT, "b3_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
