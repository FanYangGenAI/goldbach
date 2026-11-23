"""V1: conditional-sampling (hypergeometric) null for the L1 parity probe. See DESIGN.md."""
import json, os, time
import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
WINDOWS = [10**6, 10**7, 10**8]
W = 2000
t0 = time.time()


@nb.njit(cache=True)
def lpf_sieve(n):
    lpf = np.zeros(n + 1, dtype=np.int32)
    for i in range(2, n + 1):
        if lpf[i] == 0:
            lpf[i] = i
            j = np.int64(i) * i
            while j <= n:
                if lpf[j] == 0:
                    lpf[j] = i
                j += i
    return lpf


@nb.njit(cache=True)
def icbrt(n):
    z = int(round(n ** (1.0 / 3.0)))
    while z * z * z > n:
        z -= 1
    while (z + 1) * (z + 1) * (z + 1) <= n:
        z += 1
    return z


@nb.njit(parallel=True, cache=True)
def table(Ns, lpf):
    K = len(Ns)
    out = np.zeros((K, 4), dtype=np.int64)  # M, KP, r, Pc
    for k in nb.prange(K):
        N = Ns[k]; z = icbrt(N)
        M = 0; KP = 0; r = 0; pc = 0
        m = N // 2 + 1 if (N // 2) % 2 == 0 else N // 2
        while m <= N - 3:
            f = lpf[m]; g = lpf[N - m]
            if f > z and g > z:
                M += 1
                P = f == m; Q = g == N - m
                if P: KP += 1
                if Q: r += 1
                if P and Q: pc += 1
            m += 2
        out[k, 0] = M; out[k, 1] = KP; out[k, 2] = r; out[k, 3] = pc
    return out


lpf = lpf_sieve(int(1.1 * max(WINDOWS)) + 10)
rng = np.random.default_rng(2026)
res = {}
for N0 in WINDOWS:
    Ns = np.sort(2 * rng.choice(np.arange(N0 // 2, int(1.1 * N0) // 2), size=W, replace=False)).astype(np.int64)
    T = table(Ns, lpf).astype(float)
    M, KP, r, Pc = T.T
    mean = r * KP / M
    var = r * KP * (M - KP) * (M - r) / (M**2 * (M - 1))
    z = (Pc - mean) / np.sqrt(var)
    # binomial-style z (as in E1.2, unweighted) for comparison
    f0 = KP / M; fb = Pc / r
    zb = (fb - f0) / np.sqrt(f0 * (1 - f0) / r)
    res[N0] = dict(z_hyp_mean=float(z.mean()), z_hyp_sd=float(z.std()), z_binom_sd=float(zb.std()),
                   mean_sampling_fraction=float((r / M).mean()), fpc_factor=float(np.sqrt(((M - r) / (M - 1)).mean())),
                   z_hyp_min=float(z.min()), z_hyp_max=float(z.max()))
    print(N0, res[N0], f"{time.time()-t0:.1f}s", flush=True)
res["V1"] = all(0.85 <= res[N0]["z_hyp_sd"] <= 1.15 and abs(res[N0]["z_hyp_mean"]) < 0.1 for N0 in WINDOWS)
print("V1:", res["V1"])
json.dump(res, open(os.path.join(OUT, "v1_summary.json"), "w"), indent=2)
