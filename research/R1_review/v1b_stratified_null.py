"""V1b (stratified by position, 64 blocks): conditional-sampling (hypergeometric) null for the L1 parity probe. See DESIGN.md."""
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
    B = 64
    out = np.zeros((K, B, 4), dtype=np.int64)  # per block: M, KP, r, Pc
    for k in nb.prange(K):
        N = Ns[k]; z = icbrt(N)
        lo = N // 2
        m = N // 2 + 1 if (N // 2) % 2 == 0 else N // 2
        while m <= N - 3:
            f = lpf[m]; g = lpf[N - m]
            if f > z and g > z:
                b = (m - lo) * B // (N - lo)
                if b >= B: b = B - 1
                out[k, b, 0] += 1
                P = f == m; Q = g == N - m
                if P: out[k, b, 1] += 1
                if Q: out[k, b, 2] += 1
                if P and Q: out[k, b, 3] += 1
            m += 2
    return out


lpf = lpf_sieve(int(1.1 * max(WINDOWS)) + 10)
rng = np.random.default_rng(2026)
res = {}
for N0 in WINDOWS:
    Ns = np.sort(2 * rng.choice(np.arange(N0 // 2, int(1.1 * N0) // 2), size=W, replace=False)).astype(np.int64)
    T = table(Ns, lpf).astype(float)
    Mb, KPb, rb, Pcb = T[:, :, 0], T[:, :, 1], T[:, :, 2], T[:, :, 3]
    ok = Mb > 1
    meanb = np.where(ok, rb * KPb / np.maximum(Mb, 1), 0)
    varb = np.where(ok, rb * KPb * (Mb - KPb) * (Mb - rb) / (np.maximum(Mb, 1) ** 2 * np.maximum(Mb - 1, 1)), 0)
    z = (Pcb.sum(1) - meanb.sum(1)) / np.sqrt(varb.sum(1))
    M, KP, r, Pc = Mb.sum(1), KPb.sum(1), rb.sum(1), Pcb.sum(1)
    f0 = KP / M; fb = Pc / r
    zb = (fb - f0) / np.sqrt(f0 * (1 - f0) / r)
    res[N0] = dict(z_hyp_mean=float(z.mean()), z_hyp_sd=float(z.std()), z_binom_sd=float(zb.std()),
                   mean_sampling_fraction=float((r / M).mean()), fpc_factor=float(np.sqrt(((M - r) / (M - 1)).mean())),
                   z_hyp_min=float(z.min()), z_hyp_max=float(z.max()))
    print(N0, res[N0], f"{time.time()-t0:.1f}s", flush=True)
res["V1b"] = all(0.85 <= res[N0]["z_hyp_sd"] <= 1.15 and abs(res[N0]["z_hyp_mean"]) < 0.15 for N0 in WINDOWS)
print("V1b:", res["V1b"])
json.dump(res, open(os.path.join(OUT, "v1b_summary.json"), "w"), indent=2)
