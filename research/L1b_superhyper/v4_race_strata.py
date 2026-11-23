"""V4: stratify the hypergeometric null of V1b by position block x residue class of m. See DESIGN.md."""
import json, os, time
import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
WINDOWS = [10**6, 10**7, 10**8]
W = 2000
SCHEMES = {"S0_pos64": (64, 1), "S1_pos8_mod210": (8, 210), "S2_pos4_mod2310": (4, 2310)}
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
def zstat(Ns, lpf, B, Qm):
    K = len(Ns)
    out = np.zeros((K, 3))  # observed Pc, sum of hypergeometric means, sum of variances
    S = B * Qm
    for k in nb.prange(K):
        N = Ns[k]; z = icbrt(N)
        T = np.zeros((S, 4), dtype=np.int64)
        lo = N // 2
        m = N // 2 + 1 if (N // 2) % 2 == 0 else N // 2
        while m <= N - 3:
            f = lpf[m]; g = lpf[N - m]
            if f > z and g > z:
                b = (m - lo) * B // (N - lo)
                if b >= B:
                    b = B - 1
                s = b * Qm + (m % Qm)
                T[s, 0] += 1
                P = f == m; Q = g == N - m
                if P: T[s, 1] += 1
                if Q: T[s, 2] += 1
                if P and Q: T[s, 3] += 1
            m += 2
        pc = 0.0; mu = 0.0; var = 0.0
        for s in range(S):
            M = T[s, 0]
            pc += T[s, 3]
            if M > 1:
                KP = T[s, 1]; r = T[s, 2]
                mu += r * KP / M
                var += r * KP * (M - KP) * (M - r) / (M * M * (M - 1.0))
            elif M == 1:
                mu += T[s, 3]  # a single element: no randomness
        out[k, 0] = pc; out[k, 1] = mu; out[k, 2] = var
    return out


lpf = lpf_sieve(int(1.1 * max(WINDOWS)) + 10)
rng = np.random.default_rng(2026)  # same sampling as V1/V1b
res = {}
for N0 in WINDOWS:
    Ns = np.sort(2 * rng.choice(np.arange(N0 // 2, int(1.1 * N0) // 2), size=W, replace=False)).astype(np.int64)
    res[N0] = {}
    for name, (B, Qm) in SCHEMES.items():
        o = zstat(Ns, lpf, B, Qm)
        z = (o[:, 0] - o[:, 1]) / np.sqrt(o[:, 2])
        res[N0][name] = dict(z_mean=float(z.mean()), z_sd=float(z.std()), var_ratio=float(z.var()))
        print(N0, name, res[N0][name], f"{time.time()-t0:.1f}s", flush=True)
res["H_race_supported"] = all(res[N0]["S2_pos4_mod2310"]["z_sd"] <= 1.10 for N0 in WINDOWS)
res["H_race_rejected"] = all(res[N0]["S2_pos4_mod2310"]["z_sd"] >= 1.15 for N0 in WINDOWS)
print({k: v for k, v in res.items() if isinstance(k, str)})
json.dump(res, open(os.path.join(OUT, "v4_summary.json"), "w"), indent=2)
