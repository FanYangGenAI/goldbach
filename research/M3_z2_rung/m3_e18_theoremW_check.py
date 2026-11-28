"""M3-E18（预注册见 DESIGN.md 第 15 节）：定理 W 证明的逐步数值核对。"""
import sys, json, random
from m3_e8_ratio import primes_upto
def run(N, label, Gp, pmax=31):
    P = primes_upto(N - 1)
    spf = list(range(N)); 
    for i in range(2, int(N ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, N, i):
                if spf[j] == j: spf[j] = i
    f = [0] * N
    for n in range(2, N): f[n] = f[n // spf[n]] ^ Gp[spf[n]]
    def Gint(m):  # G 在 m < N^2 的整数上（素因子都 < N 时）
        s = 0; p = 2
        while p * p <= m:
            while m % p == 0: s ^= Gp[p]; m //= p
            p += 1
        if m > 1: s ^= Gp[m]
        return s
    refl = [f[z] ^ f[N - z] for z in range(1, N)]
    etap = 1 if sum(refl) * 2 > N - 1 else 0
    E_m1 = {z for z in range(1, N) if f[N - z] != f[z] ^ etap}; B = len(E_m1) // 2
    inv = lambda v: pow(v, N - 2, N)
    def kappa_v(v):
        s = 0
        while v > 1: s ^= Gp[spf[v]]; v //= spf[v]
        return s
    def E_word(eps, v):  # s = eps * v^{-1}
        sres = (inv(v) * (1 if eps == 1 else N - 1)) % N; k = kappa_v(v) ^ (etap if eps == -1 else 0)
        return {z for z in range(1, N) if f[(sres * z) % N] != f[z] ^ k}
    def Ep(p): return {n for n in range(1, N) if Gint(p * n) != f[(p * n) % N]}
    e = {}; viol = []; rows = []
    Ecache = {}
    for p in [p for p in P if p <= pmax]:
        e[p] = len(Ep(p))
        # (i) Thue 表示
        reps = {}
        for v in range(1, p):
            for eps in (1, -1):
                sres = (inv(v) * (1 if eps == 1 else N - 1)) % N
                for y in range(1, (N - 1) // p + 1):
                    if p * y < N: reps.setdefault((sres * y) % N, (eps, v, y))
        missing = (N - 1) - len(reps)
        # (ii) 第 4 步不等式
        stepsum = 0
        for v in range(1, p):
            for eps in (1, -1):
                key = (eps, v)
                if key not in Ecache: Ecache[key] = E_word(eps, v)
                stepsum += 2 * len(Ecache[key])
        # (iii) 递推不等式
        Ev_bound = 0
        for v in range(2, p):
            m = v
            while m > 1: Ev_bound += e[spf[m]]; m //= spf[m]
        recur = 4 * (p - 1) * B + 4 * Ev_bound
        ok = missing == 0 and e[p] <= stepsum and stepsum <= recur + 0
        # 注：stepsum ≤ recur 本身也应成立（|E_{-v^{-1}}| ≤ 2B + |E_v|，|E_v| ≤ Σ e(ℓ)）
        if not ok: viol.append(p)
        rows.append(dict(p=p, e_p=e[p], ratio=round(e[p] / B, 2) if B else None, stepsum=stepsum, recur=recur, thue_missing=missing))
    return dict(N=N, g=label, B=B, violations=viol, rows=rows)
if __name__ == "__main__":
    out = []
    for N in map(int, sys.argv[1].split(",")):
        P = primes_upto(N - 1); rng = random.Random(N)
        chi = {p: (0 if pow(p, (N - 1) // 2, N) == 1 else 1) for p in P}
        cases = [("random", {p: rng.randint(0, 1) for p in P})]
        for k in (1, 3, 10):
            G = dict(chi); 
            for p in rng.sample(P[:60], k): G[p] ^= 1
            cases.append((f"chi_flip{k}", G))
        G = dict(chi); G[7] ^= 1; cases.append(("chi_flip_q7", G))
        for lab, G in cases:
            r = run(N, lab, G); print(json.dumps(r), flush=True)
