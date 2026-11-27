"""M2-E7（预注册，DESIGN 第 5 节）：给定小模数剩余 σ = y mod lcm(1..12) 的条件二阶矩。
对每个 σ 抽 R 个 y ≡ σ (mod Q_small)，算条件比值 E[X|σ]²/E[X²|σ]（TV 终点、非回溯、长度 L），
再对 σ 平均：P(X > 0) ≥ E_σ[条件比值]。"""
import sys, json, math, random, time
from collections import defaultdict
from m2_e4_random_model import bits_needed
QS = 27720  # lcm(1..12)
def counts_for_y(y, q, A, Lmax, V0):
    Z = 2 * A * q; adj = defaultdict(list)
    adj[0].append((-1, 1)); adj[-1].append((0, 1))
    for d in range(2, Z + 1):
        e = y % d; adj[e].append((e - d, d)); adj[e - d].append((e, d))
    j0 = y % q
    if j0 == 0: return None
    cur = {(j0, 0): 1}; out = []
    for L in range(1, Lmax + 1):
        nxt = defaultdict(int)
        for (v, din), c in cur.items():
            for (w, d) in adj[v]:
                if d != din: nxt[(w, d)] += c
        cur = nxt
        out.append(sum(c for (v, _), c in cur.items() if -V0 <= v < V0))
    return out
def run(q, A, nsig, R, seed, Lmax=8, V0=8):
    rng = random.Random(seed); B = bits_needed(2 * A * q, q); t0 = time.time()
    cond = [[] for _ in range(Lmax)]; unc1 = [0.0] * Lmax; unc2 = [0.0] * Lmax; N = 0
    for _ in range(nsig):
        sig = rng.randrange(QS); s1 = [0.0] * Lmax; s2 = [0.0] * Lmax; n = 0
        for _ in range(R):
            r = counts_for_y(sig + QS * rng.getrandbits(B), q, A, Lmax, V0)
            if r is None: continue
            n += 1
            for i, x in enumerate(r): s1[i] += x; s2[i] += x * x
        if n == 0: continue
        N += n
        for i in range(Lmax):
            unc1[i] += s1[i]; unc2[i] += s2[i]
            cond[i].append((s1[i] / n) ** 2 / (s2[i] / n) if s2[i] else 0.0)
    res = dict(q=q, A=A, nsig=nsig, R=R, sec=round(time.time() - t0, 1),
               L=[dict(L=i + 1, cond_ratio=round(sum(cond[i]) / len(cond[i]), 4),
                       uncond_ratio=round((unc1[i] / N) ** 2 / (unc2[i] / N), 4)) for i in range(Lmax)])
    return res
if __name__ == "__main__":
    q, A, nsig, R, seed = map(int, sys.argv[1:6])
    print(json.dumps(run(q, A, nsig, R, seed)), flush=True)
