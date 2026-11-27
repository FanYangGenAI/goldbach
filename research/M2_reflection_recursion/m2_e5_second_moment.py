"""M2-E5（预注册，见 DESIGN.md 第 4 节）：plain 随机模型中，固定长度非回溯路径计数的二阶矩比 E[X]²/E[X²]。

y 为模 lcm(2..2Aq) 的均匀随机整数（同 M2-E4）；边 {y mod d, y mod d − d}，d = 1..2Aq。
X_L(T) = 从 j0 = y mod q 出发、长度恰为 L、终点在 T 中的非回溯游走数。
  T0 = {0, −1}；TV = {−V0, …, V0 − 1}（V0 = 8）。
Paley–Zygmund：P(X > 0) ≥ E[X]²/E[X²]，而 X > 0 推出 j0 与 T 连通。比值只依赖成对相关，
所以若在随机模型中可以解析地证明，它可以严格转移到真实 p（RESULTS 第 10 节）。"""
import sys, json, math, random, time
from collections import defaultdict
from m2_e4_random_model import bits_needed

def sample_counts(rng, q, A, B, Lmax, V0):
    Z = 2 * A * q
    y = rng.getrandbits(B)
    adj = defaultdict(list)          # v -> [(w, d)]
    adj[0].append((-1, 1)); adj[-1].append((0, 1))
    for d in range(2, Z + 1):
        e = y % d
        adj[e].append((e - d, d)); adj[e - d].append((e, d))
    j0 = y % q
    if j0 == 0:
        return None
    # 非回溯：状态 (v, 进入时用的边 d)
    cur = {(j0, 0): 1}
    outT0 = []; outTV = []
    for L in range(1, Lmax + 1):
        nxt = defaultdict(int)
        for (v, din), c in cur.items():
            for (w, d) in adj[v]:
                if d != din:
                    nxt[(w, d)] += c
        cur = nxt
        x0 = sum(c for (v, _), c in cur.items() if v in (0, -1))
        xV = sum(c for (v, _), c in cur.items() if -V0 <= v < V0)
        outT0.append(x0); outTV.append(xV)
    return outT0, outTV

def run(q, A, S, seed, Lmax=8, V0=8):
    rng = random.Random(seed); B = bits_needed(2 * A * q, q)
    s1 = {k: [0.0] * Lmax for k in ("T0", "TV")}; s2 = {k: [0.0] * Lmax for k in ("T0", "TV")}
    pos = {k: [0] * Lmax for k in ("T0", "TV")}; n = 0; t0 = time.time()
    for _ in range(S):
        r = sample_counts(rng, q, A, B, Lmax, V0)
        if r is None: continue
        n += 1
        for k, xs in zip(("T0", "TV"), r):
            for i, x in enumerate(xs):
                s1[k][i] += x; s2[k][i] += x * x; pos[k][i] += (x > 0)
    res = dict(q=q, A=A, S=n, sec=round(time.time() - t0, 1))
    for k in ("T0", "TV"):
        res[k] = [dict(L=i + 1, EX=round(s1[k][i] / n, 3),
                       ratio=round((s1[k][i] / n) ** 2 / (s2[k][i] / n), 4) if s2[k][i] else 0.0,
                       P_pos=round(pos[k][i] / n, 4)) for i in range(Lmax)]
    return res

if __name__ == "__main__":
    q, A, S, seed = map(int, sys.argv[1:5])
    print(json.dumps(run(q, A, S, seed)), flush=True)
