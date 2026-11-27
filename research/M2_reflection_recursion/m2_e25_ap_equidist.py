"""M2-E25（预注册，DESIGN 第 23 节）：0 的连通块 G 在算术级数中的等分布（(IH*) 的数值前提）。
plain 随机模型，窗口 W（全部乘数 1..W）。对模数 m 与层级带 [X, 2X)（两侧），
ratio(c) = |G ∩ {v ≡ c (mod m)} ∩ 带| / (|带|/m)，记录各剩余类 ratio 的最小值与均值。"""
import sys, json, math, random, time
from collections import defaultdict, deque
from m2_e4_random_model import bits_needed
def run(W, S, seed, ms=(3, 5, 7, 11, 13), exps=(0.3, 0.5, 0.7)):
    B = bits_needed(W, 3); rng = random.Random(seed); t0 = time.time()
    res = defaultdict(list)
    for _ in range(S):
        y = rng.getrandbits(B)
        adj = defaultdict(list); adj[0].append(-1); adj[-1].append(0)
        for d in range(2, W + 1):
            e = y % d; adj[e].append(e - d); adj[e - d].append(e)
        G = {0}; dq = deque([0])
        while dq:
            v = dq.popleft()
            for w in adj[v]:
                if w not in G: G.add(w); dq.append(w)
        for ex in exps:
            X = int(W ** ex)
            band = [v for v in range(X, 2 * X)] + [-v for v in range(X, 2 * X)]
            for m in ms:
                cnt = defaultdict(int); tot = defaultdict(int)
                for v in band:
                    tot[v % m] += 1; cnt[v % m] += (v in G)
                ratios = [cnt[c] / (tot[c]) for c in range(m) if tot[c]]
                dens = sum(cnt.values()) / len(band)
                res[(ex, m)].append((min(ratios) / dens if dens else 0.0, dens))
    out = {}
    for (ex, m), vals in sorted(res.items()):
        mins = sorted(v[0] for v in vals); dens = sorted(v[1] for v in vals)
        out[f"X=W^{ex},m={m}"] = dict(min_ratio_over_density_median=round(mins[len(mins) // 2], 3),
                                      min_ratio_over_density_min=round(mins[0], 3), density_median=round(dens[len(dens) // 2], 3))
    return dict(W=W, S=S, results=out, sec=round(time.time() - t0, 1))
if __name__ == "__main__":
    W, S, seed = map(int, sys.argv[1:4])
    print(json.dumps(run(W, S, seed)), flush=True)
