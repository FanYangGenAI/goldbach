"""M2-E24（预注册，DESIGN 第 22 节）：两条素数束超边相交概率的 Paley–Zygmund 下界（第 34 节公式）。
对随机 y 与随机的一对 ℓ ≠ ℓ′ ∈ 𝒫，计算 n_c（c 跑遍模 ℓℓ′ 的剩余类），给出 PZ = (Σn_c)²/(ℓℓ′Σn_c²)，
以及"n_c > 0 的剩余类比例"（= 给定 σ1 时 e_ℓ 与 e_ℓ′ 相交的精确概率）。"""
import sys, json, math, random, time
from collections import defaultdict
from m2_e4_random_model import bits_needed
from m2_e23_bundles import primes_in, small_G
def run(q, eta, S, pairs, seed, a=0.15, b=0.45):
    W = round(q ** (1 + eta)); B = bits_needed(W, q); rng = random.Random(seed); t0 = time.time()
    P = primes_in(int(W ** a) + 1, int(W ** b))
    D1 = [d for d in range(2, W + 1) if all(d % l for l in P)]
    PZs = []; exact = []
    for _ in range(S):
        y = rng.getrandbits(B); ymods = {m: y % m for m in D1}
        cacheG = {}
        def I(l):
            if l not in cacheG:
                Gs = [small_G(ymods, r, l, W // l, D1) for r in range(l)]
                cacheG[l] = Gs
            return cacheG[l]
        for _ in range(pairs):
            l1, l2 = rng.sample(P, 2)
            G1, G2 = I(l1), I(l2)
            M = l1 * l2; n = defaultdict(int)
            for z in range(-W, W):
                if (z // l1) in G1[z % l1] and (z // l2) in G2[z % l2]:
                    n[z % M] += 1
            s1 = sum(n.values()); s2 = sum(v * v for v in n.values())
            PZs.append(s1 * s1 / (M * s2) if s2 else 0.0); exact.append(len(n) / M)
    PZs.sort(); exact.sort()
    return dict(q=q, eta=eta, W=W, nP=len(P), n_pairs=len(PZs), PZ_median=round(PZs[len(PZs) // 2], 4), PZ_min=round(PZs[0], 4),
                exact_median=round(exact[len(exact) // 2], 4), exact_min=round(exact[0], 4), sec=round(time.time() - t0, 1))
if __name__ == "__main__":
    q, S, pairs, seed = int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); eta = float(sys.argv[2])
    bb = float(sys.argv[6]) if len(sys.argv) > 6 else 0.45
    print(json.dumps(dict(run(q, eta, S, pairs, seed, b=bb), b=bb)), flush=True)
