"""M2-E23（预注册，DESIGN 第 21 节）：素数束超边的数值核对（引理 B、C）与 Λ 的大小。
𝒫 = (W^a, W^b] 中的素数，D1 = 与 𝒫 互素的乘数。起点块 A = j0 在 D1 子图中的连通块。
对每个 z ∈ A、ℓ ∈ 𝒫：p_{z,ℓ} = ℓ^{-1}·1[⌊z/ℓ⌋ ∈ G_ℓ(z mod ℓ)]，G_ℓ(r) 为小图 H_{(y−r)/ℓ}[D1, 窗口 W/ℓ] 中 0 的块。
预测：P(A 不被任何实际超边 e_ℓ 碰到 | σ1) = Π_ℓ (1 − Σ_{z 的不同剩余类} p_{z,ℓ})。"""
import sys, json, math, random, time
from collections import defaultdict, deque
from m2_e4_random_model import bits_needed
def primes_in(lo, hi):
    return [p for p in range(max(2, lo), hi + 1) if all(p % t for t in range(2, int(p ** .5) + 1))]
def comp_from(start, adj):
    seen = {start}; dq = deque([start])
    while dq:
        v = dq.popleft()
        for w in adj[v]:
            if w not in seen: seen.add(w); dq.append(w)
    return seen
def small_G(ymods, r, l, Wl, D1small):
    adj = defaultdict(list); adj[0].append(-1); adj[-1].append(0)
    for m in D1small:
        if m > Wl: break
        s = ((ymods[m] - r) * pow(l, -1, m)) % m
        adj[s].append(s - m); adj[s - m].append(s)
    return comp_from(0, adj)
def run(q, eta, S, seed, a=0.15, b=0.45):
    W = round(q ** (1 + eta)); B = bits_needed(W, q); rng = random.Random(seed); t0 = time.time()
    P = primes_in(int(W ** a) + 1, int(W ** b)); Pset = set(P)
    D1 = [d for d in range(2, W + 1) if all(d % l for l in P)]
    obs = pred = 0.0; Lams = []; lemmaB_ok = lemmaB_tot = 0; n = 0
    for _ in range(S):
        y = rng.getrandbits(B); j0 = y % q
        if j0 in (0, q - 1): continue
        n += 1
        ymods = {m: y % m for m in D1}
        adj1 = defaultdict(list); adj1[0].append(-1); adj1[-1].append(0)
        for d in D1:
            e = ymods[d]; adj1[e].append(e - d); adj1[e - d].append(e)
        A = comp_from(j0, adj1)
        prod = 1.0; Lam = 0.0; touched = False
        for l in P:
            Wl = W // l
            rl = y % l
            # 实际超边：r_ℓ 在乘数 ℓm（m ∈ D1，m ≤ W/ℓ）下的连通块
            adjb = defaultdict(list)
            e = y % l; adjb[e].append(e - l); adjb[e - l].append(e)   # m = 1：E_ℓ 本身
            for m in D1:
                if m > Wl: break
                d = l * m; e = y % d; adjb[e].append(e - d); adjb[e - d].append(e)
            el = comp_from(rl, adjb)
            # 引理 B 核对：e_ℓ = r_ℓ + ℓ·G_ℓ(r_ℓ)
            Gl = small_G(ymods, rl, l, Wl, D1)
            lemmaB_tot += 1; lemmaB_ok += (el == {rl + l * v for v in Gl})
            if A & el: touched = True
            # 预测：按 A 的剩余类合并
            cls = {}
            for z in A:
                c = z % l
                if c in cls: continue
                G = Gl if c == rl else small_G(ymods, c, l, Wl, D1)
                cls[c] = (z // l) in G
            # 同一剩余类里只要有一个 z 满足即可（同一 r）
            hit = defaultdict(bool)
            for z in A:
                c = z % l
                if not hit[c]:
                    G = Gl if c == rl else small_G(ymods, c, l, Wl, D1)
                    hit[c] = (z // l) in G
            s = sum(1 for c in hit if hit[c]) / l
            prod *= (1 - s); Lam += s
        obs += (not touched); pred += prod; Lams.append(Lam)
    Lams.sort()
    return dict(q=q, eta=eta, W=W, nP=len(P), n=n, lemmaB_match=f"{lemmaB_ok}/{lemmaB_tot}",
                P_untouched_obs=round(obs / n, 4), P_untouched_pred=round(pred / n, 4),
                Lambda_median=round(Lams[len(Lams) // 2], 3), sec=round(time.time() - t0, 1))
if __name__ == "__main__":
    q, S, seed = int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4]); eta = float(sys.argv[2])
    print(json.dumps(run(q, eta, S, seed)), flush=True)
