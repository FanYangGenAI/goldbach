"""M2-E8（预注册，DESIGN 第 6 节）：plain 模型在小 A 下的覆盖率随 q 的趋势，以及"逃离 q 的倍数"的概率。

动机（Ford 2008，乘法表问题）：随机整数在区间 (x, Cx] 内有因子的概率 ≍ (log x)^{-δ}(log log x)^{-3/2}，
δ = 1 − (1 + log log 2)/log 2 ≈ 0.086，C 固定时随 x → ∞ 趋于 0。层级 ≈ q 的顶点对应的整数要在 (u, u + 2Aq]
中有因子才有边，所以固定 A 时覆盖率可能随 q 极慢地衰减。

对每个样本记录：
  covered：j0 与 0 连通；
  escape ：j0 的连通块中含有一个 ≢ j0 (mod q) 的顶点（即离开"q 的倍数缩放副本"）；
  comp   ：未覆盖时 j0 连通块的大小（上限 20000）。"""
import sys, json, math, random, time
from collections import defaultdict, deque
from m2_e4_random_model import bits_needed

def one(rng, q, A, B):
    Z = 2 * A * q
    y = rng.getrandbits(B)
    adj = defaultdict(list); adj[0].append(-1); adj[-1].append(0)
    for d in range(2, Z + 1):
        e = y % d; adj[e].append(e - d); adj[e - d].append(e)
    j0 = y % q
    if j0 == 0:
        return True, True, 0
    seen = {j0}; dq = deque([j0]); esc = False; cov = False
    while dq and len(seen) < 20000:
        v = dq.popleft()
        if v in (0, -1): cov = True; break
        if (v - j0) % q: esc = True
        for w in adj[v]:
            if w not in seen: seen.add(w); dq.append(w)
    if cov:
        return True, True, len(seen)
    if not dq:  # 连通块已穷尽
        return False, esc, len(seen)
    # 块太大仍未找到 0：继续 BFS 到底以判定
    while dq:
        v = dq.popleft()
        if v in (0, -1): return True, True, len(seen)
        if (v - j0) % q: esc = True
        for w in adj[v]:
            if w not in seen: seen.add(w); dq.append(w)
    return False, esc, len(seen)

def run(q, A, S, seed):
    rng = random.Random(seed); B = bits_needed(2 * A * q, q); t0 = time.time()
    unc = 0; noesc = 0; comps = []
    for _ in range(S):
        c, e, n = one(rng, q, A, B)
        if not c: unc += 1; comps.append(n)
        if not e: noesc += 1
    u = unc / S; ne = noesc / S
    comps.sort()
    return dict(q=q, A=A, S=S, u=round(u, 4), se_u=round(math.sqrt(max(u * (1 - u), 1e-12) / S), 4),
                no_escape=round(ne, 4), se_ne=round(math.sqrt(max(ne * (1 - ne), 1e-12) / S), 4),
                comp_median=comps[len(comps) // 2] if comps else 0,
                comp_max=comps[-1] if comps else 0,
                frac_small_comp=round(sum(1 for c in comps if c <= 8 * A + 8) / max(1, len(comps)), 3),
                sec=round(time.time() - t0, 1))

if __name__ == "__main__":
    q, A, S, seed = map(int, sys.argv[1:5])
    print(json.dumps(run(q, A, S, seed)), flush=True)
