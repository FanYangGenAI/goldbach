"""M2-E16（预注册，DESIGN 第 14 节）：相交顶点数 X（有界指示量之和）的二阶矩比 E[X]²/E[X²]。
g = y 的最小素因子（要求 g ≤ q^{η/2} 且 g ≠ q；否则 X = 0）；X = #{z ∈ R ∩ T_g : |z| ≤ Z^{1−δ}}，δ = η/(4(1+η))。"""
import sys, json, math, random, time
from collections import defaultdict, deque
from m2_e4_random_model import bits_needed
def bfs(start, adj, ok):
    seen = {start}; dq = deque([start])
    while dq:
        v = dq.popleft()
        for w, d in adj[v]:
            if ok(d) and w not in seen: seen.add(w); dq.append(w)
    return seen
def run(q, eta, S, seed):
    Z = round(q ** (1 + eta)); B = bits_needed(Z, q); rng = random.Random(seed); t0 = time.time()
    G = max(2, int(q ** (eta / 2))); delta = eta / (4 * (1 + eta)); Zc = Z ** (1 - delta)
    s1 = s2 = 0.0; pos = n = nog = 0
    for _ in range(S):
        y = rng.getrandbits(B); j0 = y % q
        if j0 in (0, q - 1): continue
        n += 1
        g = next((r for r in range(2, G + 1) if y % r == 0 and all(r % t for t in range(2, int(r ** .5) + 1))), None)
        if g is None or g == q: nog += 1; continue
        adj = defaultdict(list); adj[0].append((-1, 1)); adj[-1].append((0, 1))
        for d in range(2, Z + 1):
            e = y % d; adj[e].append((e - d, d)); adj[e - d].append((e, d))
        R = bfs(j0, adj, lambda d: d % q == 0)
        T = bfs(0, adj, lambda d: d % g == 0)
        X = sum(1 for z in R & T if abs(z) <= Zc)
        s1 += X; s2 += X * X; pos += (X > 0)
    EX, EX2 = s1 / n, s2 / n
    return dict(q=q, eta=eta, Z=Z, G=G, n=n, P_no_small_prime=round(nog / n, 4), EX=round(EX, 3),
                PZ=round(EX * EX / EX2, 4) if EX2 else 0.0, P_pos=round(pos / n, 4), sec=round(time.time() - t0, 1))
if __name__ == "__main__":
    q, S, seed = int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4]); eta = float(sys.argv[2])
    print(json.dumps(run(q, eta, S, seed)), flush=True)
