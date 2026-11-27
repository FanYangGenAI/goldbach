"""M2-E4（预注册，见 DESIGN.md 第 3 节）：层级图的随机模型，q 到 10^4。

模型：y 为模 lcm(D) 的均匀随机整数（取 B 位随机整数，B ≥ log2 lcm(D) + 64，
所以 (y mod d)_{d∈D} 与均匀分布的全变差 < 2^-64；用 ψ(x) < 1.03883x，log2 lcm(1..x) < 1.4988x）。
图 H_y：对每个 d ∈ D 连一条边 {y mod d, y mod d − d}，再加 d = 1 的边 {0, −1}。
起点 j0 = y mod q（对应 qn 所在的层级），终点 0。
  model "qa"   : D = {2..Aq} ∪ {q·a : a ≤ Aq}（与 M2-E1 的乘数集相同，Z = Aq）
  model "plain": D = {2..2Aq}
  model "smooth": D = {2 ≤ a ≤ Aq : P⁺(a) ≤ √q} ∪ {q·a : a ≤ Aq, P⁺(a) ≤ √q}
输出：未覆盖率 u（j0 与 0 不连通的比例）及标准误、连通时的最短路长均值。"""
import sys, json, math, random, time
from collections import deque, defaultdict

def D_of(model, q, A):
    Z = A * q
    if model == "qa":
        return sorted(set(range(2, Z + 1)) | {q * a for a in range(1, Z + 1)}), Z
    if model == "smooth":
        # 只用 K-光滑乘数（K = ⌊√q⌋），对应"所有乘数整除一个 Q ≤ p^{1/2}"的可转移版本（RESULTS 第 8 节）
        K = math.isqrt(q)
        def P_plus(m):
            r, x, f = 1, m, 2
            while f * f <= x:
                while x % f == 0: r = max(r, f); x //= f
                f += 1
            return max(r, x) if x > 1 else r
        sm = [a for a in range(1, Z + 1) if P_plus(a) <= K]
        return sorted({a for a in sm if a >= 2} | {q * a for a in sm}), Z
    return list(range(2, 2 * Z + 1)), 2 * Z

def bits_needed(x, q):
    return math.ceil(1.4988 * x) + 2 * max(1, q.bit_length()) + 64

def one_sample(rng, D, B, q):
    y = rng.getrandbits(B)
    adj = defaultdict(list)
    adj[0].append(-1); adj[-1].append(0)
    for d in D:
        e = y % d
        adj[e].append(e - d); adj[e - d].append(e)
    j0 = y % q
    if j0 == 0:
        return True, 0
    # BFS from 0 until j0 is found
    dist = {0: 0, -1: 0}; dq = deque([0, -1])
    while dq:
        v = dq.popleft()
        for w in adj[v]:
            if w not in dist:
                dist[w] = dist[v] + 1
                if w == j0:
                    return True, dist[w]
                dq.append(w)
    return False, None

def run(model, q, A, S, seed):
    D, x = D_of(model, q, A)
    B = bits_needed(x, q)
    rng = random.Random(seed)
    unc = 0; lens = []
    t0 = time.time()
    for _ in range(S):
        ok, L = one_sample(rng, D, B, q)
        if not ok: unc += 1
        elif L: lens.append(L)
    u = unc / S
    return dict(model=model, q=q, A=A, S=S, u=round(u, 5), se=round(math.sqrt(max(u * (1 - u), 1e-12) / S), 5),
                mean_len=round(sum(lens) / max(1, len(lens)), 3), max_len=max(lens) if lens else 0,
                sec=round(time.time() - t0, 1))

if __name__ == "__main__":
    model, q, A, S, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    print(json.dumps(run(model, q, A, S, seed)), flush=True)
