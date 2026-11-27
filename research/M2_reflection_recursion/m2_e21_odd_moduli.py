"""M2-E21（预注册，DESIGN 第 19 节）：只允许奇数乘数时的覆盖率（Liouville 阶梯 z = 2 台阶：偶数 N 的奇数子纤维）。
在奇数子纤维上，反射 t ↦ dN − t 只能用奇数 d（保持奇偶）。随机模型同前：边 {y mod d, y mod d − d}，d 奇数 ≤ Z；
终点 0、−1 由 d = 1 的边相连；起点 j0 = y mod q（q 为奇素数）。"""
import sys, json, math, random, time
from collections import defaultdict, deque
from m2_e4_random_model import bits_needed
def run(q, eta, S, seed):
    Z = round(q ** (1 + eta)); B = bits_needed(Z, q); rng = random.Random(seed); t0 = time.time()
    unc = 0; n = 0
    for _ in range(S):
        y = rng.getrandbits(B); j0 = y % q
        if j0 == 0: n += 1; continue
        adj = defaultdict(list); adj[0].append(-1); adj[-1].append(0)
        for d in range(3, Z + 1, 2):
            e = y % d; adj[e].append(e - d); adj[e - d].append(e)
        seen = {j0}; dq = deque([j0]); ok = False
        while dq:
            v = dq.popleft()
            if v in (0, -1): ok = True; break
            for w in adj[v]:
                if w not in seen: seen.add(w); dq.append(w)
        n += 1; unc += (not ok)
    u = unc / n
    return dict(q=q, eta=eta, Z=Z, S=n, u_odd=round(u, 4), se=round(math.sqrt(max(u * (1 - u), 1e-12) / n), 4), sec=round(time.time() - t0, 1))
if __name__ == "__main__":
    q, S, seed = int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4]); eta = float(sys.argv[2])
    print(json.dumps(run(q, eta, S, seed)), flush=True)
