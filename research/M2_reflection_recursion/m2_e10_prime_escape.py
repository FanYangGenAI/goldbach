"""M2-E10（预注册，DESIGN 第 8 节）：只用素数乘数时的"逃离"概率（逃离引理的数值前提）。
R 簇：从 j0 出发、只走 q | d 的边能到的顶点。prime_escape：R 簇中某顶点有一条素数 d（d ≠ q）的边。
escape：有任何 q ∤ d 的边。Z = round(q^{1+η})，plain 模型。"""
import sys, json, math, random, time
from collections import defaultdict, deque
from m2_e4_random_model import bits_needed
def isprime(n):
    if n < 2: return False
    i = 2
    while i * i <= n:
        if n % i == 0: return False
        i += 1
    return True
def run(q, eta, S, seed):
    Z = round(q ** (1 + eta)); B = bits_needed(Z, q); rng = random.Random(seed)
    P = [isprime(d) for d in range(Z + 1)]
    ne = npe = 0; t0 = time.time()
    for _ in range(S):
        y = rng.getrandbits(B); j0 = y % q
        if j0 in (0, q - 1): continue
        adjq = defaultdict(list); other = defaultdict(list)
        for d in range(2, Z + 1):
            e = y % d
            if d % q == 0: adjq[e].append(e - d); adjq[e - d].append(e)
            else: other[e].append(d); other[e - d].append(d)
        seen = {j0}; dq = deque([j0])
        while dq:
            v = dq.popleft()
            for w in adjq[v]:
                if w not in seen: seen.add(w); dq.append(w)
        esc = any(other[v] for v in seen)
        pesc = any(P[d] for v in seen for d in other[v])
        ne += (not esc); npe += (not pesc)
    return dict(q=q, eta=eta, Z=Z, S=S, no_escape=round(ne / S, 4), no_prime_escape=round(npe / S, 4),
                se_npe=round(math.sqrt(max(npe / S * (1 - npe / S), 1e-12) / S), 4), sec=round(time.time() - t0, 1))
if __name__ == "__main__":
    q, S, seed = int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4]); eta = float(sys.argv[2])
    print(json.dumps(run(q, eta, S, seed)), flush=True)
