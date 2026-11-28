"""M3-E52：ℓ = 2 的格点计数与对 a_{t-2} 平均的界（预注册：DESIGN.md 第 36 节）。"""
import math, json, random
from m3_e49_coprime import strings, is_prime

def gauss_lambda1(Q, u):
    # 格 {(c,K): c + uK ≡ 0 (mod Q)} 的基 (Q,0), ((-u)%Q, 1)；Gauss 约化求最短向量长度
    v1 = (Q, 0); v2 = ((-u) % Q, 1)
    def n2(v): return v[0] * v[0] + v[1] * v[1]
    if n2(v1) < n2(v2): v1, v2 = v2, v1
    while True:
        # 让 v2 为较短者
        if n2(v1) < n2(v2): v1, v2 = v2, v1
        m = round((v1[0] * v2[0] + v1[1] * v2[1]) / n2(v2))
        w = (v1[0] - m * v2[0], v1[1] - m * v2[1])
        if n2(w) >= n2(v2): return math.sqrt(n2(v2))
        v1 = w

def coset_count(Q, u, z, X, Y):
    return sum(1 for K in range(Y) if (z - u * K) % Q < X)

def family(prefix, a3, a1, at, q, viol_box, rows):
    vals = []; L = a3 - a1 - 1; short = 0; nb = 0
    for b in range(a1 + 1, a3):
        if math.gcd(b, a3) != 1 or math.gcd(b, a1) != 1 or b % q == 0: continue
        a = prefix + [a3, b, a1, at]
        if any(math.gcd(a[i], a[i + 1]) != 1 for i in range(len(a) - 1)): continue
        t = len(a) - 1; j = t - 4
        if a[j] % q != 0: continue
        S = strings(a, True)
        if not S: continue
        nb += 1
        u = a[t - 2] * a[t]; Q = q * a[t - 1]
        lam = gauss_lambda1(Q, u)
        if lam < math.sqrt(L) / 2: short += 1
        X, Y = a[t], a[t - 1] + 1
        bound = X * Y / Q + 2 * (X + Y) / lam + 1
        byeps = {}
        for M, eps, c in S:
            key = tuple(eps); d = byeps.setdefault(key, [0, 0, 0]); d[0] += 1
            d[1] += (M % q == 0); d[2] += ((M + 1) % q == 0)
        for eps, (tot, n0, n1) in byeps.items():
            k1 = (-eps[t - 1] + eps[t - 2] * a[t]) % a[t - 1]
            k2 = (eps[j] * a[t - 1] * u + eps[t - 2] * a[t]) % q
            for k2x, nreal in ((k2, n0), ((k2 - 1) % q, n1)):
                # CRT: z ≡ k1 (mod a_{t-1}), z ≡ k2x (mod q)
                z = next(z for z in range(k1, Q, a[t - 1]) if z % q == k2x)
                cnt = coset_count(Q, u, z, X, Y)
                if nreal > cnt: viol_box[1] += 1   # 真实串数应不超过格点数
                if cnt > bound + 1e-9: viol_box[0] += 1
                viol_box[2] += 1
        P = sum(1 for M, _, _ in S if M % q == 0 or (M + 1) % q == 0) / len(S)
        vals.append(P - 2 / q)
    if not vals: return None
    mean = sum(vals) / len(vals)
    r = dict(prefix=prefix, a3=a3, a1=a1, at=at, q=q, n=nb, mean_excess=round(mean, 5),
             H2_stat=round(mean * at / (math.sqrt(a1) + 1), 4), frac_short=round(short / nb, 3))
    rows.append(r); return r

rows = []; viol = [0, 0, 0]
family([397, 394], 393, 100, 99, 197, viol, rows)
rng = random.Random(52)
tries = 0
while len(rows) < 31 and tries < 5000:
    tries += 1
    a1 = rng.randint(12, 70); at = rng.randint(4, a1 - 1)
    if math.gcd(a1, at) != 1: continue
    a3 = rng.randint(a1 + 20, a1 + 250)
    qs = [x for x in range(3, 150) if is_prime(x) and a1 % x and at % x and a3 % x]
    q = rng.choice(qs)
    m = rng.randint(a3 // q + 1, (a3 + 400) // q); a4 = q * m
    if a4 <= a3 or math.gcd(a4, a3) != 1: continue
    p = next(x for x in range(a4 + 1, 5 * a4) if is_prime(x))
    family([p, a4], a3, a1, at, q, viol, rows)
h2max = max(r['H2_stat'] for r in rows); h3max = max(r['frac_short'] for r in rows)
print(json.dumps(dict(families=len(rows), H1_bound_viol=viol[0], real_exceeds_lattice_count=viol[1], checks=viol[2],
                      H2_max=h2max, H2_holds=h2max <= 2, H3_max_frac_short=h3max, H3_holds=h3max <= 0.5)))
for r in rows[:6]: print(json.dumps(r))
