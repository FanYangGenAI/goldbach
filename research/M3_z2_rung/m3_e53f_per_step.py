"""探索性（非预注册）：松弛律（下一乘数在 [3, a) 中与 a 互素均匀）下，逐步核对
E[X_t] − Ω(a_{t−1}) 与启发值 H_t = Σ_{q^k | P_t，排除 a_{t−1} 的部分} 2/q^k 的差是否 O(1)、且不随 t 增长。"""
import random, json, math
from sympy import factorint
from m3_e49_coprime import strings
def Omega_gcd(n, P):
    g = math.gcd(n, P); return sum(factorint(g).values()) if g > 1 else 0
rng = random.Random(5356); rows = {}
for _ in range(120):
    p = rng.choice([x for x in range(200, 1500) if all(x % d for d in range(2, int(x ** .5) + 1))])
    a = [p]
    while a[-1] > 6:
        c = [b for b in range(3, a[-1]) if math.gcd(b, a[-1]) == 1]
        a.append(rng.choice(c))
    for t in range(2, len(a)):
        pre = a[:t + 1]
        if pre[-1] < 4 or 2 ** t * pre[-1] > 60000: continue
        P = math.prod(pre); S = strings(pre, True)
        EX = sum(Omega_gcd(M * (M + 1), P) for (M, _, _) in S) / len(S)
        Om = sum(factorint(pre[t - 1]).values())
        fP = factorint(P); fa = factorint(pre[t - 1])
        H = sum(2 / q ** k for q, v in fP.items() for k in range(1 + fa.get(q, 0), v + 1))
        rows.setdefault(t, []).append(EX - Om - H)
out = {t: dict(n=len(v), mean=round(sum(v) / len(v), 3), max=round(max(v), 3), min=round(min(v), 3)) for t, v in sorted(rows.items())}
print(json.dumps(out))
