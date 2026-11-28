"""M3-E51b（探索性，非预注册）：ℓ = 2 时固定 (a_{t-4}, a_{t-3}, a_{t-1}, a_t, q)，让 a_{t-2} 取遍 (a_{t-1}, a_{t-3}) 中
与两邻互素、且不被 q 整除的值，计算 |P(q | M(M+1)) − 2/q| 的平均与最大值（纯 Bohr，精确枚举），与 (log a_{t-1})²/a_t 比较。
第一组为第 64.3 节反例所在的参数族。"""
import math, json, random
from m3_e49_coprime import strings, is_prime
def run(prefix, a3, a1, at, q):
    vals = []; worst = None
    for b in range(a1 + 1, a3):
        if math.gcd(b, a3) != 1 or math.gcd(b, a1) != 1 or b % q == 0: continue
        a = prefix + [a3, b, a1, at]
        if any(math.gcd(a[i], a[i + 1]) != 1 for i in range(len(a) - 1)): continue
        pure = strings(a, False); R = len(pure)
        P = sum(1 for M, _, _ in pure if M % q == 0 or (M + 1) % q == 0) / R
        d = abs(P - 2 / q); vals.append(d)
        if worst is None or d > worst[0]: worst = (round(d, 4), b)
    return dict(prefix=prefix, a3=a3, a1=a1, at=at, q=q, n=len(vals), mean_dev=round(sum(vals) / len(vals), 5),
                max_dev=worst, log2_over_at=round(math.log(a1) ** 2 / at, 4))
out = [run([397, 394], 393, 100, 99, 197)]
rng = random.Random(8)
while len(out) < 7:
    a1 = rng.randint(15, 60); at = rng.randint(5, a1 - 1)
    if math.gcd(a1, at) != 1: continue
    a3 = rng.randint(a1 + 40, a1 + 200)
    if math.gcd(a3, a1) == 1: pass
    q = rng.choice([x for x in range(5, 120) if is_prime(x) and a1 % x and at % x and a3 % x])
    m = rng.randint(a3 // q + 1, (a3 + 300) // q)
    a4 = q * m
    if a4 <= a3 or math.gcd(a4, a3) != 1: continue
    p = next(x for x in range(a4 + 1, 5 * a4) if is_prime(x))
    out.append(run([p, a4], a3, a1, at, q))
for o in out: print(json.dumps(o))
