"""M3-E49e（探索性，非预注册）：二阶矩路线中大素数的相容性。
取两个相邻互素、末乘数相同的随机序列 s、s''，素数 r 同时整除两者的某个乘数（且不整除各自更后面的乘数）。
在松弛″均匀测度下计算 κ := r·Σ_x D(x)·D''(x)，其中 D、D'' 是 M mod r 的分布。κ ≈ 1 表示"相容概率 ≈ 1/r"，
正好抵消 gcd 中的因子 r。"""
import random, json, math
from m3_e49_coprime import strings, prime_factors, is_prime
def gen_to(rng, t, last, r, pos):
    # 生成 a_0 > ... > a_t = last，相邻互素，且 r | a_pos，r 不整除 a_{pos+1..t}
    for _ in range(2000):
        p = rng.choice([q for q in range(max(last + 10, 2 * r + 1), 1022) if is_prime(q)])
        a = [p]; ok = True
        for i in range(1, t):
            lo = max(last + (t - i), a[-1] // 3)
            cands = [b for b in range(lo, a[-1]) if math.gcd(b, a[-1]) == 1 and b > last
                     and ((b % r == 0) if i == pos else (b % r != 0 or i < pos))]
            if not cands: ok = False; break
            a.append(rng.choice(cands))
        if not ok or math.gcd(a[-1], last) != 1 or last % r == 0: continue
        a.append(last)
        if all(a[i] > a[i + 1] for i in range(t)): return a
    return None
rng = random.Random(4242); ks = []; worst = []
for it in range(300):
    last = rng.randint(3, 40); r = rng.choice([q for q in range(5, 60) if is_prime(q) and last % q])
    t1, t2 = rng.randint(3, 6), rng.randint(3, 6)
    s1 = gen_to(rng, t1, last, r, rng.randint(1, t1 - 2)); s2 = gen_to(rng, t2, last, r, rng.randint(1, t2 - 2))
    if not s1 or not s2: continue
    D1 = {}; D2 = {}
    for M, _, _ in strings(s1, True): D1[M % r] = D1.get(M % r, 0) + 1
    for M, _, _ in strings(s2, True): D2[M % r] = D2.get(M % r, 0) + 1
    n1, n2 = sum(D1.values()), sum(D2.values())
    if not n1 or not n2: continue
    k = r * sum(D1[x] / n1 * D2.get(x, 0) / n2 for x in D1)
    ks.append(k); worst.append((round(k, 3), s1, s2, r))
ks.sort()
print(json.dumps(dict(pairs=len(ks), mean_kappa=round(sum(ks) / len(ks), 3), median=round(ks[len(ks) // 2], 3),
                      q90=round(ks[int(0.9 * len(ks))], 3), max=round(ks[-1], 3))))
for w in sorted(worst, key=lambda w: -w[0])[:4]: print(json.dumps(w))
