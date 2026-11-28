"""M3-E49：互素相邻乘数情形下的反向确定性与尾数等分布（预注册：DESIGN.md 第 34 节）。"""
import random, json, math
from fractions import Fraction as Fr

def is_prime(n):
    if n < 2: return False
    i = 2
    while i * i <= n:
        if n % i == 0: return False
        i += 1
    return True

def prime_factors(n):
    f = set(); d = 2
    while d * d <= n:
        while n % d == 0: f.add(d); n //= d
        d += 1
    if n > 1: f.add(n)
    return f

def gen_seq(rng, t):
    while True:
        p = rng.choice([q for q in range(31, 1022) if is_prime(q)])
        a = [p]
        ok = True
        for _ in range(t):
            lo = max(3, a[-1] // 4)
            cands = [b for b in range(lo, a[-1]) if math.gcd(b, a[-1]) == 1]
            if not cands: ok = False; break
            a.append(rng.choice(cands))
        if ok: return a

def strings(a, relaxed2):
    """反向枚举：返回 [(M, eps, digits)]。"""
    t = len(a) - 1; out = []
    inv = [pow(a[i + 1], -1, a[i]) for i in range(t)]
    for e in range(2 ** t):
        eps = [(e >> i) & 1 for i in range(t)]
        rng_ct = range(1, a[t] - 1) if relaxed2 else range(a[t])
        for ct in rng_ct:
            c = [0] * (t + 1); c[t] = ct; good = True
            for i in range(t - 1, -1, -1):
                c[i] = ((-c[i + 1] - eps[i]) * inv[i]) % a[i]
                if relaxed2 and not (1 <= c[i] <= a[i] - 2): good = False; break
            if good:
                M = 0
                for i in range(t + 1): M = M * a[i] + c[i]
                out.append((M, eps, c))
    return out

def check_direct(a, M):
    """在 θ = (M + 1/2)/P 处逐步验证 ||a_{i+1} u_i|| < 1/a_i（u_i = {P_{i-1} θ}）。"""
    P = 1
    for x in a: P *= x
    th = Fr(2 * M + 1, 2 * P); Pm = 1
    for i in range(len(a) - 1):
        u = (Pm * th) % 1
        v = (a[i + 1] * u) % 1
        if min(v, 1 - v) >= Fr(1, a[i]): return False
        Pm *= a[i]
    return True

def main():
    rng = random.Random(20261128)
    res = dict(H1_pure_count_viol=0, H1_relaxed2_range_viol=0, direct_check_viol=0, direct_checked=0,
               brute_checked=0, brute_viol=0, H2_cases=0, H2_viol=0, H2_max_excess_over_bound=-1e9,
               H3_cases=0, H3_viol=0, H3_maxD=-1e9, H3_by_ell={})
    worst = []
    for it in range(400):
        t = rng.randint(3, 8)
        a = gen_seq(rng, t)
        P = 1
        for x in a: P *= x
        pure = strings(a, False)
        if len(pure) != 2 ** t * a[t]: res['H1_pure_count_viol'] += 1
        rel = strings(a, True)
        lo, hi = 2 ** t * (a[t] - 2 - 2 * t), 2 ** t * (a[t] - 2)
        if not (lo <= len(rel) <= hi): res['H1_relaxed2_range_viol'] += 1
        for (M, _, _) in rng.sample(pure, min(20, len(pure))):
            res['direct_checked'] += 1
            if not check_direct(a, M): res['direct_check_viol'] += 1
        if P <= 3 * 10 ** 5:
            Ms = set(M for M, _, _ in pure)
            cnt = sum(1 for M in range(P) if check_direct(a, M))
            res['brute_checked'] += 1
            if cnt != len(Ms): res['brute_viol'] += 1
        if not rel: continue
        R = len(rel)
        for j in range(t - 1):
            for r in prime_factors(a[j]):
                if any(a[k] % r == 0 for k in range(j + 1, t + 1)): continue
                pr = sum(1 for M, _, _ in rel if M % r == 0) / R
                ell = t - j - 2
                if ell in (0, 1):
                    res['H2_cases'] += 1
                    bound = 1 / r + 2 / (a[t] - 2)
                    res['H2_max_excess_over_bound'] = max(res['H2_max_excess_over_bound'], pr - bound)
                    if pr > bound + 1e-12:
                        res['H2_viol'] += 1; worst.append(dict(kind='H2', a=a, j=j, r=r, P=round(pr, 4), bound=round(bound, 4), R=R))
                elif ell >= 2:
                    res['H3_cases'] += 1
                    D = (pr - 1 / r) * 2 ** (ell / 2) * a[t]
                    res['H3_maxD'] = max(res['H3_maxD'], D)
                    b = res['H3_by_ell'].setdefault(ell, [0, -1e9]); b[0] += 1; b[1] = max(b[1], round(D, 3))
                    if D > 3: res['H3_viol'] += 1; worst.append(dict(kind='H3', a=a, j=j, r=r, P=round(pr, 4), D=round(D, 3), R=R))
    res['H2_max_excess_over_bound'] = round(res['H2_max_excess_over_bound'], 4)
    res['H3_maxD'] = round(res['H3_maxD'], 3)
    print(json.dumps(res))
    for w in worst[:12]: print(json.dumps(w))

if __name__ == "__main__":
    main()
