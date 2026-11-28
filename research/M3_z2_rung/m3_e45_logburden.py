"""M3-E45：实际脊柱上入度的"对数负担"（预注册：DESIGN.md 第 30 节）。

N 为固定的 1024 位素数。对根乘数 p，抽 n（pn >= N），沿实际脊柱（thue 取最小 v，用连分数实现，
并与暴力 thue 交叉核对）计算每个深度 i >= 1 节点的：
  indeg_i  = #{a' in (a_i, p] : a' | w_i 或 a' | N - w_i}，w_i = a_i x_i mod N
  对照     = 同一 a_i、均匀随机 w 的同一统计量
  X_i      = Omega(gcd(M_i (M_i + 1), P_i))，M_i = floor(P_i n / N)，P_i = a_0...a_i
  Delta_i  = X_i - Omega(a_{i-1})
"""
import sys, json, math, random
import gmpy2
from gmpy2 import mpz

def primes_upto(m):
    s = bytearray([1]) * (m + 1); s[0] = s[1] = 0
    for i in range(2, int(m ** 0.5) + 1):
        if s[i]: s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(m + 1) if s[i]]

def thue_brute(a, x, N):
    for v in range(1, a):
        u = (x * v) % N
        if u * a < N: return v, 1
        if (N - u) * a < N: return v, -1
    raise RuntimeError

def thue_cf(a, x, N):
    # 最小 v < a 使 ||v x / N|| < 1/a；它必是 x/N 的连分数渐近分母
    # 标准 Euclid：x/N = [0; t1, t2, ...]
    num, den = int(x), int(N)
    qs = [1]; q_prev, q_cur = 0, 1
    a0 = num // den; num, den = den, num - a0 * den  # 现在 num/den = 1/(x/N - a0)
    while den:
        t = num // den; num, den = den, num - t * den
        q_prev, q_cur = q_cur, t * q_cur + q_prev
        if q_cur >= a: break
        qs.append(q_cur)
    for v in qs:
        if v >= a: break
        u = (x * v) % N
        if u * a < N: return v, 1
        if (N - u) * a < N: return v, -1
    raise RuntimeError("no convergent")

def smooth_fact(w, primes):
    f = {}
    for r in primes:
        if w % r == 0:
            e = 0
            while w % r == 0: w //= r; e += 1
            f[r] = e
    return f

def divisors_upto(f, lim):
    ds = [1]
    for r, e in f.items():
        nd = []
        for d in ds:
            m = d
            for _ in range(e):
                m *= r
                if m > lim: break
                nd.append(m)
        ds += nd
    return ds

def indeg_bound(w, N, a, p, primes):
    c = 0
    for z in (w, N - w):
        f = smooth_fact(z, primes)
        c += sum(1 for d in divisors_upto(f, p) if d > a)
    return c

def Omega_small(m, primes):
    return sum(smooth_fact(m, primes).values())

def run(N, p, S, seed, check_brute=200):
    rng = random.Random(seed)
    primes = primes_upto(p)
    lo = (N + p - 1) // p
    sp_sum = un_sum = 0.0; burdens = []; lens = []
    by_depth = {}  # i -> [count, sum Delta, sum X, sum Omega(a_{i-1}), sum log1p spine, sum log1p unif]
    brute_mismatch = 0
    for s in range(S):
        n = mpz(rng.randrange(lo, N))
        a, x = p, n; P = mpz(p); i = 0; burden = 0.0; prev_a = None
        while a > 1 and a * x >= N:
            if s < check_brute and a <= 400:
                if thue_brute(int(a), int(x), int(N)) != thue_cf(a, x, N): brute_mismatch += 1
            v, e = thue_cf(a, x, N)
            prev_a = a; x = (a * x) % N; a = v; i += 1
            if a < 2: break
            P *= a
            w = (a * x) % N
            dg = indeg_bound(w, N, a, p, primes)
            wu = mpz(rng.randrange(1, N))
            du = indeg_bound(wu, N, a, p, primes)
            M = (P * n) // N
            assert (P * n - M * N) == w  # w = P_i n mod N
            g = gmpy2.gcd(M, P) * gmpy2.gcd(M + 1, P)
            X = Omega_small(g, primes)
            oa = Omega_small(mpz(prev_a), primes)
            assert X >= oa, (X, oa)
            ls, lu = math.log1p(dg), math.log1p(du)
            sp_sum += ls; un_sum += lu; burden += ls
            d = by_depth.setdefault(i, [0, 0, 0, 0, 0.0, 0.0])
            d[0] += 1; d[1] += X - oa; d[2] += X; d[3] += oa; d[4] += ls; d[5] += lu
        burdens.append(burden); lens.append(i)
    burdens.sort()
    T95 = burdens[int(0.95 * S)]
    G = sum(burdens) / S
    depths = sorted(k for k, d in by_depth.items() if d[0] >= 300)
    xs = depths; ys = [by_depth[k][1] / by_depth[k][0] for k in depths]
    if len(xs) >= 2:
        mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
        slope = sum((u - mx) * (v - my) for u, v in zip(xs, ys)) / sum((u - mx) ** 2 for u in xs)
    else:
        slope = float('nan')
    lp = math.log(p)
    return dict(p=p, S=S, mean_len=sum(lens) / S, R1=sp_sum / un_sum, G=G, T95=T95,
                G_over_lnp_lnlnp=G / (lp * math.log(lp)), G_over_lnp2=G / lp ** 2,
                Delta_slope=slope, brute_mismatch=brute_mismatch,
                depth_table={k: dict(cnt=d[0], Delta=round(d[1] / d[0], 3), X=round(d[2] / d[0], 3),
                                     Om_prev=round(d[3] / d[0], 3), log1p_spine=round(d[4] / d[0], 3),
                                     log1p_unif=round(d[5] / d[0], 3)) for k, d in sorted(by_depth.items())})

if __name__ == "__main__":
    rng = random.Random(20261128)
    N = gmpy2.next_prime(mpz(rng.getrandbits(1024)) | (mpz(1) << 1023))
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    print(json.dumps(dict(N_bits=int(N.bit_length()), N_mod_1e9=int(N % 10 ** 9))), flush=True)
    for p in map(int, sys.argv[1].split(",")):
        print(json.dumps(run(N, p, S, seed=p)), flush=True)
