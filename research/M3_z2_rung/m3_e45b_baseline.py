"""M3-E45b（探索性，非预注册）：Delta_i 的随机基线。
把实际的 M_i 换成在 [0, P_i) 上均匀、且满足系统性约束 M' ≡ 0 或 -1 (mod a_{i-1}) 的随机 M'，
计算 Delta'_i = Omega(gcd(M'(M'+1), P_i)) - Omega(a_{i-1})，按深度与实际的 Delta_i 对比。"""
import sys, json, random
import gmpy2
from gmpy2 import mpz
from m3_e45_logburden import primes_upto, thue_cf, Omega_small
def run(N, p, S, seed):
    rng = random.Random(seed); primes = primes_upto(p); lo = (N + p - 1) // p
    tab = {}
    for s in range(S):
        n = mpz(rng.randrange(lo, N)); a, x = p, n; P = mpz(p); i = 0
        while a > 1 and a * x >= N:
            v, e = thue_cf(a, x, N); prev = a; x = (a * x) % N; a = v; i += 1
            if a < 2: break
            P *= a; M = (P * n) // N
            X = Omega_small(gmpy2.gcd(M, P) * gmpy2.gcd(M + 1, P), primes)
            oa = Omega_small(mpz(prev), primes)
            t = mpz(rng.randrange(0, int(P // prev))); Mr = (prev * t - rng.randrange(2)) % P
            Xr = Omega_small(gmpy2.gcd(Mr, P) * gmpy2.gcd(Mr + 1, P), primes)
            d = tab.setdefault(i, [0, 0, 0]); d[0] += 1; d[1] += X - oa; d[2] += Xr - oa
    return {k: (d[0], round(d[1] / d[0], 3), round(d[2] / d[0], 3)) for k, d in sorted(tab.items()) if d[0] >= 100}
if __name__ == "__main__":
    rng = random.Random(20261128)
    N = gmpy2.next_prime(mpz(rng.getrandbits(1024)) | (mpz(1) << 1023))
    for p in map(int, sys.argv[1].split(",")):
        print(json.dumps(dict(p=p, depth_cnt_Delta_Delta_rand=run(N, p, int(sys.argv[2]), seed=p))), flush=True)
