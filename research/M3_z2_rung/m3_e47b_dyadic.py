"""M3-E47b（探索性，非预注册）：实际脊柱在每个二进尺度 [2^k, 2^{k+1}) 上平均停留的步数（命题 D(a) 的数值检验）。"""
import sys, json, random
import gmpy2
from gmpy2 import mpz
from m3_e47_large_N import thue_cf
rng = random.Random(20261128)
N = gmpy2.next_prime(mpz(rng.getrandbits(1024)) | (mpz(1) << 1023))
for p in map(int, sys.argv[1].split(",")):
    r = random.Random(p); lo = (N + p - 1) // p; S = int(sys.argv[2]); cnt = {}; slow = 0; steps = 0
    for _ in range(S):
        n = mpz(r.randrange(lo, N)); a, x = p, n
        while a > 1 and a * x >= N:
            v = thue_cf(a, x, N); steps += 1; slow += (2 * v >= a); a, x = v, (a * x) % N
            if a >= 2: k = a.bit_length() - 1; cnt[k] = cnt.get(k, 0) + 1
    print(json.dumps(dict(p=p, S=S, slow_step_frac=round(slow / steps, 4),
                          visits_per_dyadic={k: round(c / S, 3) for k, c in sorted(cnt.items())})), flush=True)
