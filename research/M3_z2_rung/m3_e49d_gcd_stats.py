"""M3-E49d（探索性）：实际脊柱（1024 位 N）上相邻乘数的 gcd 统计：互素步的比例、整条路径相邻互素的比例、E[gcd]、E[log gcd]。"""
import sys, json, math, random
import gmpy2
from gmpy2 import mpz
from m3_e47_large_N import thue_cf
rr = random.Random(20261128)
N = gmpy2.next_prime(mpz(rr.getrandbits(1024)) | (mpz(1) << 1023))
for p in map(int, sys.argv[1].split(",")):
    rng = random.Random(p); lo = (N + p - 1) // p; S = int(sys.argv[2])
    steps = cop = allcop = 0; sg = slg = 0.0; hist = {}
    for _ in range(S):
        n = mpz(rng.randrange(lo, N)); a, x = p, n; seq = [p]
        while a > 1 and a * x >= N:
            v = thue_cf(a, x, N); a, x = v, (a * x) % N
            if a >= 2: seq.append(a)
        ok = True
        for i in range(len(seq) - 1):
            g = math.gcd(seq[i], seq[i + 1]); steps += 1; cop += (g == 1); sg += g; slg += math.log(g)
            hist[min(g, 10)] = hist.get(min(g, 10), 0) + 1
            if g > 1: ok = False
        allcop += ok
    print(json.dumps(dict(p=p, S=S, frac_coprime_steps=round(cop / steps, 4), frac_paths_all_coprime=round(allcop / S, 4),
                          mean_gcd=round(sg / steps, 3), mean_log_gcd=round(slg / steps, 4),
                          gcd_hist_capped10={k: round(v / steps, 4) for k, v in sorted(hist.items())})), flush=True)
