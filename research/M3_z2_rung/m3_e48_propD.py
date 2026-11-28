"""M3-E48：命题 D 的数值检验（预注册：DESIGN.md 第 33 节）。
候选父节点 τ = (a', x/a')（a' | w^±，a' x' >= N，thue(a', x') = a_t，非谱系父节点）在脊柱上的概率，
与同一 a'、随机 x'' 的匹配对照比较；按 a' 的二进区间汇总。
"""
import sys, json, math, random, time
import gmpy2
from gmpy2 import mpz
from m3_e47_large_N import Forest, thue_cf

def run(N, p, S, seed, tlimit):
    rng = random.Random(seed); F = Forest(N, p); lo = (N + p - 1) // p; t0 = time.time()
    bins = {}  # k -> [cand, cand_sn, ctrl, ctrl_sn, sum a'*1]
    Estar = 0; Sinv = 0.0; m = 0
    for s in range(S):
        if time.time() - t0 > tlimit: break
        n = mpz(rng.randrange(lo, N)); a, x = p, n; path = [(a, x)]
        while a > 1 and a * x >= N:
            v = thue_cf(a, x, N); a, x = v, (a * x) % N; path.append((a, x))
        m += 1
        for t in range(1, len(path)):
            a, x = path[t]
            if a < 2: continue
            for (ap, xp) in F.parents(a, x):
                if (ap, xp) == path[t - 1]: continue
                k = int(ap).bit_length() - 1
                b = bins.setdefault(k, [0, 0, 0, 0])
                sn = F.is_sn(ap, xp)
                b[0] += 1; b[1] += sn; Estar += sn; Sinv += 1.0 / int(ap)
                lo2 = (N + ap - 1) // ap
                xr = mpz(rng.randrange(lo2, N))
                b[2] += 1; b[3] += F.is_sn(ap, xr)
    out = {}
    for k, (c, cs, r, rs) in sorted(bins.items()):
        mid = 1.5 * 2 ** k
        out[k] = dict(cand=c, P_cand=round(cs / c, 4), ctrl=r, P_ctrl=round(rs / r, 4),
                      ratio=(round((cs / c) / (rs / r), 3) if rs else None),
                      aP_ctrl=round(mid * rs / r, 3), aP_cand=round(mid * cs / c, 3))
    return dict(p=p, samples=m, secs=round(time.time() - t0, 1), E_star=round(Estar / m, 4),
                sum_inv_cand=round(Sinv / m, 4), C_eff=round(Estar / Sinv, 4) if Sinv else None, bins=out)

if __name__ == "__main__":
    import random as R
    rr = R.Random(20261128)
    N = gmpy2.next_prime(mpz(rr.getrandbits(1024)) | (mpz(1) << 1023))
    for p in map(int, sys.argv[1].split(",")):
        print(json.dumps(run(N, p, int(sys.argv[2]), seed=p, tlimit=float(sys.argv[3]))), flush=True)
