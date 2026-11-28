"""M3-E47d（探索性，非预注册）：1024 位 N 上，终点节点的祖先数 anc(σ_L(n))（按 n 加权，即尺寸偏置）。
anc(τ) = [τ 是源] + Σ_{脊柱父节点 π} anc(π)，用带记忆的反向递归精确计算。
E_n[anc(σ_L)] 就是第 58 节熵代价截断（等价于二阶矩路线）所需的量。"""
import sys, json, math, random
import gmpy2
from gmpy2 import mpz
from m3_e47_large_N import Forest, thue_cf
sys.setrecursionlimit(100000)
rr = random.Random(20261128)
N = gmpy2.next_prime(mpz(rr.getrandbits(1024)) | (mpz(1) << 1023))
for spec in sys.argv[1].split(","):
    p, S = map(int, spec.split(":"))
    rng = random.Random(p); F = Forest(N, p); lo = (N + p - 1) // p; memo = {}
    def anc(a, x):
        k = (a, x)
        if k in memo: return memo[k]
        if a == p: r = 1 if p * x >= N else 0
        else: r = sum(anc(ap, xp) for ap, xp in F.parents(a, x) if F.is_sn(ap, xp))
        memo[k] = r; return r
    vals = []; mids = []
    for _ in range(S):
        n = mpz(rng.randrange(lo, N)); a, x = p, n; path = [(a, x)]
        while a > 1 and a * x >= N:
            v = thue_cf(a, x, N); a, x = v, (a * x) % N; path.append((a, x))
        last = path[-1] if path[-1][0] >= 2 else path[-2]
        vals.append(anc(*last))
    vals.sort(); m = len(vals); mean = sum(vals) / m
    print(json.dumps(dict(p=p, S=m, mean_terminal_anc=round(mean, 3), median=vals[m // 2], q90=vals[int(0.9 * m)],
                          q99=vals[int(0.99 * m)], max=vals[-1], mean_over_lnp=round(mean / math.log(p), 3),
                          log_mean_over_log_p=round(math.log(mean) / math.log(p), 3))), flush=True)
