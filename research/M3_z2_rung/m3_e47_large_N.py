"""M3-E47：定理所需区域（N 为 1024 位素数）中的 Λ*（预注册：DESIGN.md 第 32 节）。

对随机 n（pn >= N）沿实际脊柱，枚举每个节点 σ = (a, x) 的候选 B 型父节点 a' ∈ (a, p]，
a' | w 或 a' | N - w（w = a x mod N），保留 thue(a', x/a') = a 且 a' * x' >= N 的那些，
再用带记忆的反向 DFS 判定父节点是否在某条脊柱上（能否回溯到乘数为 p 的源）。
"""
import sys, json, math, random, time
import gmpy2
from gmpy2 import mpz
sys.setrecursionlimit(10000)

def primes_upto(m):
    s = bytearray([1]) * (m + 1); s[0] = s[1] = 0
    for i in range(2, int(m ** 0.5) + 1):
        if s[i]: s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(m + 1) if s[i]]

def thue_cf(a, x, N):
    qs = [1]; num, den = N, x; q_prev, q_cur = mpz(0), mpz(1)
    while den:
        t = num // den; num, den = den, num - t * den
        q_prev, q_cur = q_cur, t * q_cur + q_prev
        if q_cur >= a: break
        qs.append(q_cur)
    for v in qs:
        u = (x * v) % N
        if u * a < N or (N - u) * a < N: return int(v)
    raise RuntimeError

def smooth_fact(w, primes):
    f = {}
    for r in primes:
        if w % r == 0:
            e = 0
            while w % r == 0: w //= r; e += 1
            f[r] = e
    return f

def divisors_in(f, lo, hi):
    ds = [1]
    for r, e in f.items():
        nd = []
        for d in ds:
            m = d
            for _ in range(e):
                m *= r
                if m > hi: break
                nd.append(m)
        ds += nd
    return [d for d in ds if lo < d <= hi]

class Forest:
    def __init__(self, N, p):
        self.N, self.p = N, p
        self.primes = primes_upto(p)
        self.memo_sn = {}
        self.memo_par = {}
        self.calls = 0
    def parents(self, a, x):
        key = (a, x)
        if key in self.memo_par: return self.memo_par[key]
        N, p = self.N, self.p
        w = (a * x) % N
        cands = set()
        for z in (w, N - w):
            for d in divisors_in(smooth_fact(z, self.primes), a, p):
                cands.add(d)
        res = []
        for ap in sorted(cands):
            xp = (x * gmpy2.invert(ap, N)) % N
            if ap * xp < N: continue
            try:
                if thue_cf(ap, xp, N) != a: continue
            except RuntimeError:
                continue
            res.append((ap, xp))
        self.memo_par[key] = res
        return res
    def is_sn(self, a, x):
        key = (a, x)
        if key in self.memo_sn: return self.memo_sn[key]
        self.calls += 1
        if a == self.p:
            r = self.p * x >= self.N
        else:
            r = any(self.is_sn(ap, xp) for ap, xp in self.parents(a, x))
        self.memo_sn[key] = r
        return r

def run(N, p, S, seed, tlimit):
    rng = random.Random(seed); F = Forest(N, p); lo = (N + p - 1) // p
    burdens = []; extras = []; lens = []; t0 = time.time(); indeg_hist = {}
    for s in range(S):
        if time.time() - t0 > tlimit: break
        n = mpz(rng.randrange(lo, N))
        a, x = p, n; path = [(a, x)]
        while a > 1 and a * x >= N:
            v = thue_cf(a, x, N); a, x = v, (a * x) % N; path.append((a, x))
        B = 0.0; E = 0
        for t in range(1, len(path)):
            a, x = path[t]
            if a < 2: continue
            sp = [pp for pp in F.parents(a, x) if F.is_sn(*pp)]
            assert path[t - 1] in sp, ("lineage parent missing", t)
            d = len(sp); B += math.log(d); E += d - 1
            indeg_hist[d] = indeg_hist.get(d, 0) + 1
        burdens.append(B); extras.append(E); lens.append(len(path) - 1)
    m = len(burdens); lp = math.log(p)
    Lam = sum(burdens) / m
    sb = sorted(burdens)
    return dict(p=p, samples=m, secs=round(time.time() - t0, 1), mean_len=round(sum(lens) / m, 3),
                Lambda_star=round(Lam, 4), Lambda_over_lnp=round(Lam / lp, 4),
                se=round((sum((b - Lam) ** 2 for b in burdens) / (m - 1)) ** 0.5 / m ** 0.5, 4),
                E_star=round(sum(extras) / m, 4), T95=round(sb[int(0.95 * m)], 3),
                indeg_star_hist=dict(sorted(indeg_hist.items())), sn_calls=F.calls)

if __name__ == "__main__":
    rng = random.Random(20261128)
    N = gmpy2.next_prime(mpz(rng.getrandbits(1024)) | (mpz(1) << 1023))
    S = int(sys.argv[2]); tl = float(sys.argv[3])
    for p in map(int, sys.argv[1].split(",")):
        print(json.dumps(run(N, p, S, seed=p, tlimit=tl)), flush=True)
