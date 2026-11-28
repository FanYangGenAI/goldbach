"""M3-E46：只计"脊柱父节点"的真实入度 indeg*（预注册：DESIGN.md 第 31 节）。

对每个满足 pn >= N 的 n 构造实际脊柱（与 E37 相同：a, x = p, n；当 a > 1 且 a*x >= N 时
v = thue(a, x)，(a, x) <- (v, a*x mod N)）。节点 = (a, x)。记录所有脊柱节点的父节点集合，
得到 anc(σ)（脊柱经过 σ 的 n 的个数）、indeg*(σ)（不同的脊柱父节点个数），以及
负担 B*(n) = Σ_{t>=1} log indeg*(σ_t(n))。核对引理 L*：B* <= T 的祖先数 <= e^T。
"""
import sys, json, math
from collections import defaultdict

def is_prime(n):
    if n < 2: return False
    i = 2
    while i * i <= n:
        if n % i == 0: return False
        i += 1
    return True

def thue_cf(a, x, N):
    qs = [1]; num, den = N, x; q_prev, q_cur = 0, 1
    while den:
        t = num // den; num, den = den, num - t * den
        q_prev, q_cur = q_cur, t * q_cur + q_prev
        if q_cur >= a: break
        qs.append(q_cur)
    for v in qs:
        u = (x * v) % N
        if u * a < N: return v
        if (N - u) * a < N: return v
    raise RuntimeError

def thue_brute(a, x, N):
    for v in range(1, a):
        u = (x * v) % N
        if u * a < N or (N - u) * a < N: return v
    raise RuntimeError

def run(N, p):
    assert is_prime(N) and is_prime(p)
    lo = (N + p - 1) // p
    spines = []
    parents = defaultdict(set)
    anc = defaultdict(int)
    mism = 0
    for n in range(lo, N):
        a, x = p, n; path = [(a, x)]
        while a > 1 and a * x >= N:
            v = thue_cf(a, x, N)
            if n - lo < 300 and thue_brute(a, x, N) != v: mism += 1
            a, x = v, (a * x) % N
            path.append((a, x))
        spines.append(path)
        for t in range(1, len(path)):
            parents[path[t]].add(path[t - 1])
        for s in path: anc[s] += 1
    burdens = []
    for path in spines:
        burdens.append(sum(math.log(len(parents[s])) for s in path[1:]))
    S = len(burdens)
    Lam = sum(burdens) / S
    sb = sorted(burdens); T95 = sb[int(0.95 * S)]
    lens = [len(pth) - 1 for pth in spines]
    # 引理 L* 核对：每个节点上 B* <= T95 的祖先数 <= e^T95
    cov_anc = defaultdict(int)
    for path, b in zip(spines, burdens):
        if b <= T95 + 1e-12:
            for s in path: cov_anc[s] += 1
    viol = sum(1 for s, c in cov_anc.items() if c > math.exp(T95) + 1e-9)
    nodes = [s for s in anc if s[0] < p]
    ind = [len(parents[s]) for s in nodes]
    extra_mean_path = sum(sum(len(parents[s]) - 1 for s in pth[1:]) for pth in spines) / S
    lp = math.log(p)
    return dict(N=N, p=p, sources=S, mean_len=round(sum(lens) / S, 3), Lambda_star=round(Lam, 4),
                Lambda_over_lnp=round(Lam / lp, 4), Lambda_over_lnp_lnlnp=round(Lam / (lp * math.log(lp)), 4),
                T95=round(T95, 3), exp_T95=round(math.exp(T95), 1), max_cov_anc=max(cov_anc.values()),
                lemmaLstar_violations=viol, max_anc_nonroot=max(anc[s] for s in nodes),
                max_indeg_star=max(ind), mean_indeg_star_nodes=round(sum(ind) / len(ind), 4),
                mean_extra_parents_along_path=round(extra_mean_path, 4), cf_brute_mismatch=mism)

if __name__ == "__main__":
    for N in map(int, sys.argv[1].split(",")):
        for p in map(int, sys.argv[2].split(",")):
            if N == 1048573 and p == 251 and len(sys.argv) < 4: continue
            print(json.dumps(run(N, p)), flush=True)
