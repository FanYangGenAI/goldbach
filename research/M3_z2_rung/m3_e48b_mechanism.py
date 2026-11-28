"""M3-E48b（探索性，非预注册）：候选父节点为何常在脊柱上？按子节点深度 t、以及 a' 与谱系父乘数 a_{t-1} 的大小关系，
统计 P(τ ∈ SN | 候选)；对在脊柱上的候选，找一条见证路径（反向 DFS 的第一条），记录其深度 d' 与 t-1 之差，
以及见证源 n' 的乘数序列与 n 的前缀相差的位置数。"""
import sys, json, random
import gmpy2
from gmpy2 import mpz
from m3_e47_large_N import Forest, thue_cf
def witness(F, a, x, memo):
    key = (a, x)
    if key in memo: return memo[key]
    if a == F.p: memo[key] = [a]; return memo[key]
    for ap, xp in F.parents(a, x):
        if F.is_sn(ap, xp):
            w = witness(F, ap, xp, memo)
            if w is not None: memo[key] = w + [a]; return memo[key]
    memo[key] = None; return None
rr = random.Random(20261128)
N = gmpy2.next_prime(mpz(rr.getrandbits(1024)) | (mpz(1) << 1023))
for p in map(int, sys.argv[1].split(",")):
    rng = random.Random(p); F = Forest(N, p); lo = (N + p - 1) // p; memo = {}
    by_t = {}; by_rel = {'a<a_prev': [0, 0], 'a>a_prev': [0, 0]}; dd = {}; diff = {}
    for s in range(int(sys.argv[2])):
        n = mpz(rng.randrange(lo, N)); a, x = p, n; path = [(a, x)]
        while a > 1 and a * x >= N:
            v = thue_cf(a, x, N); a, x = v, (a * x) % N; path.append((a, x))
        seq = [q for q, _ in path]
        for t in range(1, len(path)):
            a, x = path[t]
            if a < 2: continue
            for ap, xp in F.parents(a, x):
                if (ap, xp) == path[t - 1]: continue
                sn = F.is_sn(ap, xp)
                b = by_t.setdefault(t, [0, 0]); b[0] += 1; b[1] += sn
                r = by_rel['a<a_prev' if ap < seq[t - 1] else 'a>a_prev']; r[0] += 1; r[1] += sn
                if sn:
                    w = witness(F, ap, xp, memo)  # multipliers from p down to a'
                    d = len(w) - 1 - (t - 1); dd[d] = dd.get(d, 0) + 1
                    if len(w) == t:  # same depth: compare with n's prefix seq[0..t-1]
                        k = sum(1 for i in range(t) if w[i] != seq[i]); diff[k] = diff.get(k, 0) + 1
    print(json.dumps(dict(p=p, P_by_child_depth={t: (c, round(s_ / c, 3)) for t, (c, s_) in sorted(by_t.items()) if c >= 50},
                          P_by_relation={k: (c, round(s_ / c, 3)) for k, (c, s_) in by_rel.items()},
                          witness_depth_minus_lineage=dict(sorted(dd.items())),
                          same_depth_num_differing_positions=dict(sorted(diff.items())))), flush=True)
