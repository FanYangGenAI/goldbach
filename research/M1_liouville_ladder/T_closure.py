"""M1-E6（探索性）：N=2^k 奇数子纤维刚性（对一切完全积性 g）推出哪些 T(t)：g(t mod N)=g(t)，t 奇，N<t<3N。
T(t) 被推出 ⇔ 向量 v(t)+v(t mod N) 落在刚性方程组的行空间（e 分量为 0）。
统计：被推出的比例；未被推出的 t 的 max(P⁺(t), P⁺(t mod N)) 分布；是否 ⌈t/N⌉ ∣ t 的那部分全部被推出（理论预测）。"""
import sys, json
from rigidity_gf2 import odd_primes_upto
def run(k):
    N = 1 << k; L = 3 * N
    primes = odd_primes_upto(N); idx = {p: i for i, p in enumerate(primes)}; E = len(primes)
    spf = list(range(L + 1))
    for i in range(2, int(L ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, L + 1, i):
                if spf[j] == j: spf[j] = i
    def vec(n):
        v = 0; big = 1
        while n > 1:
            q = spf[n]; c = 0
            while n % q == 0: n //= q; c ^= 1
            big = max(big, q)
            if q != 2 and c:
                if q >= N: return None, big
                v ^= 1 << idx[q]
        return v, big
    rows = []
    for n in range(1, N // 2 + 1, 2):
        a, _ = vec(n); b, _ = vec(N - n); rows.append(a ^ b ^ (1 << E))
    piv = {}
    for r in rows:
        while r:
            h = r.bit_length() - 1
            if h in piv: r ^= piv[h]
            else: piv[h] = r; break
    def in_span(v):
        while v:
            h = v.bit_length() - 1
            if h not in piv: return False
            v ^= piv[h]
        return True
    tot = imp = free_ok = free_tot = 0; fail_big = []
    for t in range(N + 1, L, 2):
        vt, bt = vec(t); vs, bs = vec(t % N)
        if vt is None or vs is None: continue
        tot += 1
        ok = in_span(vt ^ vs)
        imp += ok
        d = -(-t // N)
        if t % d == 0:
            free_tot += 1; free_ok += ok
        if not ok: fail_big.append(max(bt, bs) / N)
    fail_big.sort()
    return dict(k=k, N=N, t_considered=tot, implied_frac=imp / tot, free_region_all_implied=(free_ok == free_tot), free_region_size=free_tot,
                n_not_implied=len(fail_big), min_Pplus_over_N_among_not_implied=round(fail_big[0], 4) if fail_big else None,
                frac_not_implied_with_Pplus_gt_N_over_7=(sum(1 for x in fail_big if x > 1 / 7) / len(fail_big)) if fail_big else None)
if __name__ == "__main__":
    for k in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
        print(json.dumps(run(k)), flush=True)
