"""M1-E2 续：求 2^k 奇数子纤维刚性解空间的一组基，看"异类"解落在哪些素数上，
以及解空间限制到小素数（p < B）后是否只剩模 8 特征。"""
import sys, json
from rigidity_gf2 import odd_primes_upto, parity_vector
def nullspace(rows, nv):
    piv = {}  # pivot col -> row (fully reduced)
    for r in rows:
        for c, pr in piv.items():
            if r >> c & 1: r ^= pr
        if r:
            c = r.bit_length() - 1
            for c2 in list(piv):
                if piv[c2] >> c & 1: piv[c2] ^= r
            piv[c] = r
    free = [c for c in range(nv) if c not in piv]
    basis = []
    for f in free:
        x = 1 << f
        for c, pr in piv.items():
            if pr >> f & 1: x |= 1 << c
        basis.append(x)
    return basis
def run(k, B=200):
    N = 1 << k
    primes = odd_primes_upto(N); idx = {p: i for i, p in enumerate(primes)}; E = len(primes)
    spf = list(range(N + 1))
    for i in range(2, int(N ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j: spf[j] = i
    rows = [parity_vector(n, idx, spf) ^ parity_vector(N - n, idx, spf) ^ (1 << E) for n in range(1, N // 2 + 1, 2)]
    basis = nullspace(rows, E + 1)
    # 每个基向量的支撑中最小的素数
    def support_primes(x): return [primes[i] for i in range(E) if x >> i & 1]
    small = [p for p in primes if p < B]
    restricted = set()
    for x in basis:
        restricted.add(tuple((x >> idx[p]) & 1 for p in small) + ((x >> E) & 1,))
    # 生成限制后子空间（GF(2) 张成）
    span = {tuple([0] * (len(small) + 1))}
    for v in restricted:
        span |= {tuple(a ^ b for a, b in zip(v, s)) for s in span}
    chars = {"χ_{-4}": {1: 0, 3: 1, 5: 0, 7: 1}, "χ_8": {1: 0, 3: 1, 5: 1, 7: 0}}
    char_span = {tuple([0] * len(small))}
    for b in chars.values():
        v = tuple(b[p % 8] for p in small)
        char_span |= {tuple(a ^ c for a, c in zip(v, s)) for s in char_span}
    restricted_primes_part = {s[:-1] for s in span}
    return dict(k=k, dim=len(basis), restricted_to_primes_below=B, restricted_dim=len(span).bit_length() - 1,
                restricted_equals_char_span=restricted_primes_part <= char_span,
                min_support_prime_of_exotic=sorted({min(support_primes(x)) for x in basis if support_primes(x)})[:10])
if __name__ == "__main__":
    for k in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
        print(json.dumps(run(k), ensure_ascii=False), flush=True)
