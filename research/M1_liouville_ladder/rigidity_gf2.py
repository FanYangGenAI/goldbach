"""M1-E2：精确刚性的 GF(2) 线性代数。
问题：哪些完全积性函数 g: N→{±1} 满足"N=2^k 的奇数子纤维刚性"
    g(n)g(N−n) = η  对所有奇数 n < N ？
记 g(p) = (−1)^{x_p}，η = (−1)^e。条件对 (x_p, e) 是 GF(2) 线性的：
    Σ_p [v_p(n) + v_p(N−n)] x_p + e = 0 (mod 2)。
计算解空间维数，并检查它是否恰好由模 8 的实特征张成（χ_{−4}, χ_8；以及 g≡1）。
λ 对应 x_p ≡ 1；若 λ 不在解空间中，即为该 k 下"λ 非刚性"的线性代数证书。"""
import sys, time, json

def odd_primes_upto(n):
    s = bytearray([1]) * (n + 1); s[0] = s[1] = 0
    for i in range(2, int(n ** .5) + 1):
        if s[i]: s[i*i::i] = bytearray(len(s[i*i::i]))
    return [p for p in range(3, n + 1) if s[p]]

def parity_vector(n, idx, spf):
    v = 0
    while n > 1:
        p = spf[n]; c = 0
        while n % p == 0: n //= p; c ^= 1
        if c and p != 2: v ^= 1 << idx[p]
    return v

def run(k):
    N = 1 << k
    primes = odd_primes_upto(N)
    idx = {p: i for i, p in enumerate(primes)}
    E = len(primes)                       # e 的位
    spf = list(range(N + 1))
    for i in range(2, int(N ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j: spf[j] = i
    rows = []
    for n in range(1, N // 2 + 1, 2):
        r = parity_vector(n, idx, spf) ^ parity_vector(N - n, idx, spf) ^ (1 << E)
        rows.append(r)
    # 高斯消元（GF(2)，Python 大整数位运算）
    pivots = {}
    for r in rows:
        while r:
            h = r.bit_length() - 1
            if h in pivots: r ^= pivots[h]
            else: pivots[h] = r; break
    nvars = E + 1
    rank = len(pivots)
    def in_rowspace_complement(x):
        # x 是解 ⇔ 对每个原始行 r，popcount(r & x) 为偶。用主元基检验即可（行空间相同）
        return all(bin(r & x).count("1") % 2 == 0 for r in pivots.values())
    # 候选解：模 8 实特征（按 p mod 8 给出 x_p），e 取使之成立的值
    def char_vec(chi_bits, e):
        x = 0
        for p in primes:
            if chi_bits[p % 8]: x |= 1 << idx[p]
        return x | (e << E)
    chis = {"χ_{-4}": {1: 0, 3: 1, 5: 0, 7: 1}, "χ_8": {1: 0, 3: 1, 5: 1, 7: 0}, "χ_{-8}": {1: 0, 3: 0, 5: 1, 7: 1}}
    found = {}
    for name, b in chis.items():
        found[name] = [e for e in (0, 1) if in_rowspace_complement(char_vec(b, e))]
    lam = sum(1 << i for i in range(E))
    lam_ok = [e for e in (0, 1) if in_rowspace_complement(lam | (e << E))]
    return dict(k=k, N=N, n_vars=nvars, n_rows=len(rows), rank=rank, solution_space_dim=nvars - rank,
                characters_mod8_solutions=found, lambda_is_solution_for_e=lam_ok)

if __name__ == "__main__":
    out = []
    for k in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
        t0 = time.time(); r = run(k); r["sec"] = round(time.time() - t0, 1)
        print(json.dumps(r, ensure_ascii=False), flush=True); out.append(r)
    json.dump(out, open("m1_e2_gf2.json", "w"), ensure_ascii=False, indent=1)
