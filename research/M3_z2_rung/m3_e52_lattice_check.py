"""核对 ℓ = 2 的格约化恒等式（RESULTS 第 69 节）：对相邻互素序列、q | a_{t-4}，每个合法串满足
  (i)  u·K + c_t ≡ −ε_{t-1} + ε_{t-2}·a_t (mod a_{t-1})，u = a_{t-2}a_t，K = K_{t-2} ∈ [0, a_{t-1}]；
  (ii) M ≡ u·(K − ε_{t-4}a_{t-1}) + c_t − ε_{t-2}a_t (mod q)。
K_{t-2} 由 a_{t-1}c_{t-2} + c_{t-1} = K a_{t-2} − ε_{t-2} 给出。"""
import random, math, json
from m3_e49_coprime import gen_seq, strings, prime_factors
rng = random.Random(69); bad1 = bad2 = tot = 0; seqs = 0
while seqs < 150:
    t = rng.randint(4, 8); a = gen_seq(rng, t); j = t - 4
    qs = [q for q in prime_factors(a[j]) if all(a[i] % q for i in range(j + 1, t + 1))]
    if not qs: continue
    seqs += 1
    for q in qs:
        u = a[t - 2] * a[t]
        for M, eps, c in strings(a, True):
            num = a[t - 1] * c[t - 2] + c[t - 1] + eps[t - 2]
            assert num % a[t - 2] == 0
            K = num // a[t - 2]; assert 0 <= K <= a[t - 1]
            tot += 1
            if (u * K + c[t] - (-eps[t - 1] + eps[t - 2] * a[t])) % a[t - 1] != 0: bad1 += 1
            if (M - (u * (K - eps[j] * a[t - 1]) + c[t] - eps[t - 2] * a[t])) % q != 0: bad2 += 1
print(json.dumps(dict(sequences=seqs, strings_checked=tot, viol_i=bad1, viol_ii=bad2)))
