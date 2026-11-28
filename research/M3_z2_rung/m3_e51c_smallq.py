"""M3-E51c（探索性，非预注册）：小素数 q ∈ {3, 5, 7}、尾数长度 ℓ ≥ 2 时，相对超出 (P − 2/q)/(2/q) 是否随 a_t 增长
（绝对值估计预言的损失为 log a_t 量级）。随机相邻互素序列，松弛″与纯两种测度都算。"""
import random, json, math
from m3_e49_coprime import gen_seq, strings, prime_factors
rng = random.Random(1357); bins = {}
for it in range(1500):
    t = rng.randint(4, 9); a = gen_seq(rng, t)
    for relaxed in (False, True):
        S = strings(a, relaxed)
        if not S: continue
        R = len(S)
        for q in (3, 5, 7):
            js = [k for k in range(t + 1) if a[k] % q == 0]
            if not js: continue
            j = max(js)
            if t - j - 2 < 2: continue
            P = sum(1 for M, _, _ in S if M % q == 0 or (M + 1) % q == 0) / R
            rel = (P - 2 / q) / (2 / q)
            key = (relaxed, 'a_t<16' if a[t] < 16 else ('16<=a_t<64' if a[t] < 64 else 'a_t>=64'))
            b = bins.setdefault(key, [0, 0.0, 0.0]); b[0] += 1; b[1] += rel; b[2] = max(b[2], abs(rel))
print(json.dumps({('relaxed2' if k[0] else 'pure') + ':' + k[1]: dict(n=v[0], mean_rel_excess=round(v[1] / v[0], 4), max_abs_rel=round(v[2], 3)) for k, v in sorted(bins.items())}, ensure_ascii=False))
