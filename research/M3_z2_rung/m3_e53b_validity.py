"""探索性核对（非预注册）：松弛″的合法性是否只由 c_t 决定？
命题：若 a_{i+1} < a_i，则 c_{i+1} ∈ [1, a_{i+1} − 2] 推出 c_i ∈ [1, a_i − 2]。
因此每个 ε 恰有 a_t − 2 个合法串（c_t ∈ [1, a_t − 2]）。"""
import random, json
from m3_e49_coprime import gen_seq, strings
rng = random.Random(53); bad = 0; n = 0; tot = 0
for _ in range(300):
    t = rng.randint(2, 9); a = gen_seq(rng, t); n += 1
    S = strings(a, True); tot += len(S)
    if len(S) != 2 ** t * (a[t] - 2): bad += 1
    # 逐串核对：c_t 合法 ⇒ 全部合法
    allS = strings(a, False)
    for (M, eps, c) in allS:
        if 1 <= c[t] <= a[t] - 2 and not all(1 <= c[i] <= a[i] - 2 for i in range(t + 1)): bad += 1
print(json.dumps(dict(sequences=n, valid_strings=tot, violations=bad)))
