"""M3-E6（预注册，DESIGN 第 5 节）：一起删去枢纽行后失去的 n。"""
import sys, json, itertools, time
from collections import Counter
from m3_e2_deletion import setup, span_test
from m3_e4_cert_size import refl_cov
def run(k, s=2, q=3):
    N, primes, rows, targets = setup(k, q)
    deriv = span_test(rows, targets)
    R = [r for _, r in rows]; half = {}
    for t in range(0, s + 1):
        for comb in itertools.combinations(range(len(R)), t):
            x = 0
            for i in comb: x ^= R[i]
            if x not in half: half[x] = comb
    use = Counter()
    for n, v in targets:
        if n not in deriv or refl_cov(N, q, n, 6 * q): continue
        best = None
        for x, comb in half.items():
            y = v ^ x
            if y in half:
                S = set(comb) ^ set(half[y])
                if best is None or len(S) < len(best): best = S
        if best:
            for i in best: use[i] += 1
    order = [i for i, _ in use.most_common()]
    out = dict(k=k, N=N, top_rows=[rows[i][0] for i in order[:8]])
    for m in (5, 20, 50):
        B = set(order[:m])
        after = span_test([rw for i, rw in enumerate(rows) if i not in B], targets)
        kd = len(deriv - after); out[f"m{m}_killed"] = kd; out[f"m{m}_ratio"] = round(kd / m, 3)
    return out
if __name__ == "__main__":
    for k in (11, 12):
        t0 = time.time(); r = run(k); r["sec"] = round(time.time() - t0, 1); print(json.dumps(r), flush=True)
