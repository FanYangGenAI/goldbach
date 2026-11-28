"""M3-E5（预注册，DESIGN 第 4 节）：短证书占比与行的多重度。"""
import sys, json, itertools, time
from collections import defaultdict, Counter
from m3_e2_deletion import setup, span_test
from m3_e4_cert_size import refl_cov
def run(k, s, q=3):
    N, primes, rows, targets = setup(k, q)
    deriv = span_test(rows, targets)
    R = [r for _, r in rows]
    half = {}
    for t in range(0, s + 1):
        for comb in itertools.combinations(range(len(R)), t):
            x = 0
            for i in comb: x ^= R[i]
            if x not in half: half[x] = comb
    use = Counter(); sizes = Counter(); nd = 0
    for n, v in targets:
        if n not in deriv or refl_cov(N, q, n, 6 * q): continue
        nd += 1; best = None
        for x, comb in half.items():
            y = v ^ x
            if y in half:
                S = set(comb) ^ set(half[y])
                if best is None or len(S) < len(best): best = S
        if best is None: sizes["long"] += 1
        else:
            sizes[len(best)] += 1
            for i in best: use[i] += 1
    short = sum(c for kk, c in sizes.items() if kk != "long")
    return dict(k=k, N=N, s=s, derivable_nonreflection=nd, frac_short=round(short / max(1, nd), 4),
                size_dist={str(a): b for a, b in sorted(sizes.items(), key=lambda x: str(x[0]))},
                row_use_max=max(use.values()) if use else 0, row_use_mean_over_used=round(sum(use.values()) / max(1, len(use)), 3))
if __name__ == "__main__":
    for k, s in ((9, 3), (10, 3), (11, 2), (12, 2)):
        t0 = time.time(); r = run(k, s); r["sec"] = round(time.time() - t0, 1); print(json.dumps(r), flush=True)
