"""M3-E3（预注册，DESIGN 第 3 节）：最坏的单行删除。"""
import sys, json, time
from collections import Counter
from m3_e2_deletion import setup, span_test
def factor(m):
    f = []; d = 2
    while d * d <= m:
        while m % d == 0: f.append(d); m //= d
        d += 1
    if m > 1: f.append(m)
    return "·".join(map(str, f))
def run(k, q=3):
    N, primes, rows, targets = setup(k, q)
    base = span_test(rows, targets)
    ks = []
    for i in range(len(rows)):
        after = span_test(rows[:i] + rows[i + 1:], targets)
        ks.append((len(base - after), rows[i][0]))
    ks.sort(reverse=True)
    c = Counter(x for x, _ in ks)
    worst = ks[:3]
    return dict(k=k, N=N, rows=len(rows), max_killed=ks[0][0], mean_killed=round(sum(x for x, _ in ks) / len(ks), 3),
                frac_zero=round(c[0] / len(ks), 3), dist=dict(sorted(c.items())),
                worst_rows=[dict(n=n, killed=x, n_fac=factor(n), Nn_fac=factor(N - n)) for x, n in worst])
if __name__ == "__main__":
    for k in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
        t0 = time.time(); r = run(k); r["sec"] = round(time.time() - t0, 1); print(json.dumps(r), flush=True)
