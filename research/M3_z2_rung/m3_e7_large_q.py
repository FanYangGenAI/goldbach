"""M3-E7（预注册，DESIGN 第 6 节）：较大的 q，以及素数 N 的完整纤维。"""
import sys, json, random, time
sys.path.insert(0, "../M1_liouville_ladder")
from m3_e2_deletion import span_test
def primes_upto(n):
    s = bytearray([1]) * (n + 1); s[0] = s[1] = 0
    for i in range(2, int(n ** .5) + 1):
        if s[i]: s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(n + 1) if s[i]]
def build(N, q, full):
    L = q * N + 1
    P = primes_upto(N - 1) if full else [p for p in primes_upto(N - 1) if p > 2]
    idx = {p: i for i, p in enumerate(P)}; E = len(P)
    spf = list(range(L + 1))
    for i in range(2, int(L ** .5) + 1):
        if spf[i] == i:
            for j in range(i * i, L + 1, i):
                if spf[j] == j: spf[j] = i
    def vec(n):
        v = 0
        while n > 1:
            p = spf[n]; c = 0
            while n % p == 0: n //= p; c ^= 1
            if c and (full or p != 2):
                if p >= N: return None
                v ^= 1 << idx[p]
        return v
    if full:
        rows = [(n, vec(n) ^ vec(N - n) ^ (1 << E)) for n in range(1, (N + 1) // 2)]
        ns = range(1, N)
    else:
        rows = [(n, vec(n) ^ vec(N - n) ^ (1 << E)) for n in range(1, N // 2 + 1, 2)]
        ns = range(1, N, 2)
    targets = []
    for n in ns:
        a, b = vec(q * n), vec((q * n) % N)
        if a is not None and b is not None: targets.append((n, a ^ b))
    return rows, targets
def measure(rows, targets, rng):
    base = span_test(rows, targets)
    rk = []
    for _ in range(10):
        B = set(rng.sample(range(len(rows)), 16))
        rk.append(len(base - span_test([r for i, r in enumerate(rows) if i not in B], targets)) / 16)
    worst = 0
    for i in range(len(rows)):
        worst = max(worst, len(base - span_test(rows[:i] + rows[i + 1:], targets)))
    return dict(coverage=round(len(base) / len(targets), 4), random16_ratio=round(sum(rk) / len(rk), 3), worst_single=worst)
if __name__ == "__main__":
    rng = random.Random(19000)
    for (N, q, full) in ((4096, 31, False), (4096, 67, False), (4093, 3, True), (4093, 31, True), (4093, 67, True)):
        t0 = time.time(); rows, targets = build(N, q, full)
        r = dict(N=N, q=q, fibre="full" if full else "odd", n_targets=len(targets), **measure(rows, targets, rng), sec=round(time.time() - t0, 1))
        print(json.dumps(r), flush=True)
