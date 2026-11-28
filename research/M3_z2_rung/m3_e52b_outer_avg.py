"""M3-E52b（探索性，非预注册）：ℓ = 3、4 时，只对最外层尾数乘数 a_{j+2} 平均（其余乘数固定）是否足以把偏差压到 O(1/a_t)。
对每个族报告 mean_b |P − 2/q|、mean_b (P − 2/q)·a_t、以及单个 b 的最大偏差。"""
import math, json, random
from m3_e49_coprime import strings, is_prime, prime_factors
def fam(a_before, a_after, q, j):
    # a_before = [a_0..a_{j+1}]，a_after = [a_{j+3}..a_t]；b = a_{j+2} ∈ (a_{j+3}, a_{j+1})
    lo, hi = a_after[0], a_before[-1]; vals = []; mx = 0
    for b in range(lo + 1, hi):
        if b % q == 0 or math.gcd(b, hi) != 1 or math.gcd(b, lo) != 1: continue
        a = a_before + [b] + a_after
        S = strings(a, True)
        if not S: continue
        P = sum(1 for M, _, _ in S if M % q == 0 or (M + 1) % q == 0) / len(S)
        vals.append(P - 2 / q); mx = max(mx, abs(P - 2 / q))
    if len(vals) < 5: return None
    at = a_after[-1]
    return dict(seq=a_before + ['b'] + a_after, q=q, n=len(vals), mean_abs=round(sum(map(abs, vals)) / len(vals), 4),
                mean_excess_x_at=round(sum(vals) / len(vals) * at, 3), max_abs=round(mx, 3), inv_at=round(1 / at, 4))
def main():
  rng = random.Random(77); out = []
  while len(out) < 14:
      ell = 3 if len(out) < 8 else 4
      # 构造 a_after（长度 ell）：相邻互素、下降
      a_after = [rng.randint(20, 60)]
      ok = True
      for _ in range(ell - 1):
          c = [x for x in range(max(4, a_after[-1] // 3), a_after[-1]) if math.gcd(x, a_after[-1]) == 1]
          if not c: ok = False; break
          a_after.append(rng.choice(c))
      if not ok: continue
      a_j1 = rng.randint(a_after[0] + 30, a_after[0] + 150)
      if math.gcd(a_j1, a_after[0]) == 1: pass
      qs = [x for x in range(3, 120) if is_prime(x) and a_j1 % x and all(y % x for y in a_after)]
      q = rng.choice(qs); m = rng.randint(a_j1 // q + 1, (a_j1 + 300) // q); aj = q * m
      if math.gcd(aj, a_j1) != 1: continue
      p = next(x for x in range(aj + 1, 5 * aj) if is_prime(x))
      r = fam([p, aj, a_j1], a_after, q, 1)
      if r: out.append(r)
  for r in out: print(json.dumps(r))

if __name__ == "__main__":
    main()
