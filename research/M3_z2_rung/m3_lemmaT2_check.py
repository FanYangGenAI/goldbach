"""引理 T''_2 的精确核对：对随机参数，直接计算 (1/m)Σ_A #{c<X : β((Ac+B) mod m)+c ≡ x (mod q)}，与 X/q + τ(m) + σ(m)/m 比较。"""
import random, math, json
def tau(n): return sum(1 for d in range(1, n + 1) if n % d == 0)
def sigma(n): return sum(d for d in range(1, n + 1) if n % d == 0)
rng = random.Random(9); viol = 0; worst = 0.0; T = 0
for _ in range(300):
    m = rng.randint(2, 120); q = rng.randint(2, 60)
    if math.gcd(m, q) != 1: continue
    beta = rng.choice([b for b in range(1, q) if math.gcd(b, q) == 1] or [1]); B = rng.randrange(m); x = rng.randrange(q)
    X = rng.randint(1, m)
    tot = sum(1 for A in range(m) for c in range(X) if (beta * ((A * c + B) % m) + c - x) % q == 0)
    lhs = tot / m; rhs = X / q + tau(m) + sigma(m) / m; T += 1
    viol += lhs > rhs + 1e-12; worst = max(worst, lhs / rhs)
print(json.dumps(dict(tested=T, violations=viol, max_lhs_over_rhs=round(worst, 3))))
