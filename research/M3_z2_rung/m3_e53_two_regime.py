"""M3-E53（预注册：DESIGN.md 第 37 节）：按素数大小分两段的一般 ℓ 尾数整除界。
H1：M ≡ X_j (mod a_j)，X_j = M^{(j+2)} − ε_j T_{j+1}，且替换 (a_j, a_{j+1}) 后 X_j 不变。
H2：q | M  ⟺  b(K_{j+2} − ε_j a_{j+3}) ≡ w (mod Q)，Q = q a_{j+3}。
H3：引理 W 的显式不等式（对 b = a_{j+2} 求和，每个 ε 单独）。
H4：引理 H 的不等式（对 a_j 求和，权重 1/a_j）。"""
import random, json, math, sys
from sympy import factorint, divisor_count, divisors, primerange
from m3_e49_coprime import gen_seq, strings, prime_factors, is_prime

def prod(xs):
    r = 1
    for x in xs: r *= x
    return r

def tails(a):
    t = len(a) - 1
    T = [prod(a[k + 1:]) for k in range(t + 1)]
    Mt = lambda c, i: sum(c[k] * T[k] for k in range(i, t + 1)) if i <= t else 0
    return t, T, Mt

def Xj(a, c, eps, j):
    t, T, Mt = tails(a)
    return Mt(c, j + 2) - eps[j] * T[j + 1]

def coprime_alt(rng, lo, hi, avoid):
    cands = [n for n in range(lo, hi) if all(math.gcd(n, v) == 1 for v in avoid)]
    return rng.choice(cands) if cands else None

def run_H1(rng, nseq=60):
    tot = v_mod = v_ind = 0
    for _ in range(nseq):
        t = rng.randint(4, 7); a = gen_seq(rng, t)
        S = strings(a, False)
        for j in range(0, t - 1):
            for (M, eps, c) in S:
                tot += 1
                if (M - Xj(a, c, eps, j)) % a[j]: v_mod += 1
            # 替换 (a_j, a_{j+1})
            a1 = coprime_alt(rng, a[j + 2] + 1, 4 * a[j + 2] + 5, [a[j + 2]])
            if a1 is None: continue
            avoid = [a1] + ([a[j - 1]] if j >= 1 else [])
            a0 = coprime_alt(rng, a1 + 1, 4 * a1 + 5, avoid)
            if a0 is None: continue
            b = list(a); b[j + 1] = a1; b[j] = a0
            S2 = strings(b, False)
            for (M, eps, c), (M2, eps2, c2) in zip(S, S2):
                assert eps == eps2 and c[t] == c2[t]
                if Xj(a, c, eps, j) != Xj(b, c2, eps2, j): v_ind += 1
    return dict(strings_x_j=tot, viol_mod=v_mod, viol_independence=v_ind)

def w_of(a, c, eps, j, q):
    t, T, Mt = tails(a)
    m = a[j + 3]; S = prod(a[j + 4:]); I = -eps[j + 2] * S + Mt(c, j + 4)
    Q = q * m
    r1 = (c[j + 3] + eps[j + 2]) % m
    r2 = (-I * pow(S, -1, q)) % q
    # CRT
    w = (r1 + m * (((r2 - r1) * pow(m, -1, q)) % q)) % Q
    return w, Q

def run_H2(rng, nseq=150):
    tot = viol = 0; cases = 0
    while cases < nseq:
        t = rng.randint(4, 8); a = gen_seq(rng, t)
        for j in range(0, t - 2):
            qs = [q for q in prime_factors(a[j]) if all(a[i] % q for i in range(j + 1, t + 1))]
            if not qs: continue
            cases += 1
            S = strings(a, True)
            for q in qs:
                m = a[j + 3]; b = a[j + 2]
                for (M, eps, c) in S:
                    K = (a[j + 3] * c[j + 2] + c[j + 3] + eps[j + 2]); assert K % b == 0; K //= b
                    w, Q = w_of(a, c, eps, j, q)
                    tot += 1
                    if (M % q == 0) != ((b * (K - eps[j] * m) - w) % Q == 0): viol += 1
    return dict(sequences_j=cases, checks=tot, viol=viol)

def sigma_mhalf(n):
    return sum(d ** -0.5 for d in divisors(n))

def inner_valid(a, c, j):
    t = len(a) - 1
    return all(1 <= c[i] <= a[i] - 2 for i in range(j + 3, t + 1))

def run_H3(rng, nfam=24):
    out = []; viol = 0; worst = 0.0
    fams = 0; tries = 0
    while fams < nfam and tries < 5000:
        tries += 1
        t = rng.randint(4, 6); ell = rng.randint(1, min(3, t - 2)); j = t - 2 - ell
        a = gen_seq(rng, t)
        if a[t] < 2 * t + 6: continue
        qs = [q for q in prime_factors(a[j]) if all(a[i] % q for i in range(j + 1, t + 1))]
        if not qs: continue
        q = min(qs) if rng.random() < 0.6 else max(qs)
        m = a[j + 3]
        Istar = [b for b in range(m + 1, a[j + 1]) if math.gcd(b, a[j + 1] * m * q) == 1]
        if len(Istar) < 8: continue
        L = a[j + 1] - m - 1; Q = q * m
        fams += 1
        # 内层数据（与 b 无关）：用 base 序列的全体串
        base = strings(a, False)
        per_eps_Y = {}
        for (M, eps, c) in base:
            if not inner_valid(a, c, j): continue
            w, _ = w_of(a, c, eps, j, q)
            per_eps_Y.setdefault(tuple(eps), []).append(math.gcd(w, Q))
        tQ = int(divisor_count(Q)); sQ = sigma_mhalf(Q)
        errc = 2 * math.ceil(L / Q) * tQ * sQ * (2 + math.log(Q)) ** 2 * math.sqrt(Q)
        lhs = {}; Pb = []
        for b in Istar:
            ab = list(a); ab[j + 2] = b
            Sb = strings(ab, True)
            nq = sum(1 for (M, _, _) in Sb if M % q == 0)
            Pb.append(nq / len(Sb))
            for (M, eps, c) in Sb:
                if M % q == 0: lhs[tuple(eps)] = lhs.get(tuple(eps), 0) + 1
        fam_lhs = fam_main = fam_err = 0.0; ratio_max = -1e9
        for e, gl in per_eps_Y.items():
            main = len(gl) * len(Istar) * (m + 1) / Q
            err = errc * sum(math.sqrt(g) for g in gl)
            l = lhs.get(e, 0)
            if l > main + err: viol += 1
            ratio_max = max(ratio_max, (l - main) / err)
            fam_lhs += l; fam_main += main; fam_err += err
        worst = max(worst, ratio_max)
        EP = sum(Pb) / len(Pb)
        out.append(dict(a=a, j=j, ell=ell, q=q, small_q=q * q <= a[j + 1], nb=len(Istar), L=L, Q=Q,
                        EP=round(EP, 5), inv_q=round(1 / q, 5), dev=round(EP - 1 / q, 5),
                        lhs=int(fam_lhs), main=round(fam_main, 1), err=round(fam_err, 1),
                        excess_over_err_max=round(ratio_max, 4)))
    return dict(families=fams, viol=viol, max_excess_over_err=round(worst, 4), detail=out)

def run_H4(rng, nseq=80):
    tot = viol = vomega = 0; norm = []
    for _ in range(nseq):
        t = rng.randint(4, 8); a = gen_seq(rng, t)
        S = strings(a, True)
        if not S: continue
        for j in range(1, t - 1):
            ell = t - 2 - j
            lo, hi = a[j + 1], a[j - 1]
            Y = math.sqrt(a[j + 1]); Lj = prod(a[j + 1:])
            ns = [n for n in range(lo + 1, hi) if math.gcd(n, a[j - 1] * a[j + 1]) == 1]
            if not ns: continue
            W = sum(1 / n for n in ns)
            for (M, eps, c) in rng.sample(S, min(15, len(S))):
                X = Xj(a, c, eps, j)
                lhs = 0.0
                for n in ns:
                    k = sum(1 for qq in prime_factors(n) if qq > Y and Lj % qq and (X * (X + 1)) % qq == 0)
                    lhs += k / n
                big = [qq for qq in primerange(int(Y) + 1, hi) if (X * (X + 1)) % qq == 0]
                rhs = sum(math.log(hi / lo) / qq + 1 / lo for qq in big)
                tot += 1
                if lhs > rhs + 1e-12: viol += 1
                fac = factorint(abs(X * (X + 1)))
                om = sum(1 for qq in fac if qq > Y)
                if om > 2 * (2 * ell + 3): vomega += 1
                norm.append((lhs / W) * math.sqrt(a[j + 1]) / (ell + 1))
    norm.sort()
    return dict(checks=tot, viol=viol, viol_omega=vomega,
                normalized_mean=round(sum(norm) / len(norm), 4),
                normalized_q90=round(norm[int(0.9 * len(norm))], 4), normalized_max=round(norm[-1], 4),
                note="normalized = (E_{a_j}[#hard q | X(X+1)]) * sqrt(a_{j+1})/(ell+1)")

if __name__ == '__main__':
    rng = random.Random(20261128)
    which = sys.argv[1:] or ['H1', 'H2', 'H3', 'H4']
    res = {}
    if 'H1' in which: res['H1'] = run_H1(rng)
    if 'H2' in which: res['H2'] = run_H2(rng)
    if 'H4' in which: res['H4'] = run_H4(rng)
    if 'H3' in which: res['H3'] = run_H3(rng)
    print(json.dumps(res, indent=1))
