"""Independent, read-only checks for the second external research review.

Run from the repository root with --part v2, v3, l1, or zeros. Inputs are read
from frozen Git commits; results are printed as JSON. No experiment is edited.
The zeros check recomputes Lambda and its convolution to 2**26 (CPU, several GB).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import numpy as np

R1 = "7c6290cc49fca1dcdb56e3593b876f0e650176bd"
L3D = "b46b0cbd4c0a7a7ae26451bd8161d31132d7cc18"


def read_git(ref, path):
    return subprocess.check_output(["git", "show", f"{ref}:{path}"]).decode("utf-8")


def v3():
    from scipy.optimize import least_squares

    data = json.loads(read_git(R1, "research/R1_review/out/v3_gpu.json"))
    cpu = json.loads(read_git("4b47b17", "research/L3_moments/out/e3_1_summary.json"))["rows"]
    gx, gw = np.polynomial.legendre.leggauss(16)

    def fit(rows, fixed_k=None):
        x = np.array([r["x"] for r in rows], float)
        h = np.array([r["H"] for r in rows], float)
        t = x[:, None] + (gx + 1) * h[:, None] / 2
        y = np.log([r["sigma2"] for r in rows])

        def model(p):
            lc, k = (p[0], fixed_k) if fixed_k is not None else p
            return lc + np.log((t * np.log(t) ** k * gw / 2).sum(1))

        res = least_squares(lambda p: model(p) - y, [-2] if fixed_k is not None else [-2, 2.5])
        cov = np.linalg.inv(res.jac.T @ res.jac) * (res.fun @ res.fun) / (len(rows) - len(res.x))
        return dict(k=float(fixed_k if fixed_k is not None else res.x[1]),
                    k_se=None if fixed_k is not None else float(np.sqrt(cov[1, 1])),
                    rms_log=float(np.sqrt(np.mean(res.fun**2))))

    rows = data["fixed"]
    out = {f"j{lo}_{hi}": fit([r for r in rows if lo <= r["j"] <= hi])
           for lo, hi in [(20, 32), (20, 26), (21, 26), (22, 26), (26, 32), (21, 32)]}
    out["pure_power_rms"] = {str(k): fit(rows, k) for k in [2, 2.5, 3]}
    out["first_half_leave_one_out"] = {
        str(j): fit([r for r in rows if r["j"] <= 26 and r["j"] != j])["k"]
        for j in range(20, 27)}
    out["calibration"] = []
    for r in data["strat"][:4]:
        c = next(c for c in cpu if c["j"] == r["j"])
        out["calibration"].append(dict(j=r["j"], B2_ratio=r["B2_strat"] / c["B2"],
                                       B4_ratio=r["B4_strat"] / c["B4"]))
    # Asymptotic left-rule bias if integrands are N^-1 or N^-2.
    u = 1 + np.arange(8) / 8
    out["left_rule_limit_bias"] = {
        "N^-1": float(np.mean(1 / u) / np.log(2)),
        "N^-2": float(np.mean(1 / u**2) / 0.5)}
    return out


def v2():
    rows = json.loads(read_git("b0e8c56", "research/L3c_4tuples/out/e3_3_rows.json"))["rows"]
    ratios = {}
    for row in rows:
        corrected = np.mean([(d+o)/s-n*s for n, s, d, o, _ in row["samples"]])
        ratios[str(row["j"])] = float(corrected/row["sigma2_obs"])
    sieve = np.ones(8193, bool)
    sieve[:2] = False
    for p in range(2, 91):
        if sieve[p]:
            sieve[p*p::p] = False
    theta = np.log(np.maximum(np.arange(8193), 1))*sieve
    errors = []
    for n in [1000, 2048, 4096, 6000, 8190, 8192]:
        g = theta[:n+1]*theta[n::-1]
        support = np.flatnonzero(g)
        outer = g[support, None]*g[None, support]
        off = outer[(support[:, None] != support[None, :]) &
                    (support[:, None]+support[None, :] != n)].sum()
        total = 2*(g*g).sum()-g[n//2]**2+off
        errors.append(float(abs(total-g.sum()**2)/g.sum()**2))
    return dict(corrected_ratios=ratios, identity_max_rel_err=max(errors))


def l1():
    import numba as nb

    nb.set_num_threads(min(12, nb.get_num_threads()))
    # Load just the original definitions, without its top-level experiment.
    source = read_git(R1, "research/R1_review/v1b_stratified_null.py")
    defs = source[source.index("@nb.njit"):source.index("lpf = lpf_sieve")]
    scope = {"nb": nb, "np": np}
    exec(defs.replace("cache=True", "cache=False"), scope)
    lpf = scope["lpf_sieve"](1_100_010)
    rng = np.random.default_rng(2026)
    ns = np.sort(2 * rng.choice(np.arange(500_000, 550_000), size=2000, replace=False)).astype(np.int64)
    out = {}
    for blocks in [16, 32, 64, 128, 256]:
        local = {"nb": nb, "np": np}
        exec(defs.replace("cache=True", "cache=False").replace("B = 64", f"B = {blocks}"), local)
        table = local["table"](ns, lpf).astype(float)
        m, k, r, p = [table[:, :, i] for i in range(4)]
        # M=1 is deterministic, not a zero-expectation cell.
        mean = np.divide(r * k, m, out=np.zeros_like(m), where=m > 0)
        var = np.divide(r * k * (m-k) * (m-r), m*m*(m-1), out=np.zeros_like(m), where=m > 1)
        z = (p.sum(1) - mean.sum(1)) / np.sqrt(var.sum(1))
        out[str(blocks)] = dict(mean=float(z.mean()), sd=float(z.std()), variance=float(z.var()),
                                singleton_cells=int((m == 1).sum()))
        if blocks == 64:
            out["64"]["one_per_N_class_mod30"] = {
                str(a): dict(n=int((ns % 30 == a).sum()), sd=float(z[ns % 30 == a].std()),
                             mean=float(z[ns % 30 == a].mean())) for a in range(0, 30, 2)}
            # The analytic conditional null itself can be simulated without primes.
            sim = rng.hypergeometric(k.astype(int), (m-k).astype(int), r.astype(int))
            zs = (sim.sum(1) - mean.sum(1)) / np.sqrt(var.sum(1))
            out["permutation_null"] = dict(mean=float(zs.mean()), sd=float(zs.std()))
    # A small, explicitly exploratory check of the proposed residue-race explanation.
    table_src = defs[defs.index("@nb.njit(parallel=True"):]
    for modulus in [6, 30, 210]:
        local = {"nb": nb, "np": np, "icbrt": scope["icbrt"]}
        joint = table_src.replace("cache=True", "cache=False")
        joint = joint.replace("B = 64", f"B = 64\n    QMOD = {modulus}")
        joint = joint.replace("(K, B, 4)", "(K, B * QMOD, 4)")
        joint = joint.replace("if b >= B: b = B - 1", "if b >= B: b = B - 1\n                b = b * QMOD + m % QMOD")
        exec(joint, local)
        table = local["table"](ns, lpf).astype(float)
        m, k, r, p = [table[:, :, i] for i in range(4)]
        mean = np.divide(r*k, m, out=np.zeros_like(m), where=m > 0)
        var = np.divide(r*k*(m-k)*(m-r), m*m*(m-1), out=np.zeros_like(m), where=m > 1)
        z = (p.sum(1)-mean.sum(1))/np.sqrt(var.sum(1))
        out[f"64_blocks_joint_mod{modulus}"] = dict(mean=float(z.mean()), sd=float(z.std()),
                                                   singleton_cells=int((m == 1).sum()))
    return out


def zeros():
    import hashlib
    from pathlib import Path
    from scipy.fft import rfft, irfft

    t0 = time.time()
    x = 1 << 26
    s = np.ones(x + 1, bool)
    s[:2] = False
    s[4::2] = False
    for i in range(3, int(x**0.5) + 1, 2):
        if s[i]:
            s[i*i::2*i] = False
    primes = np.flatnonzero(s)
    lam = np.zeros(x + 1)
    lam[primes] = np.log(primes)
    for p in primes[primes <= int(x**0.5)]:
        q = int(p) ** 2
        while q <= x:
            lam[q] = np.log(p)
            q *= int(p)
    psi = np.cumsum(lam)
    a = rfft(lam, 2*x, workers=4)
    a *= a
    r = irfft(a, 2*x, workers=4)[:x+1].copy()
    del a
    out = {}
    c2 = 0.6601618158468695739278121100145
    for j in [24, 26]:
        lo = (1 << j) - (1 << 21) if j == 26 else (1 << j)
        n = np.arange(lo, lo + (1 << 21), 2)
        sg = np.full(len(n), 2*c2)
        for p in primes[1:]:
            if p > n[-1]:
                break
            st = (-lo) % p
            if st % 2:
                st += p
            sg[st//2::p] *= (p-1)/(p-2)
        f = r[n] - n*sg
        q = 2*(psi[n]-n)
        old = (q-q.mean())*sg
        full = q*sg
        # The raw q=1 linear term, with no singular-series modulation.
        raw = 2*(psi[n-1]-(n-1))

        def stats(v):
            cov = float(np.mean((f-f.mean())*(v-v.mean())))
            return dict(var_ratio=float(v.var()/f.var()), corr=float(np.corrcoef(f, v)[0, 1]),
                        ols_r2=float(np.corrcoef(f, v)[0, 1]**2), slope=float(cov/v.var()),
                        fixed_coefficient_variance_reduction=float(1-(f-v).var()/f.var()),
                        covariance_over_varF=float(cov/f.var()))

        out[str(j)] = dict(var_F=float(f.var()), mean_q=float(q.mean()), sd_q=float(q.std()),
                           original=stats(old), product_then_center=stats(full), raw_linear_term=stats(raw))
    # Verify the actual smoothed linear term rather than the midpoint proxy.
    cr = np.cumsum(r)
    cached_zeros = Path("research/L3d_zeros/out/zeta_zeros_20000.npy")
    if cached_zeros.exists():
        gamma = np.load(cached_zeros)
        rho = 0.5+1j*gamma
        xs = np.unique(np.round(np.exp(np.linspace(np.log(2**14), np.log(x-1), 400))).astype(np.int64))
        zero_sum = np.zeros(len(xs))
        for offset in range(0, len(xs), 50):
            xx = xs[offset:offset+50, None]
            zero_sum[offset:offset+50] = -4*np.real(np.exp((rho+1)*np.log(xx))/(rho*(rho+1))).sum(1)
        g = cr[xs]-xs.astype(float)**2/2
        out["P1_cached_zeros"] = dict(count=len(gamma), gamma_last=float(gamma[-1]),
            cache_sha256=hashlib.sha256(cached_zeros.read_bytes()).hexdigest(),
            corr=float(np.corrcoef(g, zero_sum)[0, 1]),
            rms_ratio=float(np.linalg.norm(g-zero_sum)/np.linalg.norm(g)))
    cs = np.cumsum(psi)
    hw = 1 << 14
    starts = np.unique(np.round(np.exp(np.linspace(np.log(2**20), np.log(x-hw-1), 2000))).astype(np.int64))
    xb = starts + (hw+1)/2
    yr = (cr[starts+hw]-cr[starts])/hw-xb
    midpoint = 2*(psi[np.round(xb).astype(np.int64)]-xb)
    # sum_{n=s+1}^{s+H} psi(n-1) = sum_{t=s}^{s+H-1} psi(t)
    average = 2*((cs[starts+hw-1]-cs[starts-1])/hw-(xb-1))-1
    out["P2"] = {}
    for name, predictor in [("midpoint", midpoint), ("window_average", average)]:
        slope, intercept = np.polyfit(predictor, yr, 1)
        out["P2"][name] = dict(slope=float(slope), intercept=float(intercept),
                               r2=float(np.corrcoef(predictor, yr)[0, 1]**2))
    out["elapsed_s"] = time.time()-t0
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["v2", "v3", "l1", "zeros"], required=True)
    args = parser.parse_args()
    print(json.dumps(globals()[args.part](), ensure_ascii=False, indent=2))
