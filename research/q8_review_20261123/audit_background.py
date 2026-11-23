"""Read-only Q5 audit: saved integer polynomials, coefficient contract, numerics.

Run from the repository root: python research/q8_review_20261123/audit_background.py
No Q5 top-level program is imported; no billion-element sieve is run.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy import integrate, special

ROOT = Path(__file__).resolve().parents[2]
REF = "f9b92144766cd40811e685bb319cdc227fa10b69"
SOURCE = "research/Q5_background/out/q5d_summary.json"
A = 0.45
ZS = [0.25, 0.5, -0.25, -0.5, 0.5j]


def git_bytes(path):
    return subprocess.check_output(["git", "show", f"{REF}:{path}"], cwd=ROOT)


def primes_upto(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, math.isqrt(n) + 1):
        if sieve[p]:
            sieve[p * p :: p] = False
    return np.flatnonzero(sieve)


PR = primes_upto(1_000_000).astype(float)
ODD = PR[1:]


def factors(n):
    out = {}
    p = 2
    while p * p <= n:
        while n % p == 0:
            out[p] = out.get(p, 0) + 1
            n //= p
        p += 1
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def j_and_derivative(p, e, z):
    j = (p - 2 + z) / (p - z)
    dj = 2 * (p - 1) / (p - z) ** 2
    for _ in range(e):
        dj, j = (2 * z * j + z * z * dj) / p, 1 - 1 / p + z * z * j / p
    return j, dj


def local_c(p, e, z):
    j, dj = j_and_derivative(p, e, z)
    return z * dj / j - 2 * z / (p - z)


def weighted_derivative(n, z):
    # C_p = (d/du) log r_p(zu,zu) at u=1; use algebraic derivatives.
    c0 = -2 * z * (z - 1) / ((ODD - 1) ** 2 - (z - 1) ** 2)
    total = np.sum(np.log(ODD) * c0)
    for p, e in factors(n).items():
        if 2 < p <= 1_000_000:
            total -= math.log(p) * (-2 * z * (z - 1) / ((p - 1) ** 2 - (z - 1) ** 2))
        total += math.log(p) * local_c(p, e, z)
    return total


def finite_difference_sum(n, z):
    # Direct replication of the submitted finite-difference prescription.
    def bracket(p, e):
        h = 1e-6
        def logs(u):
            zz = z * u
            j = np.asarray(j_and_derivative(p, e, zz)[0], dtype=complex)
            zp = np.asarray((p - 1) / (p - zz), dtype=complex)
            return np.log(j) - 2 * np.log(zp)
        return (logs(1 + h) - logs(1 - h)) / (2 * h)
    total = np.sum(np.log(PR) * bracket(PR, 0))
    for p, e in factors(n).items():
        if p <= 1_000_000:
            total -= math.log(p) * bracket(float(p), 0)
        total += math.log(p) * bracket(float(p), e)
    return total


def r_local(n, z):
    lr = np.log(1 - (z - 1) ** 2 / (ODD - 1) ** 2 + 0j).sum()
    for p, e in factors(n).items():
        if 2 < p <= 1_000_000:
            lr -= np.log(1 - (z - 1) ** 2 / (p - 1) ** 2 + 0j)
        j = j_and_derivative(p, e, z)[0]
        lr += np.log(j / ((p - 1) / (p - z)) ** 2 + 0j)
    return np.exp(lr)


def density_grid(z, order=320, prime_limit=200_000, rule="jacobi"):
    if rule == "jacobi":
        x, weights = special.roots_jacobi(order, 0, -complex(z).real)
        u = (x + 1) / 2
        y = A * u
    else:
        x, weights = np.polynomial.legendre.leggauss(order)
        u = (x + 1) / 2
        power = 4 if rule == "legendre4" else 2
        y = A * u**power
    sigma = 1 - y
    logb = np.zeros(order, dtype=complex)
    pp = PR[PR <= prime_limit]
    for start in range(0, len(pp), 1024):
        ps = pp[None, start : start + 1024] ** (-sigma[:, None])
        logb += (z * np.log1p(-ps) - np.log1p(-z * ps)).sum(axis=1)
    logb += (z * z - z) / 2 * special.exp1((2 * sigma - 1) * math.log(prime_limit))
    logzeta = np.empty(order)
    near = y < 1e-8
    logzeta[near] = -np.log(y[near]) - np.euler_gamma * y[near]
    logzeta[~near] = np.log(np.abs(special.zeta(sigma[~near])))
    phi = np.exp(z * logzeta + logb)
    if rule == "jacobi":
        # Integrate the endpoint singularity y^{-Re z} in the rule itself.
        g = y ** complex(z) * phi
        c = (np.sin(np.pi * z) / np.pi * A ** (1 - complex(z))
             * weights * 2 ** (complex(z).real - 1)
             * u ** (-1j * complex(z).imag) * g)
    else:
        c = np.sin(np.pi * z) / np.pi * weights / 2 * A * power * u**(power-1) * phi
    return y, c


def convolution(n, y, c, exact_endpoints=True):
    a, b = y[:, None], y[None, :]
    beta = np.exp(special.betaln(1 - a, 1 - b))
    kernel = n ** (1 - a - b) * beta
    if exact_endpoints:
        eps = 2 / n
        kernel *= 1 - special.betainc(1 - a, 1 - b, eps) - special.betainc(1 - b, 1 - a, eps)
    else:
        kernel -= 2 ** (1 - a) * n ** (-b) / (1 - a)
        kernel -= 2 ** (1 - b) * n ** (-a) / (1 - b)
    value = c @ kernel @ c
    first_mellin = -(y * c) @ kernel @ c
    return value, first_mellin


def coefficient_integrals(n):
    # Scale to [2/N, 1/2]; independent integration from the submitted t integral.
    def f(u, cut):
        t = n * u
        v = n - t
        value = 1 / (math.log(t) * math.log(v))
        if cut:
            value *= (1 - t ** (-A)) * (1 - v ** (-A))
        return value
    vals = []
    for cut in (False, True):
        points = sorted({2/n, 1e-6, 1e-4, 0.01, 0.5})
        v, err = 0.0, 0.0
        for left,right in zip(points,points[1:]):
            part, part_err = integrate.quad(f,left,right,args=(cut,),epsabs=1e-14,epsrel=2e-11,limit=300)
            v += part
            err += part_err
        vals.append((2 * n * v, 2 * n * err))
    return vals


def stats(values):
    v = np.asarray(values, float)
    sem = v.std(ddof=1) / math.sqrt(len(v))
    return {"n": len(v), "mean": float(v.mean()), "sem_ddof1": float(sem),
            "three_sem_pass": bool(abs(v.mean()) < 3 * sem),
            "max_abs": float(np.max(np.abs(v)))}


def main():
    raw = git_bytes(SOURCE)
    data = json.loads(raw)
    rows = data["rows"]
    out = {"source_commit": REF, "source_path": SOURCE,
           "source_sha256": hashlib.sha256(raw).hexdigest(),
           "n_rows": len(rows), "source_flags": {k: data[k] for k in ("P9", "P10", "P11")}}
    out["integer_polynomial_checks"] = {
        "all_nonnegative_integers": all(all(isinstance(h, int) and h >= 0 for h in r["h"]) for r in rows),
        "all_h0_h1_zero": all(r["h"][:2] == [0, 0] for r in rows),
        "all_total_N_minus_3": all(sum(r["h"]) == r["N"] - 3 for r in rows),
        "all_parities_match_reflection": all(all(h % 2 == (1 if j == 2 * sum(factors(r["N"] // 2).values()) else 0)
                                                 for j, h in enumerate(r["h"])) for r in rows),
    }
    summaries = {}
    for scale in (10**7, 10**8, 10**9):
        rr = [r for r in rows if r["scale"] == scale]
        ss = {"b2_full": stats([r["rel_b2"] for r in rr])}
        for z in ZS:
            for div in (True, False):
                group = [r for r in rr if (r["mod3"] == 0) == div]
                key = f"z={z},3div={div}"
                ss[key] = stats([r[f"E_{z}"][0] for r in group])
                ss[key]["imag"] = stats([r[f"E_{z}"][1] for r in group])
        summaries[str(scale)] = ss
    out["summaries_ddof1"] = summaries
    flags = {"P9": True, "P10": True, "P11": True}
    for scale, ss in summaries.items():
        flags["P11"] &= ss["b2_full"]["three_sem_pass"]
        for z in (0.25, 0.5, -0.25):
            for div in (True, False):
                st = ss[f"z={z},3div={div}"]
                if st["n"] >= 8 and (z != -0.25 or int(scale) >= 10**8):
                    flags["P10" if z == -0.25 else "P9"] &= st["three_sem_pass"]
    out["flags_ddof1"] = flags
    coeffs = []
    completion = []
    fd_errors = []
    for row in rows:
        n = row["N"]
        (full, err_f), (cut, err_c) = coefficient_integrals(n)
        bb = r_local(n, 0).real * full
        coeffs.append({"N": n, "scale": row["scale"], "full_minus_cut_over_full": (full-cut)/full,
                       "relative_recompute_difference_full_b2": (row["h"][2]/bb-1)-row["rel_b2"],
                       "relative_residual_actual_cut_coefficient": row["h"][2]/(bb*cut/full)-1,
                       "quad_est_relative_error": max(err_f/full, err_c/cut)})
        for z in (0.25, 0.5, -0.25):
            delta = -(z-1)/math.log(n)*weighted_derivative(n,z)
            old = row[f"E_{z}"][0]
            linear = (1+old)*np.exp(delta)/(1+delta)-1
            completion.append({"N": n, "scale": row["scale"], "div3": row["mod3"]==0, "z":z,
                               "delta":float(delta), "exp_residual":old, "linear_residual":float(linear)})
    out["coefficient_checks"] = coeffs
    out["first_order_completion_comparison"] = completion
    representative = [next(r["N"] for r in rows if r["scale"]==s and r["mod3"]==0) for s in (10**7,10**8,10**9)]
    for n in representative:
        for z in ZS:
            fd = finite_difference_sum(n,z)
            exact = weighted_derivative(n,z)
            fd_errors.append({"N":n,"z":str(z),"abs_delta_difference":float(abs((z-1)/math.log(n)*(fd-exact)))})
    out["finite_difference_checks"] = fd_errors
    numerical = []
    n = representative[-1]
    for z in (0.25, 0.5, -0.25, 0.5j):
        models = {}
        moments = {}
        for label, order, limit, rule in (("submitted_rule",400,200000,"legendre"),
                                         ("jacobi320",320,200000,"jacobi"),
                                         ("jacobi640",640,200000,"jacobi"),
                                         ("prime1m",640,1000000,"jacobi"),
                                         ("legendre4_320",320,1000000,"legendre4"),
                                         ("legendre4_640",640,1000000,"legendre4")):
            y,c = density_grid(z,order,limit,rule)
            val,mom = convolution(n,y,c)
            approx,_ = convolution(n,y,c,False)
            models[label] = val
            moments[label] = mom/val
            numerical.append({"N":n,"z":str(z),"label":label,
                              "integral":[float(val.real),float(val.imag)],
                              "relative_endpoint_approximation_error":float(abs(approx/val-1))})
        ref = models["legendre4_640"]
        for item in numerical:
            if item["z"]==str(z):
                vv=complex(*item["integral"])
                item["relative_difference_from_legendre4_640_prime1m"] = float(abs(vv/ref-1))
        row = next(r for r in rows if r["N"]==n)
        s = weighted_derivative(n,z)
        hz = np.polynomial.polynomial.polyval(z,np.array(row["h"],float))
        base = r_local(n,z)*models["legendre4_640"]
        unlogged = base*(1-s*moments["legendre4_640"])
        out.setdefault("mellin_linear_diagnostics",[]).append({"N":n,"z":str(z),
                      "relative_residual":[float((hz/unlogged-1).real),float((hz/unlogged-1).imag)]})
    out["numerical_sensitivity_not_rigorous_error_bounds"] = numerical
    out["notes"] = ["The prime tail uses the submitted PNT/E1 approximation, not interval bounds.",
                    "Completion comparisons are exploratory, using saved H/Bexp and algebraic Delta.",
                    "Saved integer consistency does not independently recount the 1e9 samples."]
    path = Path(__file__).with_name("audit_results.json")
    path.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"path":str(path),"flags_ddof1":flags,"checks":out["integer_polynomial_checks"],
                      "max_full_b2_recompute_difference":max(abs(c["relative_recompute_difference_full_b2"]) for c in coeffs),
                      "max_delta_fd_difference":max(c["abs_delta_difference"] for c in fd_errors)},indent=2))


if __name__ == "__main__":
    main()
