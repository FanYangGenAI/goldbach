"""Audit round-13 fixes and isolate the finite-q cubic moment problem.

New cubic computations use no-root and quadratic-discriminant tests, rather
than the experiment's polynomial factor sieve. Only prime q are implemented.
All added computations are exploratory, not preregistered confirmations.
"""
from __future__ import annotations

import ast
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy import fft

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("audit_results.json")
FIX = "42d54e8"
DILATION = "fd22f54"
HUB = "90d49b7"


def moment_candidates(q):
    """Exploratory rational-polynomial reconstructions, NOT proved identities.

    Per residue class, the first 10 prime fields reconstructed the degree-9
    second moment; the first 7 reconstructed each degree-6 class first moment.
    Later fields, including q=131,139, are numerical checks only.
    """
    if q % 3 == 0:
        return None
    same = q*(q-2)*(q-1)**2*(q+1)*(q+2)//36
    if q % 3 == 1:
        return {"sum_r2": q*(q-1)**2*(q**6+2*q**5-3*q**3+9*q-3)//36,
                "I3": q*(q-1)**2*(q**3+q*q-q-4)//18,
                "L_I2": q*q*(q-1)**3*(q+2)//12,
                "L_L_L": same,
                "L2_L": q*q*(q-1)**2*(q+2)//6,
                "L3": q*(q-1)**2*(q+2)//6}
    return {"sum_r2": q*(q-1)*(q+1)*(q**6-2*q**4-q**3+4*q*q+3*q+5)//36,
            "I3": q*(q-2)*(q-1)*(q+1)*(q*q+q+1)//18,
            "L_I2": q*(q-1)*(q+1)*(q**3-q*q-2*q+4)//12,
            "L_L_L": same,
            "L2_L": q*(q-1)*(q+1)*(q*q-2)//6,
            "L3": q*q*(q-1)*(q+1)//6}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8")


def read(ref, path):
    return json.loads(git("show", f"{ref}:{path}"))


def fix_audit():
    current = read(FIX, "research/R13_ff_chi_fix/out/r13_summary.json")
    old = read("b73ccb4", "research/O7_ff_refined/out/o7_summary.json")
    lookup = {(r["q"], r["n"]): r for r in old["rows"]}
    errors = []
    for r in current["rows"]:
        prior = lookup[(r["q"], r["n"])]
        errors += [abs(r["chi_code_mean"]-prior["chi_mean"]), abs(r["chi_code_sd"]-prior["chi_sd"])]
    # Test source definitions against independent influence-function algebra.
    tree = ast.parse(git("show", f"{FIX}:research/R13_ff_chi_fix/r13_ff_chi_fix.py"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("chi_conn", "nullv3_rows")]
    ns = {"np": np}
    exec(compile(ast.Module(nodes, type_ignores=[]), "<audited-functions>", "exec"), ns)
    rng = np.random.default_rng(2026112414)
    p = rng.dirichlet([1, 1, 1, 1], size=1000)
    c = np.array([1, -1, -1, 1]); left = np.array([-1, -1, 1, 1]); right = np.array([-1, 1, -1, 1])
    ml, mr = p@left, p@right
    expected_chi = p@c-ml*mr
    expected_u = c-mr[:, None]*left-ml[:, None]*right
    expected_var = (p*expected_u**2).sum(axis=1)-(p*expected_u).sum(axis=1)**2
    assert np.max(abs(ns["chi_conn"](*p.T)-expected_chi)) < 1e-12
    assert np.max(abs(ns["nullv3_rows"](p)-expected_var)) < 1e-12
    return {"n_rows": len(current["rows"]), "old_stat_max_error": max(errors),
            "rho_range": [min(r["div"]["rho"] for r in current["rows"]), max(r["div"]["rho"] for r in current["rows"])],
            "R2_range": [min(r["div"]["R2_pred"] for r in current["rows"]), max(r["div"]["R2_pred"] for r in current["rows"])],
            "D_after_range": [min(r["div"]["D_after"] for r in current["rows"]), max(r["div"]["D_after"] for r in current["rows"])],
            "max_abs_observed_mean": max(abs(r["chi_mean"]) for r in current["rows"]),
            "max_relative_sd_change": max(abs(r["chi_sd"]/r["chi_code_sd"]-1) for r in current["rows"]),
            "all_meet_support_rule": all(r["div"]["rho"]>.5 and .5<=r["div"]["sd_ratio"]<=2 and r["div"]["R2_pred"]>.25 for r in current["rows"]),
            "source_function_checks": "chi and 3-variable variance match independent algebra on 1000 probability tables"}


def dilation_audit():
    x = read(DILATION, "research/R13_liouville_dilation/out/r13_dilation.json")
    rows = []
    for key, value in x["results"].items():
        if key.startswith("N,") or key in ("D1", "D2"):
            continue
        a, b = [int(v[:-1]) for v in key.split(",")]
        predicted = math.gcd(a, b)/math.sqrt(a*b)
        assert abs(value["pred"]-predicted) < 1e-15
        rows.append({"a": a, "b": b, "observed": value["corr"], "prediction": predicted, "error": value["corr"]-predicted})
    changes = git("diff", "--name-only", "6f50321", DILATION, "--", "research/R13_liouville_dilation").splitlines()
    assert "research/R13_liouville_dilation/r13_dilation.py" not in changes
    return {"range": x["range"], "rows": rows, "max_abs_error": max(abs(r["error"]) for r in rows),
            "control": x["results"]["N,N+2"], "runner_unchanged_after_prereg": True}


def cubic(q):
    assert q % 2 and all(q % d for d in range(2, math.isqrt(q)+1))
    a2, a1 = np.indices((q, q), dtype=np.int64)
    roots = np.zeros((q, q, q), np.int8)
    # For each a2,a1,x, there is exactly one a0 for which F(x)=0.
    for x in range(q):
        a0 = (-x**3-a2*x*x-a1*x) % q
        roots[a2, a1, a0] += 1
    p = roots == 0
    quadratic = np.zeros((q, q, q), bool)
    char = -np.ones(q, np.int8); char[0] = 0
    char[(np.arange(1, q, dtype=np.int64)**2) % q] = 1
    for lead in range(1, q):
        disc = (np.arange(q, dtype=np.int64)[:, None]**2-4*lead*np.arange(q)[None, :]) % q
        quadratic[lead] = char[disc] == -1
    low = quadratic.copy(); low[0, 1:, :] = True
    pi3 = (q**3-q)//3; low_size = q*(q*q-1)//2
    assert int(p.sum()) == pi3 and int(low.sum()) == low_size
    ph = fft.rfftn(p.astype(float)); lh = fft.rfftn(low.astype(float))
    real_r = fft.irfftn(ph*lh, s=(q, q, q))
    round_error = float(abs(real_r-np.rint(real_r)).max())
    r = np.rint(real_r).astype(np.int64)
    assert round_error < .001
    triple = np.zeros_like(p)
    for x in range(q):
        triple[(-3*x) % q, (3*x*x) % q, (-x**3) % q] = True
    masks = {"I3": roots==0, "L_I2": (roots==1)&~triple,
             "L_L_L": roots==3, "L2_L": roots==2, "L3": triple}
    ntypes = {"I3": pi3, "L_I2": q*q*(q-1)//2,
              "L_L_L": q*(q-1)*(q-2)//6, "L2_L": q*(q-1), "L3": q}
    alpha = {d: (q**d-1)/(q**d-2) for d in (1, 2, 3)}
    weights = {"I3": alpha[3], "L_I2": alpha[1]*alpha[2],
               "L_L_L": alpha[1]**3, "L2_L": alpha[1]**2, "L3": alpha[1]}
    pi = {1: q, 2: q*(q-1)//2, 3: pi3}
    c3 = math.exp(sum(pi[d]*math.log1p(-1/(q**d-1)**2) for d in pi))
    Q = q**3
    total = int(r.sum()); second = int(np.sum(r*r))
    assert total == pi3*low_size
    K = total/Q
    bg = np.zeros_like(real_r)
    groups = {}
    for kind, mask in masks.items():
        assert int(mask.sum()) == ntypes[kind]
        assert np.max(r[mask])**2 * Q < 9e18
        bg[mask] = K*c3*weights[kind]
        groups[kind] = {"n": int(mask.sum()), "sum_r": int(r[mask].sum()),
                        "sum_r2": int(np.sum(r[mask]*r[mask])), "weight": weights[kind]}
    residual = r-bg
    variance = float(residual.var())
    # Same moment formula at high arithmetic precision, using exactly the
    # floating-point background value assigned to each type in the direct run.
    with localcontext() as ctx:
        ctx.prec = 60
        bgv = {t: Decimal.from_float(float(K*c3*weights[t])) for t in groups}
        mean_b = sum(Decimal(g["n"])*bgv[t] for t, g in groups.items())/Q
        second_b = sum(Decimal(g["n"])*bgv[t]**2 for t, g in groups.items())/Q
        cross = sum(Decimal(g["sum_r"])*bgv[t] for t, g in groups.items())/Q
        var_formula = Decimal(second)/Q+second_b-2*cross-(Decimal(total)/Q-mean_b)**2
    # The scalar-shift component of the cubic autocorrelation is a known exact theorem.
    ap = np.rint(fft.irfftn(abs(ph)**2, s=(q, q, q))).astype(np.int64)
    al = np.rint(fft.irfftn(abs(lh)**2, s=(q, q, q))).astype(np.int64)
    scalar_sum = int(ap[0, 0, 1:].sum())
    kappa = {0: 3, 1: 6, 2: 2}[q % 3]
    expected_scalar_sum = ((q-1)*(q**3-q*q-kappa*q))//9
    assert scalar_sum == expected_scalar_sum
    assert np.all(al[0, 0, 1:] == q*(q*q-1)//4)
    if q <= 11:
        idx = np.indices((q, q, q))
        for pos in [(0, 0, 0), (0, 1, 1), (1, 2, 3)]:
            pos = tuple(v % q for v in pos)
            opposite = tuple((pos[i]-idx[i]) % q for i in range(3))
            assert int(np.sum(low*p[opposite])) == int(r[pos])
    candidate = moment_candidates(q)
    candidate_errors = None
    if candidate is not None:
        candidate_errors = {t: (second if t == "sum_r2" else groups[t]["sum_r"])-v for t, v in candidate.items()}
        assert all(e == 0 for e in candidate_errors.values())
    return {"q": q, "q_mod3": q % 3, "Q": Q, "mean_r": K, "sum_r": total, "sum_r2": second,
            "groups": groups, "C3": c3, "var_residual": variance, "c2": variance/K,
            "q_c2": q*variance/K, "q_times_qc2_minus_1": q*(q*variance/K-1),
            "moment_formula_variance": float(var_formula), "moment_formula_error": abs(float(var_formula)-variance),
            "FFT_round_error": round_error, "sum_scalar_cubic_twins": scalar_sum,
            "BarySoroker_Stix_scalar_sum": expected_scalar_sum,
            "conjectured_integer_moment_errors": candidate_errors}


def main():
    out = {"hub": git("rev-parse", HUB).strip(), "fix_commit": git("rev-parse", FIX).strip(),
           "dilation_commit": git("rev-parse", DILATION).strip(),
           "fix": fix_audit(), "dilation": dilation_audit(), "cubic": []}
    original = read("335878f", "research/O10_ff_largeq/out/o10_summary.json")
    lookup = {r["q"]: r["c2_hl"] for r in original["rows"] if r["n"] == 3}
    # Extend both residue classes without using the old factor-sieve implementation.
    qs = [q for q in range(3, 128, 2) if all(q % d for d in range(2, math.isqrt(q)+1))]+[131, 139]
    for q in qs:
        r = cubic(q)
        if q in lookup:
            r["difference_from_O10"] = r["c2"]-lookup[q]
            assert abs(r["difference_from_O10"]) < 1e-10
        out["cubic"].append(r)
        OUT.write_text(json.dumps(out, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(q, r["q_c2"], r["q_times_qc2_minus_1"], flush=True)
    print(json.dumps({"output": str(OUT), "fix": out["fix"], "dilation_max_error": out["dilation"]["max_abs_error"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
