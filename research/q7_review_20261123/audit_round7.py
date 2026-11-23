"""Retrospective audit of round seven; no new large experiment.

Run from the repository root:
    python research/q7_review_20261123/audit_round7.py

Rechecks saved integer polynomials, finite-scale regression, and a local Euler
factor counterexample. The latter has an exact Gaussian-rational Rouche
certificate; numerical root finding only proposes its rational center.
No 1e8/1e9 Omega sieve is run. Writes round7_audit.json beside this file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy import special


SOURCE_COMMIT = "17c42efdfc7101d25e8ad626a9267805ae841a05"


def source_json(name):
    return json.loads(subprocess.check_output([
        "git", "show", SOURCE_COMMIT + ":research/Q1_fugacity/out/" + name,
    ]))


def winding(values):
    differences = np.angle(np.roll(values, -1) / values)
    turns = float(differences.sum() / (2 * np.pi))
    return int(round(turns)), float(np.max(np.abs(differences))), turns


def p2_numerator(e):
    coeff = [0] * (2 * e + 2)
    coeff[-1] = 1
    for j in range(e):
        scale = 2 ** (e - 1 - j)
        coeff[2 * j] += 2 * scale
        coeff[2 * j + 1] -= scale
    return coeff


def gaussian_add(a, b):
    return a[0] + b[0], a[1] + b[1]


def gaussian_multiply(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def translate_polynomial(coeff, center):
    # Exact coefficients of P(center+u), ascending in powers of u.
    translated = [(F(coeff[-1]), F(0))]
    for constant in reversed(coeff[:-1]):
        new = [(F(0), F(0))] * (len(translated) + 1)
        for k, item in enumerate(translated):
            new[k] = gaussian_add(new[k], gaussian_multiply(center, item))
            new[k + 1] = gaussian_add(new[k + 1], item)
        new[0] = gaussian_add(new[0], (F(constant), F(0)))
        translated = new
    return translated


def certify_p2_root():
    mp.mp.dps = 70
    coeff = p2_numerator(22)
    approximate = mp.findroot(
        lambda z: mp.polyval(list(reversed(coeff)), z),
        mp.mpc("-1.3961849784465", ".2078587210913"),
        tol=mp.mpf("1e-60"),
    )
    center = F(mp.nstr(approximate.real, 24)), F(mp.nstr(approximate.imag, 24))
    radius = F(1, 10**9)
    translated = translate_polynomial(coeff, center)
    # Upper bounds |a_k| <= |Re a_k|+|Im a_k| and a lower bound
    # |a_1| >= max(|Re a_1|,|Im a_1|) use rational arithmetic only.
    upper = abs(translated[0][0]) + abs(translated[0][1])
    for k in range(2, len(translated)):
        upper += (abs(translated[k][0]) + abs(translated[k][1])) * radius**k
    lower = max(abs(translated[1][0]), abs(translated[1][1])) * radius
    assert upper < lower
    assert (center[0] + 1) ** 2 + center[1] ** 2 < (F(9, 20) - radius) ** 2
    # Hence the disk contains exactly one root by Rouche; real coefficients
    # give a disjoint conjugate disk containing another root.
    assert center[1] > radius

    boundary = -1 + 0.45 * np.exp(2j * np.pi * np.arange(8192) / 8192)

    def jd(p, e, z):
        value = (p - 2 + z) / (p - z)
        for _ in range(e):
            value = 1 - 1 / p + z**2 * value / p
        return value

    # N=25*2^22 has only primes 2 and 5 as divisors, while 3 does not divide N.
    # Remaining Euler factors are nonzero in this disk. This numerical winding
    # is supplementary; the exact certificate already establishes >=5 zeros.
    local_model = special.rgamma(boundary)**2 * jd(2, 22, boundary) * jd(3, 0, boundary) * jd(5, 2, boundary)
    count, step, _ = winding(local_model)
    return {
        "N": 25 * 2**22, "nu2": 22, "mod3": (25 * 2**22) % 3,
        "polynomial_coefficients_ascending": coeff,
        "root_center_real_rational": str(center[0]),
        "root_center_imag_rational": str(center[1]),
        "certificate_disk_radius": str(radius),
        "exact_rouche_inequality_pass": True,
        "exact_disk_containment_pass": True,
        "upper_over_lower_diagnostic": float(upper / lower),
        "approximate_root": [float(approximate.real), float(approximate.imag)],
        "distance_to_minus_one": float(abs(approximate + 1)),
        "rigorous_model_zero_count_lower_bound": 5,
        "supplementary_numerical_model_winding": count,
        "supplementary_max_phase_step": step,
    }


def main():
    prior_path = Path(__file__).resolve().parents[1] / "q6_review_20261123/audit_partition_framework.py"
    spec = importlib.util.spec_from_file_location("_round6_audit_util", prior_path)
    previous = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = previous
    spec.loader.exec_module(previous)

    q4b = source_json("q4b_summary.json")
    q4c = source_json("q4c_summary.json")
    q4d = source_json("q4d_summary.json")
    scales = np.array([1e6, 1e7, 1e8, 1e9])
    log_n, log_log_n = np.log(scales), np.log(np.log(scales))
    slope = float(np.polyfit(log_n, log_log_n, 1)[0])
    exponent_rows = []
    for key, row in q4c.items():
        x = float(key)
        error = np.asarray(row["median_abs_error"])
        alpha = float(np.polyfit(log_n, np.log(error), 1)[0])
        assert abs(alpha - row["alpha_error"]) < 1e-12
        exponent_rows.append({
            "x": x, "reported_alpha": row["alpha_error"], "recomputed_alpha": alpha,
            "adjacent_decade_slopes": (np.diff(np.log(error)) / math.log(10)).tolist(),
            "last_three_scales_slope": float(np.polyfit(log_n[1:], np.log(error)[1:], 1)[0]),
            "first_omitted_log_term_apparent_slope": None if x == -1 else 1 + (2 * x - 3) * slope,
        })
    log_e = np.log(q4c["-1.0"]["median_abs_error"])
    fitted_power = np.polyval(np.polyfit(log_n, log_e, 1), log_n)
    constructed_log8 = log_n - 8 * log_log_n
    constructed_log8 += np.mean(log_e - constructed_log8)
    alternative_fit = {
        "free_power_log_rms": float(np.sqrt(np.mean((log_e - fitted_power)**2))),
        "constructed_N_over_log8_log_rms": float(np.sqrt(np.mean((log_e - constructed_log8)**2))),
        "warning": "A deliberately post-hoc counterfit demonstrating non-identifiability; not a proposed law",
    }
    phase_slopes = [{
        "theta": row["theta"], "reported_alpha": row["alpha_error"],
        "first_omitted_log_term_apparent_slope": 1 + (math.cos(row["theta"]) - 3) * slope,
    } for row in q4d]

    mixtures = {}
    for seed in (61, 71):
        rng = np.random.default_rng(seed)
        rows = []
        for scale in (10**6, 10**7, 10**8, 10**9):
            totals = 2 * rng.integers(scale // 2, int(1.1 * scale) // 2, size=20)
            rows.append({"scale": scale, "sample_size": 20,
                         "multiples_of_3": int(np.count_nonzero(totals % 3 == 0))})
        mixtures[str(seed)] = rows

    winding_checks = []
    root_disagreements = []
    min_boundary_root_distance = float("inf")
    max_step = 0.0
    for scale in ("1000000", "10000000", "100000000"):
        for row in q4b[scale]["rows"]:
            h = np.trim_zeros(np.asarray(row["h"], dtype=np.int64), "b")
            assert int(h.sum()) == row["N"] - 3
            roots = np.polynomial.polynomial.polyroots(h.astype(float))
            distances = np.abs(roots + 1)
            count_roots = int(np.count_nonzero(distances < 0.45))
            min_boundary_root_distance = min(min_boundary_root_distance, float(np.min(np.abs(distances - 0.45))))
            counts = []
            for points in (2048, 4096, 8192):
                boundary = -1 + 0.45 * np.exp(2j * np.pi * np.arange(points) / points)
                values = np.polynomial.polynomial.polyval(boundary, h.astype(float))
                count, step, turns = winding(values)
                assert abs(turns - round(turns)) < 1e-8
                counts.append(count)
                if points == 8192:
                    max_step = max(max_step, step)
            if any(n != row["nH"] for n in counts) or count_roots != row["nH"]:
                root_disagreements.append({"N": row["N"], "source": row["nH"],
                                           "grids": counts, "polynomial_roots": count_roots})
        winding_checks.append({
            "scale": int(scale), "n": len(q4b[scale]["rows"]),
            "nu2_max": max(row["nu2"] for row in q4b[scale]["rows"]),
            "rouche_grid_min": min(row["rouche"] for row in q4b[scale]["rows"]),
            "rouche_grid_median": float(np.median([row["rouche"] for row in q4b[scale]["rows"]])),
        })
    assert not root_disagreements

    # Independently rebuild three small saved polynomials, not the large runs.
    omega = previous.omega_by_prime_powers(1_100_000)
    for row in q4b["1000000"]["rows"][:3]:
        total = row["N"]
        degrees = omega[2 : total - 1] + omega[total - 2 : 1 : -1]
        actual = np.bincount(degrees.astype(np.int64), minlength=len(row["h"]))
        assert np.array_equal(actual, np.asarray(row["h"]))

    # Finite check of the mean-drift calculation. The asymptotic proof is in
    # the review, not inferred from these small numbers.
    limit = 1_000_000
    primes = previous.prime_table(limit)
    singular = np.zeros(limit + 1)
    singular[2::2] = 2 * 0.6601618158468695739278121100145
    for p in primes[1:]:
        if 2 * p > limit:
            break
        singular[2 * p :: 2 * p] *= (p - 1) / (p - 2)
    average_checks = []
    for cutoff in (100_000, 1_000_000):
        odd_primes = primes[(primes > 2) & (primes <= cutoff - 3)]
        true_pairs = int(np.searchsorted(odd_primes, cutoff - odd_primes, side="right").sum()) + 1
        evens = np.arange(4, cutoff + 1, 2)
        main_sum = float(np.sum(evens * singular[evens] / np.log(evens)**2))
        average_checks.append({
            "cutoff": cutoff, "exact_cumulative_even_ordered_prime_pairs": true_pairs,
            "cumulative_leading_model_coefficient": main_sum,
            "difference_scaled_by_log3_over_X2": (true_pairs - main_sum) * math.log(cutoff)**3 / cutoff**2,
        })

    q4a = source_json("q4a_summary.json")
    earlier = json.loads(prior_path.with_name("partition_audit.json").read_text(encoding="utf-8"))
    density_comparison = {}
    for key in ("0.25", "0.5", "0.75"):
        old = earlier["Q2d_reproduction"]["results"][key]["corrected_density_integral_signed_mean"]
        current = q4a["1000000"][key]["mean"]
        assert abs(old - current) < 1e-6
        density_comparison[key] = {"previous_independent_audit": old, "new_Q4a_report": current}

    result = {
        "schema_version": 1, "hub_baseline": "7e46846", "source_commit": SOURCE_COMMIT,
        "purpose": "Retrospective round-seven audit and exact local-root certificate",
        "preregistered": False, "status": "PASS",
        "log_log_regression_slope": slope,
        "real_axis_exponent_checks": exponent_rows,
        "posthoc_counterfit": alternative_fit,
        "phase_exponent_log_term_comparison": phase_slopes,
        "sample_mod3_composition": mixtures,
        "saved_polynomial_audit": {
            "cases": 180, "grids": [2048, 4096, 8192],
            "disagreements": root_disagreements,
            "minimum_numerical_root_distance_to_boundary": min_boundary_root_distance,
            "maximum_phase_step_on_8192_grid": max_step,
            "scales": winding_checks, "small_independent_histograms_rebuilt": 3,
            "note": "Root locations and grid counts are stability checks, not interval-certified counts",
        },
        "high_nu2_counterexample": certify_p2_root(),
        "finite_average_checks": average_checks,
        "Q4a_against_previous_independent_audit": density_comparison,
        "limits": [
            "No independent 1e8/1e9 Omega run",
            "No bootstrap of Q4c/d: per-N errors are not saved in their summaries",
            "The exact Gaussian-rational certificate concerns the model local factor, not an actual H_N root",
            "Finite average checks do not prove the asymptotic; the proof is supplied separately",
        ],
        "script_hash_scheme": "sha256-utf8-lf-v1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    }
    output = Path(__file__).with_name("round7_audit.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
