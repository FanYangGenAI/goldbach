"""Retrospective, bounded audit of Q1/Q2, not a preregistered experiment.

Run from the repository root:
    python research/q6_review_20261123/audit_partition_framework.py

Counts Omega by an independent prime-power sieve (not the source SPF sieve).
Reproduces the 100 Q2d samples at scale 1e6 only, audits single-variable
secondary terms through 1e7, and checks phase norms on a finite grid.
Does not rerun any 1e8 or 1e9 experiment. Only the adjacent JSON is written.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import numba as nb
import numpy as np
import scipy
from scipy import integrate, special


SOURCE_COMMIT = "1fdca2376704ecbc2e98e40f0095d3e79ea6e725"
HUB_COMMIT = "cdd152d"
GAMMA = 0.5772156649015328606


@nb.njit(cache=False)
def omega_by_prime_powers(limit):
    omega = np.zeros(limit + 1, dtype=np.uint8)
    for p in range(2, limit + 1):
        if omega[p] == 0:
            power = p
            while power <= limit:
                for multiple in range(power, limit + 1, power):
                    omega[multiple] += 1
                if power > limit // p:
                    break
                power *= p
    return omega


@nb.njit(parallel=True, cache=False)
def count_tables(totals, omega, size):
    out = np.zeros((len(totals), size, size), dtype=np.int64)
    for i in nb.prange(len(totals)):
        total = totals[i]
        for n in range(2, total - 1):
            out[i, omega[n], omega[total - n]] += 1
    return out


def prime_table(limit):
    prime = np.ones(limit + 1, dtype=bool)
    prime[:2] = False
    for p in range(2, math.isqrt(limit) + 1):
        if prime[p]:
            prime[p * p :: p] = False
    return np.flatnonzero(prime)


def factors(n):
    result = {}
    p = 2
    while p * p <= n:
        while n % p == 0:
            result[p] = result.get(p, 0) + 1
            n //= p
        p += 1
    if n > 1:
        result[n] = result.get(n, 0) + 1
    return result


def local_factor(p, e, z, w):
    value = 1 - 2 / p + (
        z * (1 - 1 / p) / (1 - z / p)
        + w * (1 - 1 / p) / (1 - w / p)
    ) / p
    for _ in range(e):
        value = 1 - 1 / p + z * w * value / p
    return value


def local_fraction(p, e, z, w):
    p = Fraction(p)
    value = 1 - 2 / p + (
        z * (1 - 1 / p) / (1 - z / p)
        + w * (1 - 1 / p) / (1 - w / p)
    ) / p
    for _ in range(e):
        value = 1 - 1 / p + z * w * value / p
    return value


class EulerModel:
    def __init__(self, primes):
        self.primes = primes
        self.pmax = int(primes[-1])
        self.cache = {}

    def base(self, z, w):
        # The p=2 unshifted factor can vanish. Omit it and insert its actual
        # even-N factor separately. Multiply the other factors directly.
        key = (np.asarray(z).shape, np.asarray(w).shape,
               np.asarray(z).tobytes(), np.asarray(w).tobytes())
        if key not in self.cache:
            shape = np.broadcast_shapes(np.shape(z), np.shape(w))
            value = np.ones(shape, dtype=np.complex128)
            for p in self.primes[1:]:
                value *= local_factor(int(p), 0, z, w) * np.exp(
                    (z + w - 2) * math.log1p(-1 / int(p))
                )
            self.cache[key] = value
        return self.cache[key].copy()

    def value(self, total, z, w):
        assert total % 2 == 0
        value = self.base(z, w)
        for p, e in factors(total).items():
            if p == 2:
                value *= local_factor(p, e, z, w) * np.exp(
                    (z + w - 2) * math.log(0.5)
                )
            elif p <= self.pmax:
                value *= local_factor(p, e, z, w) / local_factor(p, 0, z, w)
            else:
                value *= local_factor(p, e, z, w) * np.exp(
                    (z + w - 2) * math.log1p(-1 / p)
                )
        return (total * math.log(total) ** (z + w - 2) * value
                * special.rgamma(z) * special.rgamma(w))

    def coefficient_11(self, total):
        value = self.base(0.0, 0.0).real.item()
        for p, e in factors(total).items():
            if p == 2:
                value *= local_factor(p, e, 0.0, 0.0) * 4
            elif p <= self.pmax:
                value *= local_factor(p, e, 0.0, 0.0) / local_factor(p, 0, 0.0, 0.0)
            else:
                value *= local_factor(p, e, 0.0, 0.0) / (1 - 1 / p) ** 2
        return total / math.log(total) ** 2 * value


def one_variable_terms(z, primes):
    p = primes.astype(float)
    log_h = np.sum(z * np.log1p(-1 / p) - np.log1p(-z / p))
    log_derivative_h = z * (1 - z) * np.sum(np.log(p) / ((p - 1) * (p - z)))
    density_bracket = z * GAMMA + log_derivative_h
    cumulative_relative = (z - 1) * (density_bracket - 1)
    return float(np.exp(log_h) * special.rgamma(z)), float(density_bracket), float(cumulative_relative)


def integral_ratio(total, z, density_bracket):
    log_n = math.log(total)
    coeff = (z - 1) * density_bracket

    def integrand(u):
        left = math.log(u * total)
        right = math.log((1 - u) * total)
        return ((left / log_n) ** (z - 1) * (right / log_n) ** (z - 1)
                * (1 + coeff / left) * (1 + coeff / right))

    result, error = integrate.quad(integrand, 2 / total, 0.5,
                                  epsabs=1e-10, epsrel=1e-10, limit=250)
    assert error < 1e-7
    return 2 * result


def load_source_json(filename):
    return json.loads(subprocess.check_output([
        "git", "show", SOURCE_COMMIT + ":research/Q1_fugacity/out/" + filename,
    ]))


def main():
    nb.set_num_threads(min(4, nb.get_num_threads()))
    primes = prime_table(1_000_000)
    omega = omega_by_prime_powers(10_000_000)
    # Independent spot checks using trial factorization, including prime powers.
    for n in (2, 4, 8, 12, 81, 992, 1009, 100_000, 1_000_000, 9_999_991):
        assert int(omega[n]) == sum(factors(n).values())
    rng = np.random.default_rng(41)
    totals = (2 * rng.integers(500_000, 550_000, size=100)).astype(np.int64)
    size = int(omega.max()) + 1
    tables = count_tables(totals, omega, size)
    assert np.all(tables.sum(axis=(1, 2)) == totals - 3)
    assert np.array_equal(tables, tables.transpose(0, 2, 1))
    xs = (0.25, 0.5, 0.75)
    model = EulerModel(primes)
    published_d = load_source_json("q2d_summary.json")["1000000"]
    published_f = load_source_json("q2f_summary.json")["1000000"]
    reproduction = {}
    single = {}
    for x in xs:
        power = x ** np.arange(size)
        observed = np.einsum("j,njk,k->n", power, tables, power)
        main_values = np.array([model.value(int(n), x, x).real.item() for n in totals])
        relative = observed / main_values - 1
        leading, bracket, cumulative = one_variable_terms(x, primes)
        correct_ratio = np.array([integral_ratio(int(n), x, bracket) for n in totals])
        wrong_ratio = np.array([integral_ratio(int(n), x, bracket + 1) for n in totals])
        corrected_errors = observed / (main_values * correct_ratio) - 1
        wrong_errors = observed / (main_values * wrong_ratio) - 1
        key = str(x)
        assert abs(relative.mean() - published_d[key]["mean"]) < 1e-8
        # The source integrates the unnormalized, much smaller integrand at
        # scipy's default absolute tolerance. Allow that quadrature difference.
        assert abs(wrong_errors.mean() - published_f[key]["mean"]) < 1e-6, (
            key, float(wrong_errors.mean()), published_f[key]["mean"]
        )
        reproduction[key] = {
            "base_signed_mean": float(relative.mean()),
            "published_base_signed_mean": published_d[key]["mean"],
            "source_Q2f_signed_mean": float(wrong_errors.mean()),
            "published_Q2f_signed_mean": published_f[key]["mean"],
            "Q2f_reproduction_absolute_difference": abs(float(wrong_errors.mean()) - published_f[key]["mean"]),
            "corrected_density_integral_signed_mean": float(corrected_errors.mean()),
            "corrected_density_integral_sd": float(corrected_errors.std()),
            "density_bracket_correct": bracket,
            "density_bracket_source_Q2f": bracket + 1,
            "cumulative_relative_coefficient_correct": cumulative,
        }
        rows = []
        for n in (1_000_000, 10_000_000):
            hist = np.bincount(omega[2 : n + 1].astype(np.int64))
            actual = float(np.dot(hist, x ** np.arange(len(hist))))
            baseline = leading * n * math.log(n) ** (x - 1)
            corrected = baseline * (1 + cumulative / math.log(n))
            wrong = baseline * (1 + (cumulative + x - 1) / math.log(n))
            rows.append({
                "x_limit": n, "exact_weighted_sum_excluding_1": actual,
                "relative_error_leading": actual / baseline - 1,
                "relative_error_correct_second_term": actual / corrected - 1,
                "relative_error_Q2f_bracket_induced_second_term": actual / wrong - 1,
            })
        single[key] = rows

    # An exact local identity: for p not dividing N, J_p(x,x)=(p-2+x)/(p-x).
    exact_local_cases = 0
    for p in (2, 3, 5, 7, 11):
        for x in (Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)):
            assert local_fraction(p, 0, x, x) == (p - 2 + x) / (p - x)
            assert local_fraction(p, 0, x, Fraction(1)) == (p - 1) / (p - x)
            exact_local_cases += 1
    assert local_fraction(3, 0, Fraction(-1), Fraction(-1)) == 0
    assert all(local_fraction(3, e, Fraction(-1), Fraction(-1)) > 0 for e in range(1, 8))
    ratio_cases = 0
    for p in (2, 3, 5, 7):
        for z in (Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)):
            for w in (Fraction(-1, 4), Fraction(1, 4), Fraction(1)):
                zp = Fraction(p - 1) / (p - z)
                wp = Fraction(p - 1) / (p - w)
                for e in range(4):
                    a, d = z * w / p, (z - 1) * (w - 1)
                    expected = 1 + d / (p - 1) * sum(a**j for j in range(e)) - d * a**e / (p - 1)**2
                    assert local_fraction(p, e, z, w) / (zp * wp) == expected
                    ratio_cases += 1

    # Finite phase-map diagnostic; the grid maximum is NOT a contour supremum.
    phase_model = EulerModel(prime_table(10_000))
    total = int(totals[0])
    table = tables[0].astype(float)
    c11 = phase_model.coefficient_11(total)
    phase = []
    points = 64
    angles = np.exp(2j * np.pi * np.arange(points) / points)
    for radius in (0.125, 0.25, 0.5, 0.75):
        z = radius * angles[:, None]
        w = radius * angles[None, :]
        approximation = phase_model.value(total, z, w)
        powers = (radius * angles)[:, None] ** np.arange(size)[None, :]
        observed = powers @ table @ powers.T
        absolute = np.abs(observed - approximation)
        relative = absolute / np.abs(approximation)
        ia, ib = np.unravel_index(np.argmax(absolute), absolute.shape)
        ir, jr = np.unravel_index(np.argmax(relative), relative.shape)
        phase.append({
            "radius": radius,
            "grid_budget": float(absolute.max() / (radius**2 * c11)),
            "max_absolute_error_angles": [float(np.angle(angles[ia])), float(np.angle(angles[ib]))],
            "max_relative_error_angles": [float(np.angle(angles[ir])), float(np.angle(angles[jr]))],
            "absolute_error_positive": float(absolute[0, 0]),
            "absolute_error_negative": float(absolute[points // 2, points // 2]),
            "relative_error_positive": float(relative[0, 0]),
            "relative_error_negative": float(relative[points // 2, points // 2]),
        })

    source_checks = {}
    for name in ("q1q3_summary.json", "q1b_summary.json", "q2_summary.json"):
        raw = subprocess.check_output([
            "git", "show", SOURCE_COMMIT + ":research/Q1_fugacity/out/" + name,
        ])
        try:
            json.loads(raw, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
            strict = True
        except ValueError:
            strict = False
        source_checks[name] = {"strict_json": strict, "sha256": hashlib.sha256(raw).hexdigest()}

    result = {
        "schema_version": 1,
        "purpose": "Small retrospective audit; not evidence for an infinite conjecture",
        "hub_commit": HUB_COMMIT, "source_commit": SOURCE_COMMIT,
        "preregistered": False, "status": "PASS",
        "versions": {"numpy": np.__version__, "numba": nb.__version__, "scipy": scipy.__version__},
        "omega_sieve_limit": len(omega) - 1,
        "Q2d_reproduction": {"count": len(totals), "seed": 41,
                             "N_min": int(totals.min()), "N_max": int(totals.max()),
                             "results": reproduction},
        "single_variable_control": single,
        "exact_local_identity_cases": exact_local_cases,
        "exact_local_correlation_ratio_cases": ratio_cases,
        "model_zero_order_at_minus_one": {"3_divides_N": 2, "3_does_not_divide_N": 3},
        "phase_grid": {"N": total, "points_per_axis": points, "prime_cutoff": 10_000,
                       "observed_ordered_prime_pairs": int(table[1, 1]),
                       "model_coefficient_11": c11,
                       "prime_pair_count_over_model": float(table[1, 1]) / c11,
                       "budget_limit_at_zero": abs(float(table[1, 1]) / c11 - 1), "rows": phase},
        "source_json_checks": source_checks,
        "limits": [
            "No independent rerun of 1e8 or 1e9 experiments",
            "Corrected density product is a diagnostic heuristic, not a two-variable theorem",
            "Phase maxima are sampled grid values, not certified continuous suprema",
            "The zero orders refer to the untruncated analytic model, not the exact finite polynomial",
        ],
        "script_hash_scheme": "sha256-utf8-lf-v1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    }
    output = Path(__file__).with_name("partition_audit.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
