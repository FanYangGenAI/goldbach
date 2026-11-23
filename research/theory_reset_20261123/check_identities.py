"""Exact sanity checks for the theory-reset memo; no asymptotic inference.

Run from any directory:
    python research/theory_reset_20261123/check_identities.py

Only the JSON beside this script is written. All arithmetic assertions use
integers or fractions. These checks supplement the elementary proofs in the memo.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path


def sieve(limit: int) -> tuple[list[int], list[int], list[bool]]:
    spf = list(range(limit + 1))
    for p in range(2, int(limit**0.5) + 1):
        if spf[p] == p:
            for n in range(p * p, limit + 1, p):
                if spf[n] == n:
                    spf[n] = p
    liouville = [1] * (limit + 1)
    prime = [False] * (limit + 1)
    for n in range(2, limit + 1):
        prime[n] = spf[n] == n
        liouville[n] = -liouville[n // spf[n]]
    return spf, liouville, prime


def integer_cuberoot(n: int) -> int:
    lo, hi = 0, n + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mid**3 <= n:
            lo = mid
        else:
            hi = mid
    return lo


def affine_product(g, h):
    a, b = g
    c, d = h
    return a * c, a * d + b


def check_affine() -> int:
    count = 0
    for a in [Fraction(-3), Fraction(-1), Fraction(1, 2), Fraction(1), Fraction(2)]:
        for b in [Fraction(-2), Fraction(0), Fraction(1, 3), Fraction(5)]:
            dilation = a, Fraction(0)
            translation = Fraction(1), b
            inverse = 1 / a, Fraction(0)
            actual = affine_product(affine_product(dilation, translation), inverse)
            assert actual == (Fraction(1), a * b)
            count += 1
    return count


def check_boolean_reflection() -> int:
    """The signed-count and majority identities also hold for arbitrary labels."""
    count = 0
    for size in range(1, 9):
        for mask in range(1 << size):
            bits = [(mask >> i) & 1 for i in range(size)]
            signs = [1 - 2 * bit for bit in bits]
            pp = sum(bits[i] * bits[size - 1 - i] for i in range(size))
            ss = sum((1 - bits[i]) * (1 - bits[size - 1 - i]) for i in range(size))
            linear = sum(signs)
            quadratic = sum(signs[i] * signs[size - 1 - i] for i in range(size))
            assert 4 * pp == size - 2 * linear + quadratic
            assert pp - ss == -linear
            assert pp >= -linear
            rho = Fraction(sum(bits), size)
            covariance = Fraction(quadratic, 4 * size) - Fraction(linear, size) ** 2 / 4
            assert Fraction(pp, size) == rho**2 + covariance
            count += 1
    return count


def run_checks() -> dict:
    limit = 2000
    spf, liouville, prime = sieve(limit)
    arithmetic_cases = 0
    nonempty_cases = 0
    for total in range(4, limit + 1, 2):
        z = integer_cuberoot(total)
        survivors = [
            n for n in range(2, total - 1) if spf[n] > z and spf[total - n] > z
        ]
        for n in survivors:
            assert 2 * int(prime[n]) == 1 - liouville[n]
        size = len(survivors)
        linear = sum(liouville[n] for n in survivors)
        quadratic = sum(liouville[n] * liouville[total - n] for n in survivors)
        rough_pp = sum(prime[n] and prime[total - n] for n in survivors)
        rough_ss = sum(not prime[n] and not prime[total - n] for n in survivors)
        full_pp = sum(prime[n] and prime[total - n] for n in range(2, total - 1))
        assert 4 * rough_pp == size - 2 * linear + quadratic
        assert rough_pp - rough_ss == -linear
        assert rough_pp >= -linear
        boundary = full_pp - rough_pp
        assert 0 <= boundary <= 2 * sum(prime[: z + 1])
        if size:
            marginal = sum(prime[n] for n in survivors)
            rho = Fraction(marginal, size)
            covariance = Fraction(quadratic, 4 * size) - Fraction(linear, size) ** 2 / 4
            assert Fraction(rough_pp, size) == rho**2 + covariance
            nonempty_cases += 1
        arithmetic_cases += 1

    multiplicative_checks = 0
    for a in range(1, 41):
        for n in range(1, 41):
            assert liouville[a * n] == liouville[a] * liouville[n]
            assert a * (n + 1) == a * n + a
            multiplicative_checks += 1

    # The same maximally entangled reflection state, with a selected predicate
    # disjoint from its reflection, has zero joint projection probability.
    toy_total = 10
    predicate = {2, 3}
    toy_overlap = sum(n in predicate and toy_total - n in predicate for n in range(1, toy_total))
    assert toy_overlap == 0

    # An explicit counterfactual prime subset omitting one prescribed sum.
    total = 2000
    actual_primes = {n for n in range(2, total - 1) if prime[n]}
    unordered_pairs = [(p, total - p) for p in sorted(actual_primes) if p <= total - p and total - p in actual_primes]
    removed = {p for p, _ in unordered_pairs}
    modified = actual_primes - removed
    assert not any(total - p in modified for p in modified)

    return {
        "schema_version": 1,
        "purpose": "Exact finite checks of identities, not evidence for an infinite conjecture",
        "arithmetic": "integer and rational; no fitted constants",
        "status": "PASS",
        "affine_conjugacy_cases": check_affine(),
        "arbitrary_boolean_reflection_cases": check_boolean_reflection(),
        "rough_sieve_identities": {
            "even_totals_from": 4,
            "even_totals_through": limit,
            "cases": arithmetic_cases,
            "nonempty_survivor_sets": nonempty_cases,
            "cutoff": "floor(cuberoot(N)); both sides have all prime factors greater than cutoff",
            "verified": [
                "prime indicator equals (1-Liouville)/2 on survivors",
                "4*r_rough = T-2*L+C",
                "r_rough-s_rough = -L",
                "r_rough/T = rho^2 + covariance when T>0",
                "0 <= r_full-r_rough <= 2*pi(cutoff)",
            ],
        },
        "semigroup_and_parity_checks": multiplicative_checks,
        "entanglement_counterexample": {
            "N": toy_total,
            "state": "sum_{n=1}^{N-1}|n>|N-n>/sqrt(N-1)",
            "schmidt_rank": toy_total - 1,
            "entropy": "log(9)",
            "selected_predicate": sorted(predicate),
            "joint_projection_probability": "0",
            "meaning": "Maximal entanglement alone does not force overlap for specified predicates",
        },
        "counterfactual_subset": {
            "N": total,
            "original_prime_set_size": len(actual_primes),
            "deleted_endpoints": len(removed),
            "remaining_set_size": len(modified),
            "target_sum_representations": 0,
            "limitation": "A proper prime subset; not a counterexample for the full primes and not an exact preservation of arithmetic statistics",
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    result = run_checks()
    output = Path(__file__).with_name("identity_checks.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))
