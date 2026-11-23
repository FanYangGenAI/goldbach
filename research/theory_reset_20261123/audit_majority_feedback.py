"""Small, retrospective audit of ed6c5f8; not a preregistered experiment.

Run: python research/theory_reset_20261123/audit_majority_feedback.py
Uses an independent NumPy sieve and counts; never imports the exploratory
script, whose top-level code allocates a sieve through 110 million.
Writes only majority_feedback_audit.json beside this file.
"""

from __future__ import annotations

import hashlib
import json
from math import isqrt, log
from pathlib import Path

import numpy as np


def integer_cuberoot(n: int) -> int:
    lo, hi = 0, n + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mid**3 <= n:
            lo = mid
        else:
            hi = mid
    return lo


def prime_table(limit: int) -> np.ndarray:
    prime = np.ones(limit + 1, dtype=np.bool_)
    prime[:2] = False
    for p in range(2, isqrt(limit) + 1):
        if prime[p]:
            prime[p * p :: p] = False
    return prime


def count_cases(totals: range, prime: np.ndarray) -> list[tuple[int, int, int, int]]:
    records = []
    previous_z = -1
    rough = np.zeros(len(prime), dtype=np.bool_)
    for total in totals:
        z = integer_cuberoot(total)
        if z != previous_z:
            rough[:] = True
            rough[:2] = False
            for p in np.flatnonzero(prime[: z + 1]):
                rough[:: int(p)] = False
            previous_z = z
        mask = rough[2 : total - 1] & rough[total - 2 : 1 : -1]
        left = prime[2 : total - 1]
        right = prime[total - 2 : 1 : -1]
        size = int(np.count_nonzero(mask))
        marginal = int(np.count_nonzero(mask & left))
        pp = int(np.count_nonzero(mask & left & right))
        # Count all four cells directly to check reflection and bookkeeping.
        ps = int(np.count_nonzero(mask & left & ~right))
        sp = int(np.count_nonzero(mask & ~left & right))
        ss = int(np.count_nonzero(mask & ~left & ~right))
        assert ps == sp
        assert size == pp + ps + sp + ss
        assert marginal == pp + ps
        assert pp - ss == 2 * marginal - size
        assert pp >= 2 * marginal - size
        assert size > 0
        records.append((total, size, marginal, pp))
    return records


def summarize(records: list[tuple[int, int, int, int]]) -> dict:
    counts = np.asarray(records, dtype=np.int64)
    totals, size, marginal, pp = counts.T
    share = marginal / size
    joint = pp / size
    covariance = joint - share**2
    return {
        "first_N": int(totals[0]),
        "last_N": int(totals[-1]),
        "step": 2,
        "count": len(records),
        "rho_mean": float(share.mean()),
        "rho_min": float(share.min()),
        "rho_min_first_N": int(totals[np.argmin(share)]),
        "rho_max": float(share.max()),
        "joint_min": float(joint.min()),
        "kappa_mean": float(covariance.mean()),
        "kappa_rms": float(np.sqrt(np.mean(covariance**2))),
        "kappa_min": float(covariance.min()),
        "kappa_max": float(covariance.max()),
        "rho_at_most_one_half_count": int(np.count_nonzero(2 * marginal <= size)),
        "counts_sha256": hashlib.sha256(counts.astype("<i8").tobytes()).hexdigest(),
    }


def main() -> None:
    prime = prime_table(110_000)
    windows = {
        "all_even_6_through_20000": range(6, 20_001, 2),
        "scale_1000": range(1_000, 1_100, 2),
        "scale_10000": range(10_000, 11_000, 2),
        "scale_100000": range(100_000, 110_000, 2),
    }
    result = {
        "schema_version": 1,
        "reviewed_commit": "ed6c5f825c418128b5c122a0dc47378c0861d164",
        "status": "PASS",
        "purpose": "Independent small-scale replication and explicit covariance audit",
        "preregistered": False,
        "arithmetic": "Exact integer counts; floating-point descriptive ratios only",
        "numpy_version": np.__version__,
        "heuristic_reference_only": 1 / (1 + log(2)),
        "windows": {key: summarize(count_cases(totals, prime)) for key, totals in windows.items()},
        "limitations": [
            "No reproduction of the reported random samples at scales 1e6 through 1e8",
            "No inference of an asymptotic limit, a uniform lower bound, or statistical independence",
            "This retrospective audit does not constitute the proposed preregistered Task Zero",
        ],
        "script_hash_scheme": "sha256-utf8-lf-v1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    }
    output = Path(__file__).with_name("majority_feedback_audit.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
