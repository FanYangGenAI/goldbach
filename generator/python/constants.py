"""Shared constants for Goldbach dataset generation."""

from __future__ import annotations

MAGIC = 0x47424B48  # "GBKH"
VERSION = 1
VERSION_V2 = 2
HEADER_SIZE = 64
SAMPLE_SIZE = 26
VALID_MASK_BYTES = 63  # ceil(500 / 8)
SAMPLE_SIZE_V2 = 8 + 10 + 4 + VALID_MASK_BYTES  # 85 bytes

SMALL_PRIMES = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
MR_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
MAX_P1 = 5000

# IGOT v2: default candidate vocabulary size (first K odd primes starting at 3).
CANDIDATE_PRIME_COUNT = 500

# Training/eval domain: exclude N=4 (Goldbach pair 2+2 needs p=2, not in odd-prime vocabulary).
MIN_TRAINING_EVEN = 6

# IGOT explicit sieve: number of modulus bases r_j for sieve features.
MODULUS_BASE_COUNT = 32

# Full production bucket boundaries (2n value ranges).
BUCKET_BOUNDS: list[tuple[int, int, int]] = [
    (1, 4, 10**2),
    (2, 10**2, 10**3),
    (3, 10**3, 10**4),
    (4, 10**4, 10**5),
    (5, 10**5, 10**6),
    (6, 10**6, 10**7),
    (7, 10**7, 10**8),
    (8, 10**8, 10**9),
    (9, 10**9, 10**10),
    (10, 10**10, 10**11),
    (11, 10**11, 10**12),
    (12, 10**12, 10**13),
    (13, 10**13, 10**14),
    (14, 10**14, 10**15),
    (15, 10**15, 10**16),
    (16, 10**16, 10**17),
    (17, 10**17, 10**18),
    (18, 10**18, 4 * 10**18 + 1),
]

# Phase 0: 100k samples with cross-magnitude coverage.
PHASE0_TOTAL = 100_000
PHASE0_BUCKET_TARGETS: dict[int, int] = {
    1: 48,
    2: 450,
    3: 4_500,
    4: 6_334,
    5: 6_334,
    6: 6_334,
    7: 6_334,
    8: 6_334,
    9: 6_334,
    10: 6_334,
    11: 6_333,
    12: 6_333,
    13: 6_333,
    14: 6_333,
    15: 6_333,
    16: 6_333,
    17: 6_333,
    18: 6_333,
}

# Phase 1: full enumeration for buckets 1-6 (~5M samples).
PHASE1_BUCKET_IDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
PHASE1_BUCKET_TARGETS: dict[int, int] = {
    1: 48,
    2: 450,
    3: 4_500,
    4: 45_000,
    5: 450_000,
    6: 4_500_000,
}
PHASE1_TOTAL = sum(PHASE1_BUCKET_TARGETS.values())

# Phase 2: full production dataset (100M samples).
PHASE2_TOTAL = 100_000_000
PHASE2_HARD_CASE_COUNT = 100_000
PHASE2_BUCKET_TARGETS: dict[int, int] = {
    1: 48,
    2: 450,
    3: 4_500,
    4: 45_000,
    5: 450_000,
    6: 4_500_000,
    7: 7_916_667,
    8: 7_916_667,
    9: 7_916_667,
    10: 7_916_667,
    11: 7_916_667,
    12: 7_916_667,
    13: 7_916_667,
    14: 7_916_667,
    15: 7_916_667,
    16: 7_916_666,
    17: 7_916_666,
    18: 7_916_667,
}
assert sum(PHASE2_BUCKET_TARGETS.values()) == PHASE2_TOTAL
