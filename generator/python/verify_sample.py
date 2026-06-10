"""Dataset sample audit helpers shared by tools/verify_dataset.py."""

from __future__ import annotations

import math
from typing import Protocol

from constants import CANDIDATE_PRIME_COUNT, VERSION, VERSION_V2
from format_io import GoldbachSample, GoldbachSampleV2
from labeling import compute_valid_mask_default, find_min_p1
from miller_rabin import is_prime_deterministic
from primes import PRIMES_UP_TO_MAX_P1, compute_modulo_features, first_k_odd_primes


class _HasCommonFields(Protocol):
    even_number: int
    modulo_features: tuple[int, ...]
    prime_density: float


def verify_common_fields(sample: _HasCommonFields) -> str | None:
    even_number = sample.even_number
    if even_number < 4 or even_number % 2 != 0:
        return f"invalid even_number={even_number}"
    if tuple(sample.modulo_features) != compute_modulo_features(even_number):
        return "modulo_features mismatch"
    expected_density = 1.0 / math.log(even_number)
    if abs(sample.prime_density - expected_density) > 1e-5:
        return "prime_density mismatch"
    return None


def verify_v1_label(even_number: int, p1_label: int) -> str | None:
    if p1_label > 5000:
        return f"p1 out of range: {p1_label}"
    p2 = even_number - p1_label
    if p1_label + p2 != even_number:
        return "sum mismatch"
    if not is_prime_deterministic(p2):
        return f"p2 not prime: {p2}"
    for prime in PRIMES_UP_TO_MAX_P1:
        if prime >= p1_label:
            break
        p2_candidate = even_number - prime
        if p2_candidate >= prime and is_prime_deterministic(p2_candidate):
            return f"smaller p1 exists: {prime}"
    recomputed = find_min_p1(even_number)
    if recomputed != p1_label:
        return f"label mismatch: stored={p1_label}, recomputed={recomputed}"
    return None


def verify_v1_sample(sample: GoldbachSample) -> str | None:
    error = verify_common_fields(sample)
    if error is not None:
        return error
    return verify_v1_label(sample.even_number, sample.p1_label)


def verify_v2_valid_mask(
    even_number: int,
    stored_mask: tuple[int, ...],
    candidate_count: int = CANDIDATE_PRIME_COUNT,
) -> str | None:
    if len(stored_mask) != candidate_count:
        return f"valid_mask length mismatch: {len(stored_mask)} != {candidate_count}"
    if any(bit not in (0, 1) for bit in stored_mask):
        return "valid_mask contains non-binary values"
    expected = compute_valid_mask_default(even_number, candidate_count)
    if stored_mask != expected:
        return "valid_mask mismatch with recomputed mask"
    return None


def verify_v2_p1_consistency(
    even_number: int,
    stored_mask: tuple[int, ...],
    candidate_count: int = CANDIDATE_PRIME_COUNT,
) -> str | None:
    candidates = first_k_odd_primes(candidate_count)
    try:
        p1 = find_min_p1(even_number)
    except RuntimeError:
        return None
    if p1 not in candidates:
        return None
    index = candidates.index(p1)
    if stored_mask[index] != 1:
        return f"p1={p1} in vocabulary but valid_mask[{index}]==0"
    return None


def verify_v2_sample(
    sample: GoldbachSampleV2,
    candidate_count: int = CANDIDATE_PRIME_COUNT,
) -> str | None:
    error = verify_common_fields(sample)
    if error is not None:
        return error
    error = verify_v2_valid_mask(sample.even_number, sample.valid_mask, candidate_count)
    if error is not None:
        return error
    return verify_v2_p1_consistency(sample.even_number, sample.valid_mask, candidate_count)


def dataset_version_label(version: int) -> str:
    if version == VERSION:
        return "v1"
    if version == VERSION_V2:
        return "v2"
    raise ValueError(f"Unsupported dataset version: {version}")
