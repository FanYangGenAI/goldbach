"""Goldbach label search."""

from __future__ import annotations

from collections.abc import Sequence

from constants import CANDIDATE_PRIME_COUNT
from miller_rabin import is_prime_deterministic
from primes import PRIMES_UP_TO_MAX_P1, first_k_odd_primes


def _validate_even_number(even_number: int) -> None:
    if even_number < 4 or even_number % 2 != 0:
        raise ValueError(f"even_number must be an even integer >= 4, got {even_number}")


def compute_valid_mask(
    even_number: int,
    candidate_primes: Sequence[int],
) -> tuple[int, ...]:
    """Return multi-hot mask: 1 if N - p is prime, else 0, for each candidate p."""
    _validate_even_number(even_number)
    if not candidate_primes:
        raise ValueError("candidate_primes must be non-empty")

    mask: list[int] = []
    for prime in candidate_primes:
        offset = even_number - prime
        mask.append(1 if offset >= 2 and is_prime_deterministic(offset) else 0)
    return tuple(mask)


def compute_valid_mask_default(
    even_number: int,
    k: int = CANDIDATE_PRIME_COUNT,
) -> tuple[int, ...]:
    """Compute valid_mask using the first k odd primes as candidates."""
    return compute_valid_mask(even_number, first_k_odd_primes(k))


def find_min_p1(even_number: int) -> int:
    _validate_even_number(even_number)

    for p1 in PRIMES_UP_TO_MAX_P1:
        p2 = even_number - p1
        if p2 < p1:
            break
        if is_prime_deterministic(p2):
            return p1
    raise RuntimeError(f"No Goldbach pair found for even_number={even_number}")
