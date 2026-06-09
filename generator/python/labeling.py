"""Goldbach label search."""

from __future__ import annotations

from miller_rabin import is_prime_deterministic
from primes import PRIMES_UP_TO_MAX_P1


def find_min_p1(even_number: int) -> int:
    if even_number < 4 or even_number % 2 != 0:
        raise ValueError(f"even_number must be an even integer >= 4, got {even_number}")

    for p1 in PRIMES_UP_TO_MAX_P1:
        p2 = even_number - p1
        if p2 < p1:
            break
        if is_prime_deterministic(p2):
            return p1
    raise RuntimeError(f"No Goldbach pair found for even_number={even_number}")
