"""Prime table utilities."""

from __future__ import annotations

from constants import MAX_P1, MODULUS_BASE_COUNT, SMALL_PRIMES


def _sieve(limit: int) -> list[int]:
    if limit < 2:
        return []
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0] = 0
    if limit >= 1:
        is_prime[1] = 0
    for number in range(2, int(limit**0.5) + 1):
        if is_prime[number]:
            step = number
            start = number * number
            is_prime[start : limit + 1 : step] = b"\x00" * len(is_prime[start : limit + 1 : step])
    return [index for index in range(2, limit + 1) if is_prime[index]]


PRIMES_UP_TO_MAX_P1: tuple[int, ...] = tuple(_sieve(MAX_P1))
PRIME_SET_UP_TO_MAX_P1: frozenset[int] = frozenset(PRIMES_UP_TO_MAX_P1)
# Odd primes only (excludes 2); used as IGOT candidate vocabulary.
ODD_PRIMES_UP_TO_MAX_P1: tuple[int, ...] = PRIMES_UP_TO_MAX_P1[1:]


def is_small_prime(value: int) -> bool:
    return value in PRIME_SET_UP_TO_MAX_P1


def first_m_modulus_bases(m: int = MODULUS_BASE_COUNT) -> tuple[int, ...]:
    """Return the first m odd primes used as sieve modulus bases."""
    return first_k_odd_primes(m)


def first_k_odd_primes(k: int) -> tuple[int, ...]:
    """Return the first k odd primes (3, 5, 7, ...)."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > len(ODD_PRIMES_UP_TO_MAX_P1):
        raise ValueError(
            f"k={k} exceeds available odd primes ({len(ODD_PRIMES_UP_TO_MAX_P1)})"
        )
    return ODD_PRIMES_UP_TO_MAX_P1[:k]


def compute_modulo_features(even_number: int) -> tuple[int, ...]:
    return tuple(even_number % prime for prime in SMALL_PRIMES)
