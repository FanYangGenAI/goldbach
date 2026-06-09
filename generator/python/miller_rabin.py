"""Deterministic Miller-Rabin primality test for 64-bit integers."""

from __future__ import annotations

from constants import MR_BASES


def _decompose(n_minus_one: int) -> tuple[int, int]:
    d = n_minus_one
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    return d, s


def is_prime_deterministic(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    d, s = _decompose(n - 1)
    for base in MR_BASES:
        a = base % n
        if a == 0:
            continue

        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue

        composite = True
        for _ in range(1, s):
            x = pow(x, 2, n)
            if x == n - 1:
                composite = False
                break
        if composite:
            return False
    return True
