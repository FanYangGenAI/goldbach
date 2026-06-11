#!/usr/bin/env python3
"""Unit tests for Goldbach labeling helpers."""

from __future__ import annotations

import unittest

from labeling import compute_valid_mask, compute_valid_mask_default, find_min_p1
from primes import first_k_odd_primes, first_m_modulus_bases


class TestComputeValidMask(unittest.TestCase):
    def test_n100_first_five_candidates(self) -> None:
        candidates = (3, 5, 7, 11, 13)
        mask = compute_valid_mask(100, candidates)
        self.assertEqual(mask, (1, 0, 0, 1, 0))

    def test_n4_smallest_even(self) -> None:
        mask = compute_valid_mask(4, (2,))
        self.assertEqual(mask, (1,))

    def test_offset_below_two_is_zero(self) -> None:
        mask = compute_valid_mask(10, (3, 5, 7, 9))
        self.assertEqual(mask[3], 0)

    def test_default_k500_length(self) -> None:
        mask = compute_valid_mask_default(100)
        self.assertEqual(len(mask), 500)

    def test_min_p1_implies_mask_bit_set(self) -> None:
        even_number = 1_234_567_890
        candidates = first_k_odd_primes(500)
        mask = compute_valid_mask(even_number, candidates)
        p1 = find_min_p1(even_number)
        if p1 in candidates:
            index = candidates.index(p1)
            self.assertEqual(mask[index], 1)

    def test_rejects_invalid_even_number(self) -> None:
        with self.assertRaises(ValueError):
            compute_valid_mask(99, (3, 5))

    def test_rejects_empty_candidates(self) -> None:
        with self.assertRaises(ValueError):
            compute_valid_mask(100, ())

    def test_modulus_base_offsets_are_valid(self) -> None:
        candidates = first_k_odd_primes(500)
        moduli = first_m_modulus_bases(32)
        even_number = 100
        mask = compute_valid_mask(even_number, candidates)
        for modulus in moduli:
            prime = even_number - modulus
            if prime < 3:
                continue
            index = candidates.index(prime) if prime in candidates else None
            if index is not None:
                self.assertEqual(mask[index], 1, msg=f"N={even_number}, m={modulus}")


if __name__ == "__main__":
    unittest.main()
