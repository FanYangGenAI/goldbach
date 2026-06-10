#!/usr/bin/env python3
"""Unit tests for dataset sample verification."""

from __future__ import annotations

import unittest

from format_io import GoldbachSampleV2, make_sample, make_sample_v2
from verify_sample import verify_v1_sample, verify_v2_sample


class TestVerifySample(unittest.TestCase):
    def test_verify_v1_sample_passes(self) -> None:
        sample = make_sample(100, 3)
        self.assertIsNone(verify_v1_sample(sample))

    def test_verify_v2_sample_passes(self) -> None:
        sample = make_sample_v2(100)
        self.assertIsNone(verify_v2_sample(sample))

    def test_verify_v2_detects_corrupt_mask(self) -> None:
        sample = make_sample_v2(100)
        flip_index = sample.valid_mask.index(1)
        corrupted = list(sample.valid_mask)
        corrupted[flip_index] = 0
        bad = GoldbachSampleV2(
            even_number=sample.even_number,
            modulo_features=sample.modulo_features,
            prime_density=sample.prime_density,
            valid_mask=corrupted,
        )
        self.assertEqual(verify_v2_sample(bad), "valid_mask mismatch with recomputed mask")


if __name__ == "__main__":
    unittest.main()
