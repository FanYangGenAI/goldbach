#!/usr/bin/env python3
"""Unit tests for v2 dataset serialization."""

from __future__ import annotations

import math
import unittest

from constants import CANDIDATE_PRIME_COUNT, SAMPLE_SIZE_V2, VERSION_V2
from format_io import (
    GoldbachSampleV2,
    build_header,
    decode_valid_mask,
    encode_valid_mask,
    make_sample_v2,
    sample_size_for_version,
    verify_header,
)
from labeling import compute_valid_mask


class TestValidMaskBitset(unittest.TestCase):
    def test_encode_decode_round_trip(self) -> None:
        mask = compute_valid_mask(100, (3, 5, 7, 11, 13))
        padded = mask + (0,) * (CANDIDATE_PRIME_COUNT - len(mask))
        encoded = encode_valid_mask(padded)
        self.assertEqual(len(encoded), 63)
        self.assertEqual(decode_valid_mask(encoded), padded)

    def test_lsb_is_candidate_zero(self) -> None:
        mask = (1,) + (0,) * (CANDIDATE_PRIME_COUNT - 1)
        encoded = encode_valid_mask(mask)
        self.assertEqual(encoded[0], 1)

    def test_high_bit_in_last_used_byte(self) -> None:
        mask = [0] * CANDIDATE_PRIME_COUNT
        mask[499] = 1
        encoded = encode_valid_mask(mask)
        self.assertEqual(encoded[62], 1 << 3)


class TestGoldbachSampleV2(unittest.TestCase):
    def test_to_bytes_length(self) -> None:
        sample = make_sample_v2(100)
        self.assertEqual(len(sample.to_bytes()), SAMPLE_SIZE_V2)
        self.assertEqual(len(sample.valid_mask), CANDIDATE_PRIME_COUNT)

    def test_from_bytes_round_trip(self) -> None:
        original = make_sample_v2(1_234_567_890)
        restored = GoldbachSampleV2.from_bytes(original.to_bytes())
        self.assertEqual(restored.even_number, original.even_number)
        self.assertEqual(restored.modulo_features, original.modulo_features)
        self.assertEqual(restored.valid_mask, original.valid_mask)
        self.assertAlmostEqual(restored.prime_density, original.prime_density, places=6)

    def test_fields_match_expectations(self) -> None:
        even_number = 100
        sample = make_sample_v2(even_number)
        self.assertAlmostEqual(sample.prime_density, 1.0 / math.log(even_number))
        self.assertEqual(sample.valid_mask[0], 1)  # 100 - 3 = 97


class TestHeaderV2(unittest.TestCase):
    def test_build_and_verify_v2(self) -> None:
        header = build_header(sample_count=42, version=VERSION_V2)
        parsed = verify_header(header)
        self.assertEqual(parsed.version, VERSION_V2)
        self.assertEqual(parsed.sample_count, 42)
        self.assertEqual(sample_size_for_version(parsed.version), SAMPLE_SIZE_V2)


if __name__ == "__main__":
    unittest.main()
