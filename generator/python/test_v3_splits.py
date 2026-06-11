#!/usr/bin/env python3
"""Unit tests for v3 split planning."""

from __future__ import annotations

import unittest

from constants import V3_TOTAL
from format_io import build_header, verify_header
from v3_splits import (
    assign_split_for_even,
    build_v3_bucket_targets,
    collect_v3_split_evens,
    compute_dedup_hash,
)


class TestV3Splits(unittest.TestCase):
    def test_bucket_targets_sum_to_total(self) -> None:
        targets = build_v3_bucket_targets(V3_TOTAL)
        self.assertEqual(sum(targets.values()), V3_TOTAL)

    def test_assign_split_is_stable(self) -> None:
        first = assign_split_for_even(123_456)
        second = assign_split_for_even(123_456)
        self.assertEqual(first, second)

    def test_smoke_collection_unique_and_disjoint(self) -> None:
        split_evens = collect_v3_split_evens(total_samples=1_000, seed=11)
        all_evens: list[int] = []
        for values in split_evens.values():
            self.assertEqual(len(values), len(set(values)))
            all_evens.extend(values)
        self.assertEqual(len(all_evens), len(set(all_evens)))
        self.assertEqual(len(all_evens), 1_000)

    def test_v3_header_roundtrip(self) -> None:
        header = build_header(10, version=3, split_id=1, dedup_hash=42)
        parsed = verify_header(header)
        self.assertEqual(parsed.version, 3)
        self.assertEqual(parsed.split_id, 1)
        self.assertEqual(parsed.dedup_hash, 42)

    def test_dedup_hash_stable(self) -> None:
        values = [6, 8, 10]
        self.assertEqual(compute_dedup_hash(values), compute_dedup_hash(list(reversed(values))))


if __name__ == "__main__":
    unittest.main()
