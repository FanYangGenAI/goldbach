#!/usr/bin/env python3
"""Unit tests for coverage statistics."""

from __future__ import annotations

import unittest

from coverage import (
    analyze_even,
    bucket_id_for_even,
    build_coverage_report,
    select_uncovered_examples,
)
from primes import first_k_odd_primes


class TestCoverage(unittest.TestCase):
    def test_bucket_id_for_small_even(self) -> None:
        self.assertEqual(bucket_id_for_even(100), 2)

    def test_analyze_even_n100(self) -> None:
        row = analyze_even(100, first_k_odd_primes(5))
        self.assertTrue(row.covered)
        self.assertGreater(row.valid_count, 0)
        self.assertEqual(row.min_valid_rank, 0)

    def test_build_report_structure(self) -> None:
        rows = [analyze_even(100, first_k_odd_primes(20))]
        report = build_coverage_report(
            rows,
            k=20,
            seed=7,
            mode="random",
            source="test",
        )
        self.assertEqual(report["global"]["coverage_at_k"], 1.0)
        self.assertEqual(report["buckets"][0]["id"], 2)

    def test_select_uncovered_examples(self) -> None:
        rows = [
            analyze_even(100, first_k_odd_primes(5)),
            analyze_even(100, (997,)),  # force uncovered with tiny vocab
        ]
        examples = select_uncovered_examples(rows, limit=5)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["even_number"], 100)


if __name__ == "__main__":
    unittest.main()
