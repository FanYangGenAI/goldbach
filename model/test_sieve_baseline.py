#!/usr/bin/env python3
"""Unit tests for explicit sieve baseline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_MODEL_ROOT = Path(__file__).resolve().parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from sieve_baseline import (  # noqa: E402
    INELIGIBLE_LOGIT,
    ExplicitSieveModule,
    compute_eligibility_mask,
    compute_sieve_matrix,
)


class TestExplicitSieve(unittest.TestCase):
    def test_sieve_matrix_shape(self) -> None:
        even = torch.tensor([100, 204], dtype=torch.int64)
        candidates = torch.tensor([3, 5, 7, 11], dtype=torch.int64)
        moduli = torch.tensor([3, 5], dtype=torch.int64)
        matrix = compute_sieve_matrix(even, candidates, moduli)
        self.assertEqual(tuple(matrix.shape), (2, 4, 2))
        self.assertEqual(matrix[0, 1, 1].item(), 1.0)  # 100 - 5 = 95, 95 % 5 == 0

    def test_forward_shape(self) -> None:
        model = ExplicitSieveModule(candidate_count=16, modulus_count=8)
        even = torch.tensor([100, 204], dtype=torch.int64)
        outputs = model(even)
        self.assertEqual(tuple(outputs["sieve_logits"].shape), (2, 16))

    def test_eligibility_masks_small_n(self) -> None:
        model = ExplicitSieveModule(candidate_count=500, modulus_count=32)
        for even_value in (6, 8, 10, 100):
            even = torch.tensor([even_value], dtype=torch.int64)
            outputs = model(even)
            eligible = compute_eligibility_mask(even, model.candidate_primes)[0]
            logits = outputs["sieve_logits"][0]
            for index, is_eligible in enumerate(eligible.tolist()):
                if is_eligible:
                    self.assertGreater(logits[index].item(), INELIGIBLE_LOGIT / 10)
                else:
                    self.assertAlmostEqual(logits[index].item(), INELIGIBLE_LOGIT, places=1)

    def test_sieve_matrix_offsets_for_small_n(self) -> None:
        model = ExplicitSieveModule(candidate_count=500, modulus_count=32)
        even_values = torch.tensor([6, 8, 10, 100], dtype=torch.int64)
        matrix = compute_sieve_matrix(
            even_values,
            model.candidate_primes,
            model.modulus_bases,
        )
        offsets = even_values.unsqueeze(1) - model.candidate_primes.unsqueeze(0)
        # Negative offsets must not be treated as eligible by the model forward path.
        outputs = model(even_values)
        for row, n_value in enumerate(even_values.tolist()):
            ineligible = offsets[row] < 3
            if ineligible.any():
                self.assertTrue(torch.all(outputs["sieve_logits"][row, ineligible] <= INELIGIBLE_LOGIT / 10))
            self.assertEqual(matrix[row].shape[0], model.candidate_count)
            self.assertEqual(matrix[row].shape[1], model.modulus_count)

    def test_backward_updates_parameters(self) -> None:
        model = ExplicitSieveModule(candidate_count=8, modulus_count=4)
        even = torch.tensor([100, 204], dtype=torch.int64)
        target = torch.zeros(2, 8)
        target[0, 0] = 1.0
        target[1, 1] = 1.0
        logits = model(even)["sieve_logits"]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        self.assertIsNotNone(model.modulus_weights.grad)
        self.assertIsNotNone(model.candidate_bias.grad)


if __name__ == "__main__":
    unittest.main()
