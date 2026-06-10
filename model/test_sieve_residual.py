#!/usr/bin/env python3
"""Unit tests for Phase 2 sieve + residual model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_MODEL_ROOT = Path(__file__).resolve().parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from sieve_residual import SieveResidualModel  # noqa: E402


class TestSieveResidual(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        model = SieveResidualModel(candidate_count=16, modulus_count=8, d_model=32)
        even = torch.tensor([100, 204], dtype=torch.int64)
        bits = torch.rand(2, 64)
        mods = torch.tensor(
            [
                [1, 0, 2, 6, 9, 0, 14, 20, 1, 18],
                [0, 4, 1, 6, 9, 0, 14, 20, 1, 18],
            ],
            dtype=torch.int64,
        )
        density = torch.tensor([0.2, 0.1], dtype=torch.float32)
        outputs = model(even, bits, mods, density)
        self.assertEqual(tuple(outputs["final_logits"].shape), (2, 16))
        self.assertEqual(tuple(outputs["residual_logits"].shape), (2, 16))
        self.assertTrue(torch.allclose(
            outputs["final_logits"],
            outputs["sieve_logits"] + outputs["residual_logits"],
        ))

    def test_backward_flows(self) -> None:
        model = SieveResidualModel(candidate_count=8, modulus_count=4, d_model=16)
        even = torch.tensor([100], dtype=torch.int64)
        bits = torch.rand(1, 64)
        mods = torch.tensor([[1, 0, 2, 6, 9, 0, 14, 20, 1, 18]], dtype=torch.int64)
        density = torch.tensor([0.2], dtype=torch.float32)
        target = torch.zeros(1, 8)
        target[0, 0] = 1.0
        outputs = model(even, bits, mods, density)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["final_logits"],
            target,
        )
        loss.backward()
        self.assertIsNotNone(model.residual_mlp[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
