#!/usr/bin/env python3
"""Unit tests for IGOT token builders."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_MODEL_ROOT = Path(__file__).resolve().parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from igot import IGOTModel  # noqa: E402
from sieve_baseline import (  # noqa: E402
    build_sieve_matrix,
    compute_eligibility_mask,
    compute_sieve_matrix,
)
from sieve_residual import SieveResidualModel  # noqa: E402
from tokens import ModulusTokenBuilder, compute_elimination_stats  # noqa: E402


class TestEliminationStats(unittest.TestCase):
    def test_eligible_only_at_n100(self) -> None:
        model = IGOTModel(candidate_count=500, modulus_count=32, d_model=32, num_heads=2)
        even = torch.tensor([100], dtype=torch.int64)
        z = build_sieve_matrix(even, model.sieve_module.candidate_primes, model.sieve_module.modulus_bases)
        eligible = compute_eligibility_mask(even, model.sieve_module.candidate_primes)

        eliminated, rate = compute_elimination_stats(z, eligible)
        manual_z = z * eligible.unsqueeze(-1).to(z.dtype)
        manual_eliminated = manual_z.sum(dim=1)
        eligible_count = eligible.sum(dim=1).clamp_min(1).float()
        manual_rate = manual_eliminated / eligible_count.unsqueeze(-1)

        torch.testing.assert_close(eliminated, manual_eliminated)
        torch.testing.assert_close(rate, manual_rate)

        raw = compute_sieve_matrix(even, model.sieve_module.candidate_primes, model.sieve_module.modulus_bases)
        raw_eliminated = raw.sum(dim=1)
        self.assertFalse(torch.allclose(eliminated, raw_eliminated))


class TestDownstreamSieveMatrix(unittest.TestCase):
    def test_phase2_uses_bypassed_sieve_matrix(self) -> None:
        model = SieveResidualModel(candidate_count=16, modulus_count=8, d_model=32, hidden_dim=16)
        even = torch.tensor([10], dtype=torch.int64)
        bits = torch.rand(1, 64)
        mods = torch.zeros(1, 10, dtype=torch.int64)
        density = torch.tensor([0.2], dtype=torch.float32)

        outputs = model(even, bits, mods, density)
        expected = build_sieve_matrix(
            even,
            model.sieve_module.candidate_primes,
            model.sieve_module.modulus_bases,
        )
        torch.testing.assert_close(outputs["sieve_matrix"], expected)

    def test_phase3_uses_bypassed_sieve_matrix(self) -> None:
        model = IGOTModel(candidate_count=16, modulus_count=8, d_model=32, num_heads=2)
        even = torch.tensor([10], dtype=torch.int64)
        bits = torch.rand(1, 64)
        mods = torch.zeros(1, 10, dtype=torch.int64)
        density = torch.tensor([0.2], dtype=torch.float32)

        outputs = model(even, bits, mods, density)
        expected = build_sieve_matrix(
            even,
            model.sieve_module.candidate_primes,
            model.sieve_module.modulus_bases,
        )
        torch.testing.assert_close(outputs["sieve_matrix"], expected)


if __name__ == "__main__":
    unittest.main()
