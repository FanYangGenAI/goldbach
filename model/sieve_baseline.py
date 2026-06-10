"""Explicit sieve baseline for IGOT Phase 1."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_GENERATOR_PYTHON = Path(__file__).resolve().parents[1] / "generator" / "python"
if str(_GENERATOR_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GENERATOR_PYTHON))

from constants import CANDIDATE_PRIME_COUNT, MODULUS_BASE_COUNT  # noqa: E402
from primes import first_k_odd_primes, first_m_modulus_bases  # noqa: E402


def compute_sieve_matrix(
    even_numbers: torch.Tensor,
    candidate_primes: torch.Tensor,
    modulus_bases: torch.Tensor,
) -> torch.Tensor:
    """Compute z[b, i, j] = 1 iff (N - p_i) % r_j == 0."""
    offsets = even_numbers.unsqueeze(1) - candidate_primes.unsqueeze(0)
    remainders = offsets.unsqueeze(-1) % modulus_bases.view(1, 1, -1)
    return (remainders == 0).to(dtype=torch.float32)


class ExplicitSieveModule(nn.Module):
    """logit_i = b_i + sum_j w_j * z[i, j] with no hidden layers."""

    def __init__(
        self,
        candidate_count: int = CANDIDATE_PRIME_COUNT,
        modulus_count: int = MODULUS_BASE_COUNT,
    ) -> None:
        super().__init__()
        self.candidate_count = candidate_count
        self.modulus_count = modulus_count

        candidates = torch.tensor(first_k_odd_primes(candidate_count), dtype=torch.int64)
        moduli = torch.tensor(first_m_modulus_bases(modulus_count), dtype=torch.int64)
        self.register_buffer("candidate_primes", candidates, persistent=False)
        self.register_buffer("modulus_bases", moduli, persistent=False)

        self.candidate_bias = nn.Parameter(torch.zeros(candidate_count))
        self.modulus_weights = nn.Parameter(torch.zeros(modulus_count))

    def forward(self, even_numbers: torch.Tensor) -> dict[str, torch.Tensor]:
        sieve_matrix = compute_sieve_matrix(
            even_numbers,
            self.candidate_primes,
            self.modulus_bases,
        )
        sieve_logits = self.candidate_bias + torch.einsum(
            "bkm,m->bk",
            sieve_matrix,
            self.modulus_weights,
        )
        return {
            "sieve_logits": sieve_logits,
            "sieve_matrix": sieve_matrix,
        }

    def modulus_weight_table(self) -> list[tuple[int, float]]:
        values = self.modulus_weights.detach().cpu().tolist()
        bases = self.modulus_bases.detach().cpu().tolist()
        return list(zip(bases, values))
