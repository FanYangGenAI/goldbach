"""Explicit sieve baseline for IGOT Phase 1.

Phase 1 显式筛法基线模块。
Implements: logit_i = b_i + sum_j(w_j * z[i,j]) with no hidden layers.
实现公式：logit_i = b_i + Σ_j(w_j * z[i,j])，无可学习隐藏层。
"""

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

# Candidates with m = N - p_i < 3 are invalid (p_i > N - 3); mask in loss and ranking.
INELIGIBLE_LOGIT = -1e4
MIN_CANDIDATE_OFFSET = 3


def compute_eligibility_mask(
    even_numbers: torch.Tensor,
    candidate_primes: torch.Tensor,
) -> torch.Tensor:
    """Return [B, K] bool mask: True when m = N - p_i >= MIN_CANDIDATE_OFFSET."""
    offsets = even_numbers.unsqueeze(1) - candidate_primes.unsqueeze(0)
    return offsets >= MIN_CANDIDATE_OFFSET


def apply_eligibility_mask(
    logits: torch.Tensor,
    even_numbers: torch.Tensor,
    candidate_primes: torch.Tensor,
) -> torch.Tensor:
    """Force ineligible candidate logits to INELIGIBLE_LOGIT."""
    eligible = compute_eligibility_mask(even_numbers, candidate_primes)
    return torch.where(eligible, logits, torch.full_like(logits, INELIGIBLE_LOGIT))


def compute_sieve_matrix(
    even_numbers: torch.Tensor,
    candidate_primes: torch.Tensor,
    modulus_bases: torch.Tensor,
) -> torch.Tensor:
    """Compute sieve indicators z[b, i, j].

    计算筛法指示矩阵 z[b, i, j]。

    z[b, i, j] = 1 iff (N - p_i) is divisible by modulus r_j.
    当且仅当偏移量 (N - p_i) 能被模数 r_j 整除时，z[b, i, j] = 1。

    Shapes / 张量形状:
        even_numbers:     [B]
        candidate_primes: [K]
        modulus_bases:    [M]
        return:           [B, K, M]
    """
    # offsets[b, i] = N - p_i
    # 偏移量 q = N - p_i，即候选素数对的另一半
    offsets = even_numbers.unsqueeze(1) - candidate_primes.unsqueeze(0)

    # remainders[b, i, j] = (N - p_i) mod r_j
    # 每个偏移量对每个模数的余数
    remainders = offsets.unsqueeze(-1) % modulus_bases.view(1, 1, -1)

    # Binary flag: 1 means "divisible by r_j" -> likely composite, penalize later.
    # 二元标记：1 表示被 r_j 整除，通常意味着合数，后续会被惩罚
    return (remainders == 0).to(dtype=torch.float32)


def apply_small_prime_sieve_bypass(
    sieve_matrix: torch.Tensor,
    even_numbers: torch.Tensor,
    candidate_primes: torch.Tensor,
    modulus_bases: torch.Tensor,
) -> torch.Tensor:
    """Zero sieve penalties when m = N - p_i is a modulus base (small prime).

    When m equals r_j in the sieve basis, m is prime and must not be penalized
    for being divisible by r_j.
    """
    offsets = even_numbers.unsqueeze(1) - candidate_primes.unsqueeze(0)
    is_small_prime_offset = torch.zeros_like(offsets, dtype=torch.bool)
    for modulus in modulus_bases.tolist():
        is_small_prime_offset |= offsets == int(modulus)
    keep = (~is_small_prime_offset).unsqueeze(-1).to(dtype=sieve_matrix.dtype)
    return sieve_matrix * keep


class ExplicitSieveModule(nn.Module):
    """Interpretable linear sieve: logit_i = b_i + sum_j w_j * z[i, j].

    可解释的线性筛法模块。
    """

    def __init__(
        self,
        candidate_count: int = CANDIDATE_PRIME_COUNT,
        modulus_count: int = MODULUS_BASE_COUNT,
    ) -> None:
        super().__init__()
        self.candidate_count = candidate_count
        self.modulus_count = modulus_count

        # Fixed vocabulary: first K odd primes {3, 5, 7, ...}.
        # 固定候选词表：前 K 个奇素数
        candidates = torch.tensor(first_k_odd_primes(candidate_count), dtype=torch.int64)
        # Fixed sieve moduli: first M odd primes used as divisibility bases.
        # 固定筛法模数基底：前 M 个奇素数
        moduli = torch.tensor(first_m_modulus_bases(modulus_count), dtype=torch.int64)
        self.register_buffer("candidate_primes", candidates, persistent=False)
        self.register_buffer("modulus_bases", moduli, persistent=False)

        # Learnable prior bias per candidate p_i.
        # 每个候选 p_i 的可学习先验偏置 b_i
        self.candidate_bias = nn.Parameter(torch.zeros(candidate_count))
        # Learnable penalty weight per modulus r_j (expected negative after training).
        # 每个模数 r_j 的可学习惩罚权重 w_j（训练后预期为负）
        self.modulus_weights = nn.Parameter(torch.zeros(modulus_count))

    def forward(self, even_numbers: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return sieve logits and the underlying sieve matrix.

        返回筛法 logits 及底层筛法矩阵（便于分析与可视化）。
        """
        # Step 1: build z[b, k, m] from N and fixed prime/modulus tables.
        # 步骤 1：由 N 和固定素数表构建筛法矩阵
        sieve_matrix = compute_sieve_matrix(
            even_numbers,
            self.candidate_primes,
            self.modulus_bases,
        )
        sieve_matrix = apply_small_prime_sieve_bypass(
            sieve_matrix,
            even_numbers,
            self.candidate_primes,
            self.modulus_bases,
        )

        # Step 2: linear combination over moduli for each candidate.
        # 步骤 2：对每个候选，在所有模数上做线性加权求和
        # sieve_logits[b, k] = b_k + sum_m z[b,k,m] * w_m
        sieve_logits = self.candidate_bias + torch.einsum(
            "bkm,m->bk",
            sieve_matrix,
            self.modulus_weights,
        )
        sieve_logits = apply_eligibility_mask(
            sieve_logits,
            even_numbers,
            self.candidate_primes,
        )
        eligibility_mask = compute_eligibility_mask(even_numbers, self.candidate_primes)
        return {
            "sieve_logits": sieve_logits,
            "sieve_matrix": sieve_matrix,
            "eligibility_mask": eligibility_mask,
        }

    def modulus_weight_table(self) -> list[tuple[int, float]]:
        """Export (modulus, weight) pairs for interpretability reports.

        导出 (模数, 权重) 对，用于可解释性分析报告。
        """
        values = self.modulus_weights.detach().cpu().tolist()
        bases = self.modulus_bases.detach().cpu().tolist()
        return list(zip(bases, values))
