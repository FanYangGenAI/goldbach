"""Global number context encoder for IGOT.

全局偶数 N 编码器，为 Phase 2+ 的 residual / transformer 分支提供上下文。
Encodes: bit pattern, modular residues, scale (log N, density), Fourier features.
编码：位模式、模余特征、尺度信息（log N、density）、傅里叶周期特征。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_GENERATOR_PYTHON = Path(__file__).resolve().parents[1] / "generator" / "python"
if str(_GENERATOR_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GENERATOR_PYTHON))

from constants import SMALL_PRIMES  # noqa: E402


class NumberEncoder(nn.Module):
    """Encode even N into a global context vector of size d_model.

    将偶数 N 编码为长度 d_model 的全局上下文向量。
    输出将被广播到每个候选素数，用于 residual 分支。
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.modulus_count = len(SMALL_PRIMES)

        # Bit encoder: Conv1D along the 64-bit sequence (not a single collapsed vector).
        # 位编码器：沿 64-bit 序列做 Conv1D（而非把 64 bit 压成单向量）
        self.bit_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # One embedding table per small prime r_j for residue N mod r_j.
        # 每个小素数模数单独一张 embedding 表，输入为离散余数 N mod r_j
        self.mod_embeddings = nn.ModuleList(
            nn.Embedding(prime, 8) for prime in SMALL_PRIMES
        )

        # 64 (bits) + 10*8 (mod emb) + 1 (log N) + 1 (density) + 10*2 (Fourier)
        feature_dim = 64 + self.modulus_count * 8 + 2 + self.modulus_count * 2
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        primes = torch.tensor(SMALL_PRIMES, dtype=torch.float32)
        self.register_buffer("modulus_bases", primes, persistent=False)

    def _fourier_features(self, mods: torch.Tensor) -> torch.Tensor:
        """sin/cos encoding of modular residues.

        对模余值做 sin/cos 周期编码，帮助模型表达模空间周期结构。
        """
        angles = (2.0 * math.pi * mods.float()) / self.modulus_bases.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(
        self,
        even_numbers: torch.Tensor,
        bits: torch.Tensor,
        mods: torch.Tensor,
        density: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one batch of even numbers.

        对一个 batch 的偶数 N 进行编码。

        Args:
            even_numbers: [B] int64, the even N
            bits:         [B, 64] float, binary expansion of N
            mods:         [B, 10] int64, N mod {3,5,...,31}
            density:      [B] float, 1/log(N)

        Returns:
            global_context: [B, d_model]
        """
        # --- Branch 1: bit-level local patterns via 1D convolution ---
        # --- 分支 1：用 1D 卷积提取 bit 级局部模式 ---
        bit_seq = bits.unsqueeze(1)  # [B, 1, 64]
        bit_features = self.bit_conv(bit_seq).squeeze(-1)  # [B, 64]

        # --- Branch 2: discrete modular embeddings ---
        # --- 分支 2：离散模余 embedding ---
        mod_features = []
        for index, embedding in enumerate(self.mod_embeddings):
            mod_features.append(embedding(mods[:, index]))
        mod_features_tensor = torch.cat(mod_features, dim=-1)  # [B, 80]

        # --- Branch 3: continuous scale features ---
        # --- 分支 3：连续尺度特征 ---
        log_n = torch.log(even_numbers.float()).unsqueeze(-1)  # [B, 1]
        density_feat = density.unsqueeze(-1)  # [B, 1]

        # --- Branch 4: explicit periodic Fourier features ---
        # --- 分支 4：显式傅里叶周期特征 ---
        fourier = self._fourier_features(mods)  # [B, 20]

        # Concatenate all branches, then project to d_model.
        # 拼接所有分支，再投影到 d_model 维
        features = torch.cat(
            [bit_features, mod_features_tensor, log_n, density_feat, fourier],
            dim=-1,
        )
        return self.projection(features)
