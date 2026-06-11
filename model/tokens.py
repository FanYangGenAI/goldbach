"""Prime and modulus token builders for IGOT Phase 3.

Phase 3 的 Prime / Modulus token 构建器。
Each token has explicit number-theoretic semantics for interpretability analysis.
每个 token 携带明确的数论语义，便于 Phase 4 机制分析。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def compute_elimination_stats(
    sieve_matrix: torch.Tensor,
    eligibility_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-modulus elimination counts/rates aggregated over eligible candidates only."""
    mask = eligibility_mask.unsqueeze(-1).to(dtype=sieve_matrix.dtype)
    eligible_z = sieve_matrix * mask
    eliminated_count = eligible_z.sum(dim=1)
    eligible_count = eligibility_mask.sum(dim=1).clamp_min(1).to(dtype=sieve_matrix.dtype)
    elimination_rate = eliminated_count / eligible_count.unsqueeze(-1)
    return eliminated_count, elimination_rate


class PrimeTokenBuilder(nn.Module):
    """Build per-candidate prime tokens [B, K, d_model].

    为每个候选素数 p_i 构建 token，语义：
    「对于当前偶数 N，候选 p_i 是否使得 N - p_i 为素数？」
    """

    def __init__(
        self,
        candidate_count: int,
        modulus_count: int,
        d_model: int = 128,
        prime_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.candidate_count = candidate_count
        self.modulus_count = modulus_count
        self.d_model = d_model

        # Discrete identity for candidate index i (which p_i in vocabulary).
        # 候选在词表中的离散编号 embedding
        self.prime_id_embedding = nn.Embedding(candidate_count, prime_embed_dim)

        # Feature layout per candidate:
        #   prime_emb | log(p) | log(N-p) | p%r | N%r | (N-p)%r | sieve_row | global_ctx
        # 每个候选特征布局见 forward 注释
        feature_dim = prime_embed_dim + 2 + modulus_count * 4 + d_model
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        even_numbers: torch.Tensor,
        candidate_primes: torch.Tensor,
        modulus_bases: torch.Tensor,
        sieve_matrix: torch.Tensor,
        global_context: torch.Tensor,
        log_candidate_primes: torch.Tensor,
    ) -> torch.Tensor:
        """Return prime tokens [B, K, d_model]."""
        batch_size = even_numbers.shape[0]
        device = even_numbers.device

        # --- Step 1: offsets q = N - p_i ---
        # --- 步骤 1：偏移量 q = N - p_i ---
        offsets = even_numbers.unsqueeze(1) - candidate_primes.unsqueeze(0)
        log_offsets = torch.log(offsets.float().clamp(min=2.0)).unsqueeze(-1)
        log_primes = log_candidate_primes.view(1, -1, 1).expand(batch_size, -1, -1)

        # --- Step 2: modular residues (normalized to [0,1) for continuity) ---
        # --- 步骤 2：模余特征（归一化到 [0,1)）---
        n_mod = even_numbers.unsqueeze(-1) % modulus_bases.view(1, -1)
        p_mod = candidate_primes.unsqueeze(1) % modulus_bases.unsqueeze(0)
        offset_mod = offsets.unsqueeze(-1) % modulus_bases.view(1, 1, -1)

        n_mod_norm = (n_mod.float() / modulus_bases.float().view(1, -1)).unsqueeze(1).expand(
            -1, self.candidate_count, -1
        )
        p_mod_norm = (p_mod.float() / modulus_bases.float().view(1, -1)).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        offset_mod_norm = offset_mod.float() / modulus_bases.float().view(1, 1, -1)

        # --- Step 3: candidate id embedding + broadcast global N context ---
        # --- 步骤 3：候选 id embedding + 广播全局 N 上下文 ---
        prime_ids = torch.arange(self.candidate_count, device=device)
        prime_emb = self.prime_id_embedding(prime_ids).unsqueeze(0).expand(batch_size, -1, -1)
        global_broadcast = global_context.unsqueeze(1).expand(-1, self.candidate_count, -1)

        # --- Step 4: concat all features and project to d_model ---
        # --- 步骤 4：拼接特征并投影到 d_model ---
        features = torch.cat(
            [
                prime_emb,
                log_primes,
                log_offsets,
                p_mod_norm,
                n_mod_norm,
                offset_mod_norm,
                sieve_matrix,
                global_broadcast,
            ],
            dim=-1,
        )
        return self.projection(features)


class ModulusTokenBuilder(nn.Module):
    """Build per-modulus tokens [B, M, d_model].

    为每个筛法模数 r_j 构建 token，语义：「模数 r_j 代表的筛法维度」。
    供 Prime-Modulus Cross-Attention 的 Key/Value 使用。
    """

    def __init__(
        self,
        modulus_count: int,
        d_model: int = 128,
        modulus_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.modulus_count = modulus_count
        self.d_model = d_model

        self.modulus_id_embedding = nn.Embedding(modulus_count, modulus_embed_dim)
        # One embedding table per modulus r_j for discrete residue N mod r_j.
        # 每个模数 r_j 一张离散余数 embedding 表
        self.residue_embeddings = nn.ModuleList(
            nn.Embedding(int(base), 8) for base in _default_modulus_bases(modulus_count)
        )

        # modulus_emb(16) + residue_emb(8) + stats(2) + fourier(2) + global_ctx(d_model)
        feature_dim = modulus_embed_dim + 8 + 2 + 2 + d_model
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        bases = torch.tensor(_default_modulus_bases(modulus_count), dtype=torch.float32)
        self.register_buffer("modulus_bases", bases, persistent=False)

    def _fourier(self, residues: torch.Tensor) -> torch.Tensor:
        """sin/cos periodic encoding of N mod r_j.

        对 N mod r_j 做 sin/cos 周期编码，形状 [B, M, 2]。
        """
        angles = (2.0 * math.pi * residues.float()) / self.modulus_bases.unsqueeze(0)
        return torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(
        self,
        even_numbers: torch.Tensor,
        sieve_matrix: torch.Tensor,
        global_context: torch.Tensor,
        eligibility_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return modulus tokens [B, M, d_model]."""
        batch_size = even_numbers.shape[0]
        device = even_numbers.device

        # --- Step 1: N mod r_j for each sieve modulus ---
        # --- 步骤 1：对每个筛法模数计算 N mod r_j ---
        n_mod = (even_numbers.unsqueeze(-1) % self.modulus_bases.long().view(1, -1)).long()

        residue_features = []
        for index, embedding in enumerate(self.residue_embeddings):
            residue_features.append(embedding(n_mod[:, index]).unsqueeze(1))
        residue_tensor = torch.cat(residue_features, dim=1)  # [B, M, 8]

        # --- Step 2: elimination stats from eligible candidates only (F-1c) ---
        # --- 步骤 2：仅在合法候选上聚合淘汰统计 ---
        eliminated_count, elimination_rate = compute_elimination_stats(
            sieve_matrix,
            eligibility_mask,
        )

        # --- Step 3: modulus id + fourier + global context ---
        # --- 步骤 3：模数 id + 傅里叶特征 + 全局上下文 ---
        modulus_ids = torch.arange(self.modulus_count, device=device)
        modulus_emb = self.modulus_id_embedding(modulus_ids).unsqueeze(0).expand(batch_size, -1, -1)

        fourier = self._fourier(n_mod)
        stats = torch.stack([eliminated_count, elimination_rate], dim=-1)
        global_broadcast = global_context.unsqueeze(1).expand(-1, self.modulus_count, -1)

        features = torch.cat(
            [modulus_emb, residue_tensor, stats, fourier, global_broadcast],
            dim=-1,
        )
        return self.projection(features)


def _default_modulus_bases(modulus_count: int) -> list[int]:
    """Load first M odd primes used as sieve moduli (must match ExplicitSieveModule).

    加载前 M 个奇素数作为筛法模数基底（须与 ExplicitSieveModule 一致）。
    """
    import sys
    from pathlib import Path

    generator_python = Path(__file__).resolve().parents[1] / "generator" / "python"
    if str(generator_python) not in sys.path:
        sys.path.insert(0, str(generator_python))
    from primes import first_m_modulus_bases  # noqa: E402

    return first_m_modulus_bases(modulus_count)
