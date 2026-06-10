"""Phase 2 model: explicit sieve plus per-candidate residual MLP.

Phase 2 模型：显式筛法 + 每个候选的 residual MLP。
Decomposition: final_logit = sieve_logit + residual_logit (interpretable sum).
分解：final_logit = sieve_logit + residual_logit，保持可解释加法结构。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from number_encoder import NumberEncoder
from sieve_baseline import ExplicitSieveModule


class SieveResidualModel(nn.Module):
    """Sieve + residual model for IGOT Phase 2.

    IGOT Phase 2 组合模型：
    - sieve branch: interpretable linear sieve (Phase 1)
    - residual branch: small MLP capturing patterns beyond low-order sieving
    """

    def __init__(
        self,
        candidate_count: int = 500,
        modulus_count: int = 32,
        d_model: int = 128,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        # Reuse Phase 1 sieve module unchanged in structure.
        # 复用 Phase 1 的显式筛法模块
        self.sieve_module = ExplicitSieveModule(
            candidate_count=candidate_count,
            modulus_count=modulus_count,
        )

        # Encode global information of N (bits, mods, density, Fourier).
        # 编码 N 的全局信息
        self.number_encoder = NumberEncoder(d_model=d_model)

        # Per-candidate residual MLP input:
        #   [log(p_i), log(N-p_i), sieve_row_i, global_context]
        # 每个候选的 residual 输入特征（见 _candidate_features）
        residual_input_dim = d_model + 2 + modulus_count
        self.residual_mlp = nn.Sequential(
            nn.Linear(residual_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Precompute log(p_i) for all candidates (constant buffer).
        # 预计算所有候选素数的 log(p_i)，训练时不变
        self.register_buffer(
            "log_candidate_primes",
            torch.log(self.sieve_module.candidate_primes.float()),
            persistent=False,
        )

    def _candidate_features(
        self,
        even_numbers: torch.Tensor,
        global_context: torch.Tensor,
        sieve_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Build per-candidate feature tensor [B, K, F].

        为每个候选 p_i 构建特征向量，形状 [B, K, F]。

        Features per candidate:
            log(p_i)           - candidate identity / scale
            log(N - p_i)       - offset magnitude
            z[i, :]            - sieve indicators for this candidate
            global_context     - shared N-level representation

        每个候选的特征：
            log(p_i)           - 候选自身尺度
            log(N - p_i)       - 偏移量尺度
            z[i, :]            - 该候选的筛法行
            global_context     - N 的全局上下文（所有候选共享）
        """
        # q = N - p_i for every candidate in the batch.
        # 计算每个候选的偏移量 q = N - p_i
        offsets = even_numbers.unsqueeze(1) - self.sieve_module.candidate_primes.unsqueeze(0)

        # log(N - p_i), clamped to avoid log(0) or log(1).
        # log(N - p_i)，下限截断避免 log(0)/log(1)
        log_offsets = torch.log(offsets.float().clamp(min=2.0)).unsqueeze(-1)

        # log(p_i) broadcast to [B, K, 1].
        # 将 log(p_i) 广播到 [B, K, 1]
        log_primes = self.log_candidate_primes.view(1, -1, 1).expand(even_numbers.shape[0], -1, -1)

        # Broadcast global N context to every candidate token.
        # 将全局 N 上下文广播到每个候选
        global_broadcast = global_context.unsqueeze(1).expand(-1, log_primes.shape[1], -1)

        return torch.cat([log_primes, log_offsets, sieve_matrix, global_broadcast], dim=-1)

    def forward(
        self,
        even_numbers: torch.Tensor,
        bits: torch.Tensor,
        mods: torch.Tensor,
        density: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with decomposed logits for analysis.

        前向传播，返回可分解的 logits 便于 Phase 4 归因分析。
        """
        # --- Step 1: explicit sieve branch (Phase 1) ---
        # --- 步骤 1：显式筛法分支 ---
        sieve_outputs = self.sieve_module(even_numbers)
        sieve_logits = sieve_outputs["sieve_logits"]   # [B, K]
        sieve_matrix = sieve_outputs["sieve_matrix"]   # [B, K, M]

        # --- Step 2: encode global N context ---
        # --- 步骤 2：编码 N 的全局上下文 ---
        global_context = self.number_encoder(even_numbers, bits, mods, density)  # [B, d_model]

        # --- Step 3: build per-candidate residual features ---
        # --- 步骤 3：构建每个候选的 residual 特征 ---
        candidate_features = self._candidate_features(
            even_numbers,
            global_context,
            sieve_matrix,
        )  # [B, K, F]

        # --- Step 4: residual MLP produces one logit per candidate ---
        # --- 步骤 4：residual MLP 为每个候选输出一个 logit ---
        residual_logits = self.residual_mlp(candidate_features).squeeze(-1)  # [B, K]

        # --- Step 5: additive decomposition (core IGOT design) ---
        # --- 步骤 5：可解释加法分解（IGOT 核心设计） ---
        final_logits = sieve_logits + residual_logits

        return {
            "final_logits": final_logits,
            "sieve_logits": sieve_logits,
            "residual_logits": residual_logits,
            "sieve_matrix": sieve_matrix,
        }

    def modulus_weight_table(self) -> list[tuple[int, float]]:
        """Delegate to inner sieve module for reporting."""
        return self.sieve_module.modulus_weight_table()
