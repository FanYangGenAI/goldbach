"""Full IGOT model for Phase 3: sieve + attention + residual.

Phase 3 完整 IGOT 模型。
Decomposition: final_logit = sieve_logit + attention_logit + residual_logit.
分解：final_logit = sieve_logit + attention_logit + residual_logit。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from attention import PrimeModulusCrossAttention, PrimePrimeSelfAttention
from number_encoder import NumberEncoder
from sieve_baseline import ExplicitSieveModule, apply_eligibility_mask
from tokens import ModulusTokenBuilder, PrimeTokenBuilder

StackMode = Literal["sequential", "paired"]


@dataclass
class IGOTAblation:
    """Module-level ablation flags for interpretability experiments.

    模块级消融开关，供 Phase 4 机制分析使用。
    True = zero-out that branch's contribution to final_logits.
    """

    sieve: bool = False
    attention: bool = False
    residual: bool = False


class IGOTModel(nn.Module):
    """Interpretable Goldbach Offset Transformer (Phase 3).

    IGOT Phase 3 完整模型，三路 logit 加法分解。
    Attention stack depth is configurable via cross_attn_layers / self_attn_layers.
    注意力栈深度可配置；stack_mode 控制 cross/self 的堆叠顺序。
    """

    def __init__(
        self,
        candidate_count: int = 500,
        modulus_count: int = 32,
        d_model: int = 128,
        num_heads: int = 4,
        cross_attn_layers: int = 1,
        self_attn_layers: int = 1,
        stack_mode: StackMode = "sequential",
    ) -> None:
        super().__init__()
        if cross_attn_layers < 1 or self_attn_layers < 1:
            raise ValueError("cross_attn_layers and self_attn_layers must be >= 1")
        if stack_mode not in ("sequential", "paired"):
            raise ValueError(f"Unknown stack_mode: {stack_mode}")

        self.candidate_count = candidate_count
        self.modulus_count = modulus_count
        self.d_model = d_model
        self.cross_attn_layers_count = cross_attn_layers
        self.self_attn_layers_count = self_attn_layers
        self.stack_mode = stack_mode

        # --- Branch A: interpretable explicit sieve (Phase 1, reused) ---
        # --- 分支 A：可解释显式筛法（复用 Phase 1）---
        self.sieve_module = ExplicitSieveModule(
            candidate_count=candidate_count,
            modulus_count=modulus_count,
        )

        # --- Branch B/C shared: encode N and build tokens for Transformer ---
        # --- 分支 B/C 共享：编码 N 并构建 Transformer token ---
        self.number_encoder = NumberEncoder(d_model=d_model)
        self.prime_token_builder = PrimeTokenBuilder(
            candidate_count=candidate_count,
            modulus_count=modulus_count,
            d_model=d_model,
        )
        self.modulus_token_builder = ModulusTokenBuilder(
            modulus_count=modulus_count,
            d_model=d_model,
        )

        # --- Attention stack (configurable depth) ---
        # --- 注意力栈（深度可配置）---
        self.cross_attn_layers = nn.ModuleList(
            PrimeModulusCrossAttention(d_model=d_model, num_heads=num_heads)
            for _ in range(cross_attn_layers)
        )
        self.self_attn_layers = nn.ModuleList(
            PrimePrimeSelfAttention(d_model=d_model, num_heads=num_heads)
            for _ in range(self_attn_layers)
        )

        # Per-candidate scalar heads on final prime tokens.
        # 在最终 prime token 上接两个标量输出头
        self.attention_head = nn.Linear(d_model, 1)
        self.residual_head = nn.Linear(d_model, 1)

        self.register_buffer(
            "log_candidate_primes",
            torch.log(self.sieve_module.candidate_primes.float()),
            persistent=False,
        )

        self._last_cross_attn_weights: torch.Tensor | None = None
        self._last_self_attn_weights: torch.Tensor | None = None

    def _run_attention_stack(
        self,
        prime_tokens: torch.Tensor,
        modulus_tokens: torch.Tensor,
        *,
        return_attention: bool,
    ) -> torch.Tensor:
        """Apply cross/self attention blocks according to stack_mode.

        sequential: all cross layers, then all self layers (Stage I / II).
        paired: repeat (cross -> self) units L times (Stage III).
        """
        cross_weights: torch.Tensor | None = None
        self_weights: torch.Tensor | None = None

        if self.stack_mode == "sequential":
            for layer in self.cross_attn_layers:
                if return_attention:
                    prime_tokens, cross_weights = layer(
                        prime_tokens,
                        modulus_tokens,
                        return_attention=True,
                    )
                else:
                    prime_tokens = layer(prime_tokens, modulus_tokens)
            for layer in self.self_attn_layers:
                if return_attention:
                    prime_tokens, self_weights = layer(
                        prime_tokens,
                        return_attention=True,
                    )
                else:
                    prime_tokens = layer(prime_tokens)
        else:
            depth = min(len(self.cross_attn_layers), len(self.self_attn_layers))
            for index in range(depth):
                cross_layer = self.cross_attn_layers[index]
                self_layer = self.self_attn_layers[index]
                if return_attention:
                    prime_tokens, cross_weights = cross_layer(
                        prime_tokens,
                        modulus_tokens,
                        return_attention=True,
                    )
                    prime_tokens, self_weights = self_layer(
                        prime_tokens,
                        return_attention=True,
                    )
                else:
                    prime_tokens = cross_layer(prime_tokens, modulus_tokens)
                    prime_tokens = self_layer(prime_tokens)

        if return_attention:
            self._last_cross_attn_weights = cross_weights
            self._last_self_attn_weights = self_weights
        return prime_tokens

    def forward(
        self,
        even_numbers: torch.Tensor,
        bits: torch.Tensor,
        mods: torch.Tensor,
        density: torch.Tensor,
        *,
        ablation: IGOTAblation | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward with decomposed logits for training and Phase 4 analysis.

        前向传播，返回 sieve / attention / residual 三路 logits。

        Args:
            even_numbers: [B] int64
            bits:         [B, 64] float
            mods:         [B, 10] int64
            density:      [B] float
            ablation:     optional module zero-out flags
            return_attention: if True, use manual attention and export weights
        """
        ablation = ablation or IGOTAblation()

        # --- Step 1: explicit sieve branch (interpretable, Phase 1) ---
        # --- 步骤 1：显式筛法分支 ---
        sieve_outputs = self.sieve_module(even_numbers)
        sieve_logits = sieve_outputs["sieve_logits"]      # [B, K]
        sieve_matrix = sieve_outputs["sieve_matrix"]    # [B, K, M]

        # --- Step 2: global N encoding ---
        # --- 步骤 2：全局 N 编码 ---
        global_context = self.number_encoder(even_numbers, bits, mods, density)  # [B, d]

        # --- Step 3: build prime / modulus token sequences ---
        # --- 步骤 3：构建 prime / modulus token 序列 ---
        prime_tokens = self.prime_token_builder(
            even_numbers,
            self.sieve_module.candidate_primes,
            self.sieve_module.modulus_bases,
            sieve_matrix,
            global_context,
            self.log_candidate_primes,
        )  # [B, K, d]
        modulus_tokens = self.modulus_token_builder(
            even_numbers,
            sieve_matrix,
            global_context,
        )  # [B, M, d]

        # --- Step 4: configurable cross/self attention stack ---
        # --- 步骤 4：可配置深度的 cross/self 注意力栈 ---
        prime_tokens = self._run_attention_stack(
            prime_tokens,
            modulus_tokens,
            return_attention=return_attention,
        )

        # --- Step 5: per-candidate attention / residual logits ---
        # --- 步骤 5：每个候选一个 attention / residual logit ---
        attention_logits = self.attention_head(prime_tokens).squeeze(-1)  # [B, K]
        residual_logits = self.residual_head(prime_tokens).squeeze(-1)      # [B, K]

        # --- Step 6: optional ablation (Phase 4 causal experiments) ---
        # --- 步骤 6：可选消融 ---
        if ablation.sieve:
            sieve_logits = torch.zeros_like(sieve_logits)
        if ablation.attention:
            attention_logits = torch.zeros_like(attention_logits)
        if ablation.residual:
            residual_logits = torch.zeros_like(residual_logits)

        # --- Step 7: additive decomposition (core IGOT design) ---
        # --- 步骤 7：可解释加法分解 ---
        final_logits = sieve_logits + attention_logits + residual_logits
        final_logits = apply_eligibility_mask(
            final_logits,
            even_numbers,
            self.sieve_module.candidate_primes,
        )

        outputs: dict[str, torch.Tensor] = {
            "final_logits": final_logits,
            "sieve_logits": sieve_logits,
            "attention_logits": attention_logits,
            "residual_logits": residual_logits,
            "sieve_matrix": sieve_matrix,
            "eligibility_mask": sieve_outputs["eligibility_mask"],
            "prime_tokens": prime_tokens,
            "modulus_tokens": modulus_tokens,
        }
        if return_attention:
            outputs["cross_attn_weights"] = self._last_cross_attn_weights
            outputs["self_attn_weights"] = self._last_self_attn_weights
        return outputs

    def modulus_weight_table(self) -> list[tuple[int, float]]:
        """Export sieve modulus weights for training reports."""
        return self.sieve_module.modulus_weight_table()

    def get_last_attention_weights(self) -> dict[str, torch.Tensor | None]:
        """Return cached attention weights from last forward(return_attention=True).

        返回最近一次 forward(return_attention=True) 缓存的注意力权重。
        """
        return {
            "cross_attn": self._last_cross_attn_weights,
            "self_attn": self._last_self_attn_weights,
        }
