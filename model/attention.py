"""Attention blocks for IGOT Phase 3.

Phase 3 注意力模块：Prime-Modulus Cross-Attention 与 Prime-Prime Self-Attention。
Uses PyTorch SDPA (FlashAttention / memory-efficient when available on CUDA).
训练路径使用 scaled_dot_product_attention；仅在 return_attention=True 时回退到手算权重。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSDPA(nn.Module):
    """Multi-head attention via scaled_dot_product_attention.

    多头注意力封装：
    - 训练/推理默认：F.scaled_dot_product_attention（CUDA 上可走 Flash / mem-efficient）
    - 可视化模式：return_attention=True 时手算 softmax 以导出权重矩阵
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        # Separate Q/K/V projections (same pattern as standard Transformer).
        # 独立的 Q/K/V 线性投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """[B, L, d_model] -> [B, heads, L, head_dim]."""
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
        """[B, heads, L, head_dim] -> [B, L, d_model]."""
        batch_size = tensor.shape[0]
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def _manual_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explicit softmax attention for weight export (Phase 4 interpretability).

        显式计算注意力权重，供 Phase 4 head-modulus 对齐分析使用。
        Shapes: Q [B,H,Lq,d], K [B,H,Lk,d] -> weights [B,H,Lq,Lk].
        """
        scale = self.head_dim ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        if self.training and self.dropout > 0.0:
            weights = F.dropout(weights, p=self.dropout)
        output = torch.matmul(weights, value)
        return output, weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run multi-head attention.

        Args:
            query: [B, Lq, d_model]
            key:   [B, Lk, d_model]
            value: [B, Lk, d_model]
        """
        q_len = query.shape[1]

        # --- Step 1: project and split into heads ---
        # --- 步骤 1：线性投影并拆成多头 ---
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        # --- Step 2: attention (SDPA fast path vs manual weights path) ---
        # --- 步骤 2：注意力（SDPA 快速路径 vs 手算权重路径）---
        if return_attention:
            attn_output, attn_weights = self._manual_attention(q, k, v)
        else:
            # SDPA: avoids materializing full [Lq,Lk] matrix in HBM when Flash is active.
            # SDPA：Flash 启用时不物化完整 [Lq,Lk] 注意力矩阵，降低峰值显存
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            attn_weights = None

        # --- Step 3: merge heads and output projection ---
        # --- 步骤 3：合并多头并做输出投影 ---
        merged = self._merge_heads(attn_output, q_len)
        return self.out_proj(merged), attn_weights


class PrimeModulusCrossAttention(nn.Module):
    """Cross-attention: prime tokens query modulus tokens.

    交叉注意力：Prime token 作为 Query，Modulus token 作为 Key/Value。
    语义：每个候选 p_i 在判断时关注哪些模数约束 r_j。
    仅 1 层（1×MHA + FFN），与 Phase 3 浅层 Transformer 设计一致。
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadSDPA(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        # Post-attention FFN (Transformer encoder sub-layer).
        # 注意力后的前馈子层
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        prime_tokens: torch.Tensor,
        modulus_tokens: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Update prime tokens by attending to modulus tokens.

        Args:
            prime_tokens:   [B, K, d_model]
            modulus_tokens: [B, M, d_model]
        """
        # Q=prime, K=V=modulus -> each candidate attends to M modulus dimensions.
        # Q=候选，K=V=模数 -> 每个候选关注 M 个模数维度
        attn_output, attn_weights = self.attention(
            prime_tokens,
            modulus_tokens,
            modulus_tokens,
            return_attention=return_attention,
        )
        # Residual + LayerNorm (pre-norm style stacking).
        prime_tokens = self.norm(prime_tokens + attn_output)
        prime_tokens = self.ffn_norm(prime_tokens + self.ffn(prime_tokens))

        if return_attention:
            if attn_weights is None:
                raise RuntimeError("Attention weights missing in return_attention mode")
            return prime_tokens, attn_weights
        return prime_tokens


class PrimePrimeSelfAttention(nn.Module):
    """Self-attention over prime tokens.

    自注意力：候选素数 token 之间的交互（竞争、替代、模空间等价类）。
    计算复杂度 O(K^2)；SDPA 降低峰值显存（相对物化完整 [K,K] 权重矩阵）。
    仅 1 层，是 Phase 3 batch 受限的主要瓶颈（K=500 时）。
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadSDPA(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        prime_tokens: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply self-attention among K prime tokens [B, K, d_model]."""
        attn_output, attn_weights = self.attention(
            prime_tokens,
            prime_tokens,
            prime_tokens,
            return_attention=return_attention,
        )
        prime_tokens = self.norm(prime_tokens + attn_output)
        prime_tokens = self.ffn_norm(prime_tokens + self.ffn(prime_tokens))

        if return_attention:
            if attn_weights is None:
                raise RuntimeError("Attention weights missing in return_attention mode")
            return prime_tokens, attn_weights
        return prime_tokens
