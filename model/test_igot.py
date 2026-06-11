#!/usr/bin/env python3
"""Unit tests for Phase 3 IGOT model.

Phase 3 IGOT 模型单元测试：形状、梯度、注意力权重导出、模块消融。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_MODEL_ROOT = Path(__file__).resolve().parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from igot import IGOTAblation, IGOTModel  # noqa: E402


class TestIGOT(unittest.TestCase):
    def _sample_batch(self, batch_size: int = 2) -> tuple:
        """Build a small synthetic batch for smoke tests."""
        even = torch.tensor([100, 204], dtype=torch.int64)[:batch_size]
        bits = torch.rand(batch_size, 64)
        mods = torch.tensor(
            [
                [1, 0, 2, 6, 9, 0, 14, 20, 1, 18],
                [0, 4, 1, 6, 9, 0, 14, 20, 1, 18],
            ],
            dtype=torch.int64,
        )[:batch_size]
        density = torch.tensor([0.2, 0.1], dtype=torch.float32)[:batch_size]
        return even, bits, mods, density

    def test_forward_shapes(self) -> None:
        """final = sieve + attention + residual; shapes [B,K]."""
        model = IGOTModel(candidate_count=16, modulus_count=8, d_model=32, num_heads=2)
        even, bits, mods, density = self._sample_batch()
        outputs = model(even, bits, mods, density)
        self.assertEqual(tuple(outputs["final_logits"].shape), (2, 16))
        self.assertEqual(tuple(outputs["attention_logits"].shape), (2, 16))
        self.assertEqual(tuple(outputs["residual_logits"].shape), (2, 16))
        self.assertTrue(torch.allclose(
            outputs["final_logits"],
            outputs["sieve_logits"]
            + outputs["attention_logits"]
            + outputs["residual_logits"],
        ))

    def test_attention_weights(self) -> None:
        """return_attention=True exports cross [K,M] and self [K,K] weights."""
        model = IGOTModel(candidate_count=8, modulus_count=4, d_model=16, num_heads=2)
        even, bits, mods, density = self._sample_batch(batch_size=1)
        outputs = model(even, bits, mods, density, return_attention=True)
        self.assertIn("cross_attn_weights", outputs)
        self.assertIn("self_attn_weights", outputs)
        self.assertEqual(tuple(outputs["cross_attn_weights"].shape)[-2:], (8, 4))

    def test_ablation(self) -> None:
        """Zeroing all branches via IGOTAblation yields zero final_logits."""
        model = IGOTModel(candidate_count=8, modulus_count=4, d_model=16, num_heads=2)
        even, bits, mods, density = self._sample_batch(batch_size=1)
        full = model(even, bits, mods, density)
        ablated = model(
            even,
            bits,
            mods,
            density,
            ablation=IGOTAblation(sieve=True, attention=True, residual=True),
        )
        self.assertTrue(torch.allclose(ablated["final_logits"], torch.zeros_like(full["final_logits"])))

    def test_backward_flows(self) -> None:
        """Gradients reach attention head and SDPA Q projection."""
        model = IGOTModel(candidate_count=8, modulus_count=4, d_model=16, num_heads=2)
        even, bits, mods, density = self._sample_batch(batch_size=1)
        target = torch.zeros(1, 8)
        target[0, 0] = 1.0
        outputs = model(even, bits, mods, density)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["final_logits"],
            target,
        )
        loss.backward()
        self.assertIsNotNone(model.attention_head.weight.grad)
        self.assertIsNotNone(model.cross_attn_layers[0].attention.q_proj.weight.grad)

    def test_multi_layer_sequential(self) -> None:
        """sequential stack: C cross layers then L self layers."""
        model = IGOTModel(
            candidate_count=8,
            modulus_count=4,
            d_model=16,
            num_heads=2,
            cross_attn_layers=2,
            self_attn_layers=3,
            stack_mode="sequential",
        )
        self.assertEqual(len(model.cross_attn_layers), 2)
        self.assertEqual(len(model.self_attn_layers), 3)
        even, bits, mods, density = self._sample_batch(batch_size=1)
        outputs = model(even, bits, mods, density)
        self.assertEqual(tuple(outputs["final_logits"].shape), (1, 8))
        self.assertFalse(torch.isnan(outputs["final_logits"]).any())

    def test_paired_stack(self) -> None:
        """paired stack: L times (cross -> self) units."""
        model = IGOTModel(
            candidate_count=8,
            modulus_count=4,
            d_model=16,
            num_heads=2,
            cross_attn_layers=2,
            self_attn_layers=2,
            stack_mode="paired",
        )
        even, bits, mods, density = self._sample_batch(batch_size=1)
        outputs = model(even, bits, mods, density, return_attention=True)
        self.assertEqual(tuple(outputs["cross_attn_weights"].shape)[-2:], (8, 4))
        self.assertFalse(torch.isnan(outputs["final_logits"]).any())


if __name__ == "__main__":
    unittest.main()
