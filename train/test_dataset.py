#!/usr/bin/env python3
"""Unit tests for GoldbachBinDataset."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "train") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "train"))

from dataset import GoldbachBinDataset, decode_valid_mask_array, even_to_bits  # noqa: E402

_GENERATOR_PYTHON = _REPO_ROOT / "generator" / "python"
if str(_GENERATOR_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GENERATOR_PYTHON))

from constants import VERSION_V2  # noqa: E402
from format_io import make_sample_v2  # noqa: E402
from labeling import compute_valid_mask_default  # noqa: E402


class TestDatasetHelpers(unittest.TestCase):
    def test_even_to_bits_length(self) -> None:
        bits = even_to_bits(100)
        self.assertEqual(bits.shape, (64,))
        self.assertEqual(bits[2], 1.0)

    def test_decode_valid_mask_matches_labeling(self) -> None:
        sample = make_sample_v2(204)
        mask_bytes = sample.to_bytes()[22:]
        decoded = decode_valid_mask_array(mask_bytes)
        expected = torch.tensor(compute_valid_mask_default(204), dtype=torch.float32)
        self.assertTrue(torch.equal(torch.from_numpy(decoded), expected))


class TestGoldbachBinDataset(unittest.TestCase):
    def test_load_v2_dataset(self) -> None:
        data_path = _REPO_ROOT / "data" / "goldbach_train_v2_phase0.bin"
        if not data_path.exists():
            self.skipTest(f"dataset not found: {data_path}")

        dataset = GoldbachBinDataset(data_path)
        self.assertEqual(dataset.version, VERSION_V2)
        self.assertGreater(len(dataset), 0)

        item = dataset[100]
        self.assertEqual(item["even"].item(), 204)
        self.assertEqual(tuple(item["bits"].shape), (64,))
        self.assertEqual(tuple(item["mods"].shape), (10,))
        self.assertEqual(tuple(item["valid_mask"].shape), (500,))
        self.assertEqual(item["valid_mask"].sum().item(), 28.0)


if __name__ == "__main__":
    unittest.main()
