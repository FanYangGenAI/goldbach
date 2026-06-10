#!/usr/bin/env python3
"""Smoke test for GoldbachBinDataset and DataLoader."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

from dataset import GoldbachBinDataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Smoke test IGOT PyTorch dataset loader")
    parser.add_argument(
        "--format",
        choices=("v1", "v2"),
        default="v2",
        help="Dataset format shortcut",
    )
    parser.add_argument("--data", type=Path, default=None, help="Dataset .bin path")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def _default_data_path(root: Path, dataset_format: str) -> Path:
    if dataset_format == "v2":
        return root / "data" / "goldbach_train_v2_phase0.bin"
    return root / "data" / "goldbach_phase0.bin"


def main() -> int:
    args = parse_args()
    root = _repo_root()
    data_path = args.data or _default_data_path(root, args.format)
    if not data_path.exists():
        print(f"Dataset not found: {data_path}", file=sys.stderr)
        return 1

    dataset = GoldbachBinDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    batch = next(iter(loader))

    print("PyTorch loader smoke test passed.")
    print(f"  path:           {data_path}")
    print(f"  version:        {dataset.version}")
    print(f"  dataset size:   {len(dataset):,}")
    print(f"  batch even:     {tuple(batch['even'].shape)}")
    print(f"  batch bits:     {tuple(batch['bits'].shape)}")
    print(f"  batch mods:     {tuple(batch['mods'].shape)}")
    print(f"  batch density:  {tuple(batch['density'].shape)}")
    if "valid_mask" in batch:
        print(f"  batch mask:     {tuple(batch['valid_mask'].shape)}")
        print(f"  mask sum[0]:    {batch['valid_mask'][0].sum().item():.0f}")
    if "p1_label" in batch:
        print(f"  p1_label[0]:    {batch['p1_label'][0].item()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
