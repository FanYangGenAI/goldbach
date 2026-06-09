#!/usr/bin/env python3
"""Smoke test for PyTorch dataset loading."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Smoke test mmap dataset loading")
    parser.add_argument(
        "--phase",
        choices=("phase0", "phase1", "phase2"),
        default="phase2",
        help="Dataset phase shortcut",
    )
    parser.add_argument("--data", type=Path, default=None, help="Dataset .bin path")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = _repo_root()
    if args.data is None:
        name = {
            "phase2": "goldbach_train_v1.bin",
            "phase1": "goldbach_phase1.bin",
        }.get(args.phase, "goldbach_phase0.bin")
        args.data = root / "data" / name
    if not args.data.exists():
        print(f"Dataset not found: {args.data}", file=sys.stderr)
        return 1

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        print("SKIP: torch is not installed")
        return 0

    HEADER_SIZE = 64
    SAMPLE_SIZE = 26

    class GoldbachBinDataset(Dataset):
        def __init__(self, path: Path):
            with path.open("rb") as handle:
                magic, version, count, *_ = struct.unpack("<IIQ", handle.read(16))
                if magic != 0x47424B48:
                    raise ValueError("invalid magic")
                self.count = count
            self.mmap = np.memmap(
                path,
                dtype=np.uint8,
                mode="r",
                offset=HEADER_SIZE,
                shape=(self.count, SAMPLE_SIZE),
            )

        def __len__(self) -> int:
            return self.count

        def __getitem__(self, idx: int):
            row = self.mmap[idx]
            even = int.from_bytes(row[0:8].tobytes(), "little")
            mods = row[8:18].astype(np.int64)
            density = struct.unpack("<f", row[18:22].tobytes())[0]
            p1 = int(struct.unpack("<I", row[22:26].tobytes())[0])
            bits = torch.tensor([(even >> bit) & 1 for bit in range(64)], dtype=torch.float32)
            return {
                "even": even,
                "bits": bits,
                "mods": torch.from_numpy(mods),
                "density": density,
                "p1_label": p1,
            }

    dataset = GoldbachBinDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print("PyTorch loader smoke test passed.")
    print(f"  dataset size: {len(dataset):,}")
    print(f"  batch even shape: {batch['even'].shape}")
    print(f"  batch bits shape: {batch['bits'].shape}")
    print(f"  sample p1_label[0]: {batch['p1_label'][0].item()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
