#!/usr/bin/env python3
"""Read and print one dataset sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_GENERATOR_PYTHON = _ROOT / "generator" / "python"
if str(_GENERATOR_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GENERATOR_PYTHON))

from constants import HEADER_SIZE, SAMPLE_SIZE  # noqa: E402
from format_io import GoldbachSample, verify_header  # noqa: E402


def read_sample(data_path: Path, index: int) -> GoldbachSample:
    with data_path.open("rb") as handle:
        header = verify_header(handle.read(HEADER_SIZE))
        if index < 0 or index >= header.sample_count:
            raise ValueError(f"index out of range: {index} (count={header.sample_count})")
        handle.seek(HEADER_SIZE + index * SAMPLE_SIZE)
        return GoldbachSample.from_bytes(handle.read(SAMPLE_SIZE))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Read one Goldbach dataset sample")
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "goldbach_phase0.bin",
        help="Dataset .bin path",
    )
    parser.add_argument("--index", type=int, default=0, help="Sample index")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.exists():
        print(f"Dataset not found: {args.data}", file=sys.stderr)
        return 1

    sample = read_sample(args.data, args.index)
    p2 = sample.even_number - sample.p1_label
    print(f"index:           {args.index}")
    print(f"even_number:     {sample.even_number}")
    print(f"p1_label:        {sample.p1_label}")
    print(f"p2:              {p2}")
    print(f"modulo_features: {list(sample.modulo_features)}")
    print(f"prime_density:   {sample.prime_density:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
