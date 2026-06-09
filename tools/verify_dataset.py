#!/usr/bin/env python3
"""Audit a Goldbach binary dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_GENERATOR_PYTHON = _ROOT / "generator" / "python"
if str(_GENERATOR_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GENERATOR_PYTHON))

from constants import HEADER_SIZE, SAMPLE_SIZE  # noqa: E402
from format_io import GoldbachSample, verify_header  # noqa: E402

from labeling import find_min_p1  # noqa: E402
from miller_rabin import is_prime_deterministic  # noqa: E402
from primes import PRIMES_UP_TO_MAX_P1, compute_modulo_features  # noqa: E402
import math  # noqa: E402


def _load_manifest(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_sample_at(path: Path, index: int) -> GoldbachSample:
    offset = HEADER_SIZE + index * SAMPLE_SIZE
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(SAMPLE_SIZE)
    if len(data) != SAMPLE_SIZE:
        raise ValueError(f"Failed to read sample at index {index}")
    return GoldbachSample.from_bytes(data)


def _verify_min_p1(even_number: int, p1_label: int) -> str | None:
    for prime in PRIMES_UP_TO_MAX_P1:
        if prime >= p1_label:
            break
        p2_candidate = even_number - prime
        if p2_candidate >= prime and is_prime_deterministic(p2_candidate):
            return f"smaller p1 exists: {prime}"
    recomputed = find_min_p1(even_number)
    if recomputed != p1_label:
        return f"label mismatch: stored={p1_label}, recomputed={recomputed}"
    return None


def verify_dataset(
    data_path: Path,
    manifest_path: Path | None,
    sample_size: int,
    seed: int,
) -> int:
    with data_path.open("rb") as handle:
        header_bytes = handle.read(HEADER_SIZE)
    header = verify_header(header_bytes)
    manifest = _load_manifest(manifest_path)

    if manifest is not None:
        if manifest.get("sample_count") != header.sample_count:
            print(
                "FAIL: manifest sample_count mismatch:",
                manifest.get("sample_count"),
                header.sample_count,
            )
            return 1

    expected_size = HEADER_SIZE + header.sample_count * SAMPLE_SIZE
    actual_size = data_path.stat().st_size
    if actual_size != expected_size:
        print(f"FAIL: file size mismatch: expected {expected_size}, got {actual_size}")
        return 1

    rng = random.Random(seed)
    indices = (
        list(range(header.sample_count))
        if sample_size >= header.sample_count
        else rng.sample(range(header.sample_count), sample_size)
    )

    seen_evens: set[int] = set()
    failures: list[str] = []

    for index in indices:
        sample = _read_sample_at(data_path, index)
        even_number = sample.even_number
        p1 = sample.p1_label
        p2 = even_number - p1

        if even_number < 4 or even_number % 2 != 0:
            failures.append(f"[{index}] invalid even_number={even_number}")
            continue
        if p1 > 5000:
            failures.append(f"[{index}] p1 out of range: {p1}")
            continue
        if p1 + p2 != even_number:
            failures.append(f"[{index}] sum mismatch")
            continue
        if not is_prime_deterministic(p2):
            failures.append(f"[{index}] p2 not prime: {p2}")
            continue
        if tuple(sample.modulo_features) != compute_modulo_features(even_number):
            failures.append(f"[{index}] modulo_features mismatch")
            continue
        expected_density = 1.0 / math.log(even_number)
        if abs(sample.prime_density - expected_density) > 1e-6:
            failures.append(f"[{index}] prime_density mismatch")
            continue
        min_p1_error = _verify_min_p1(even_number, p1)
        if min_p1_error is not None:
            failures.append(f"[{index}] {min_p1_error}")
        if even_number in seen_evens:
            failures.append(f"[{index}] duplicate even_number={even_number}")
        seen_evens.add(even_number)

    print(f"Dataset: {data_path}")
    print(f"Samples: {header.sample_count:,}")
    print(f"Checked: {len(indices):,}")
    print(f"Failures: {len(failures)}")

    if failures:
        for line in failures[:20]:
            print("  ", line)
        if len(failures) > 20:
            print(f"   ... and {len(failures) - 20} more")
        return 1

    print("PASS: audit checks succeeded.")
    return 0


def _default_paths(phase: str, root: Path) -> tuple[Path, Path]:
    if phase == "phase0":
        return root / "data" / "goldbach_phase0.bin", root / "data" / "goldbach_phase0_manifest.json"
    if phase == "phase1":
        return root / "data" / "goldbach_phase1.bin", root / "data" / "goldbach_phase1_manifest.json"
    if phase == "phase2":
        return root / "data" / "goldbach_train_v1.bin", root / "data" / "goldbach_manifest_v1.json"
    raise ValueError(f"Unknown phase: {phase}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify Goldbach dataset binary file")
    parser.add_argument(
        "--phase",
        choices=("phase0", "phase1", "phase2"),
        default=None,
        help="Shortcut for default data/manifest paths",
    )
    parser.add_argument("--data", type=Path, default=None, help="Dataset .bin path")
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest JSON path")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000,
        help="Number of random samples to audit",
    )
    parser.add_argument("--seed", type=int, default=7, help="Audit sampling seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase is not None:
        default_data, default_manifest = _default_paths(args.phase, root)
    else:
        default_data = root / "data" / "goldbach_phase0.bin"
        default_manifest = root / "data" / "goldbach_phase0_manifest.json"
    data_path = args.data or default_data
    manifest_path = args.manifest if args.manifest is not None else default_manifest

    if not data_path.exists():
        print(f"FAIL: dataset not found: {data_path}", file=sys.stderr)
        return 1
    return verify_dataset(data_path, manifest_path, args.sample_size, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
