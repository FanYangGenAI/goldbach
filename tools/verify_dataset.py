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

from constants import (  # noqa: E402
    CANDIDATE_PRIME_COUNT,
    HEADER_SIZE,
    VERSION_V2,
)
from format_io import (  # noqa: E402
    GoldbachSample,
    GoldbachSampleV2,
    sample_size_for_version,
    verify_header,
)
from verify_sample import (  # noqa: E402
    dataset_version_label,
    verify_v1_sample,
    verify_v2_sample,
)


def _load_manifest(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_sample_bytes(path: Path, index: int, record_size: int) -> bytes:
    offset = HEADER_SIZE + index * record_size
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(record_size)
    if len(data) != record_size:
        raise ValueError(f"Failed to read sample at index {index}")
    return data


def _candidate_count_from_manifest(manifest: dict | None) -> int:
    if manifest is None:
        return CANDIDATE_PRIME_COUNT
    return int(manifest.get("candidate_prime_count", CANDIDATE_PRIME_COUNT))


def verify_dataset(
    data_path: Path,
    manifest_path: Path | None,
    audit_sample_size: int,
    seed: int,
) -> int:
    with data_path.open("rb") as handle:
        header_bytes = handle.read(HEADER_SIZE)
    header = verify_header(header_bytes)
    manifest = _load_manifest(manifest_path)
    record_size = sample_size_for_version(header.version)
    format_label = dataset_version_label(header.version)

    if manifest is not None:
        if manifest.get("sample_count") != header.sample_count:
            print(
                "FAIL: manifest sample_count mismatch:",
                manifest.get("sample_count"),
                header.sample_count,
            )
            return 1
        if manifest.get("version") not in (None, header.version):
            print(
                "FAIL: manifest version mismatch:",
                manifest.get("version"),
                header.version,
            )
            return 1
        manifest_sample_size = manifest.get("sample_size_bytes")
        if manifest_sample_size is not None and manifest_sample_size != record_size:
            print(
                "FAIL: manifest sample_size_bytes mismatch:",
                manifest_sample_size,
                record_size,
            )
            return 1

    expected_size = HEADER_SIZE + header.sample_count * record_size
    actual_size = data_path.stat().st_size
    if actual_size != expected_size:
        print(f"FAIL: file size mismatch: expected {expected_size}, got {actual_size}")
        return 1

    rng = random.Random(seed)
    indices = (
        list(range(header.sample_count))
        if audit_sample_size >= header.sample_count
        else rng.sample(range(header.sample_count), audit_sample_size)
    )

    candidate_count = _candidate_count_from_manifest(manifest)
    seen_evens: set[int] = set()
    failures: list[str] = []

    for index in indices:
        payload = _read_sample_bytes(data_path, index, record_size)
        if header.version == VERSION_V2:
            sample = GoldbachSampleV2.from_bytes(payload)
            error = verify_v2_sample(sample, candidate_count=candidate_count)
            even_number = sample.even_number
        else:
            sample = GoldbachSample.from_bytes(payload)
            error = verify_v1_sample(sample)
            even_number = sample.even_number

        if error is not None:
            failures.append(f"[{index}] {error}")
        if even_number in seen_evens:
            failures.append(f"[{index}] duplicate even_number={even_number}")
        seen_evens.add(even_number)

    print(f"Dataset: {data_path}")
    print(f"Format:  {format_label} (version={header.version}, record_size={record_size})")
    if header.version == VERSION_V2:
        print(f"Candidates: K={candidate_count}")
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


def _default_paths(
    phase: str,
    root: Path,
    dataset_format: str,
) -> tuple[Path, Path]:
    if phase == "phase0" and dataset_format == "v2":
        return (
            root / "data" / "goldbach_train_v2_phase0.bin",
            root / "data" / "goldbach_manifest_v2_phase0.json",
        )
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
    parser.add_argument(
        "--format",
        choices=("v1", "v2"),
        default="v1",
        help="Default dataset format when using --phase shortcuts",
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
        default_data, default_manifest = _default_paths(args.phase, root, args.format)
    elif args.format == "v2":
        default_data = root / "data" / "goldbach_train_v2_phase0.bin"
        default_manifest = root / "data" / "goldbach_manifest_v2_phase0.json"
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
