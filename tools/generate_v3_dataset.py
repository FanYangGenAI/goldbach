#!/usr/bin/env python3
"""Generate IGOT v3 train/val/test datasets (WP-N N-1)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_GENERATOR_PYTHON = _ROOT / "generator" / "python"
if str(_GENERATOR_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GENERATOR_PYTHON))

from constants import (  # noqa: E402
    CANDIDATE_PRIME_COUNT,
    MIN_TRAINING_EVEN,
    SAMPLE_SIZE_V2,
    VERSION_V3,
    V3_ASSIGNMENT_SEED,
    V3_SPLIT_COUNTS,
    V3_SPLIT_IDS,
    V3_SPLIT_NAMES,
    V3_TOTAL,
)
from format_io import build_header  # noqa: E402
from generate import _chunk_list, _git_commit, _sha256_file  # noqa: E402
from v3_splits import (  # noqa: E402
    build_v3_bucket_targets,
    collect_v3_split_evens,
    compute_dedup_hash,
)
from worker import label_even_chunk_v2  # noqa: E402


def _label_split_to_file(
    output_path: Path,
    even_numbers: list[int],
    *,
    split_name: str,
    workers: int,
    chunk_size: int,
    progress_every: int,
) -> None:
    dedup_hash = compute_dedup_hash(even_numbers)
    split_id = V3_SPLIT_IDS[split_name]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    total = len(even_numbers)
    written = 0
    errors: list[str] = []

    with output_path.open("wb") as handle:
        handle.write(
            build_header(0, version=VERSION_V3, split_id=split_id, dedup_hash=dedup_hash)
        )
        chunks = _chunk_list(even_numbers, chunk_size)
        if workers <= 1:
            for chunk_index, chunk in enumerate(chunks, start=1):
                payload, chunk_errors = label_even_chunk_v2(chunk)
                handle.write(payload)
                errors.extend(chunk_errors)
                written += len(payload) // SAMPLE_SIZE_V2
                if written % progress_every < chunk_size or written == total:
                    elapsed = time.perf_counter() - started
                    rate = written / elapsed if elapsed > 0 else 0.0
                    print(f"  [{split_name}] {written:,}/{total:,} ({rate:,.0f}/s)")
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for chunk_index, (payload, chunk_errors) in enumerate(
                    pool.map(label_even_chunk_v2, chunks),
                    start=1,
                ):
                    handle.write(payload)
                    errors.extend(chunk_errors)
                    written += len(payload) // SAMPLE_SIZE_V2
                    if written % progress_every < chunk_size or written == total:
                        elapsed = time.perf_counter() - started
                        rate = written / elapsed if elapsed > 0 else 0.0
                        print(f"  [{split_name}] {written:,}/{total:,} ({rate:,.0f}/s)")

    if errors:
        error_log = output_path.with_suffix(".errors.log")
        error_log.write_text("\n".join(errors), encoding="utf-8")
        raise RuntimeError(f"Labeling failed for {len(errors)} evens; see {error_log}")
    if written != total:
        raise RuntimeError(f"Write count mismatch for {split_name}: {written} != {total}")

    with output_path.open("r+b") as handle:
        handle.seek(0)
        handle.write(
            build_header(
                written,
                version=VERSION_V3,
                split_id=split_id,
                dedup_hash=dedup_hash,
            )
        )


def generate_v3(
    *,
    output_dir: Path,
    seed: int,
    workers: int,
    chunk_size: int,
    progress_every: int,
    total_samples: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"Planning v3 splits ({total_samples:,} total samples)...")
    bucket_targets = build_v3_bucket_targets(total_samples)
    split_evens = collect_v3_split_evens(
        total_samples=total_samples,
        seed=seed,
        assignment_seed=V3_ASSIGNMENT_SEED,
    )

    split_files: dict[str, str] = {}
    split_records: list[dict] = []
    for split_name in V3_SPLIT_NAMES:
        file_name = f"goldbach_train_v3_{split_name}.bin"
        output_path = output_dir / file_name
        print(f"Labeling split={split_name} ({len(split_evens[split_name]):,} samples)...")
        _label_split_to_file(
            output_path,
            split_evens[split_name],
            split_name=split_name,
            workers=workers,
            chunk_size=chunk_size,
            progress_every=progress_every,
        )
        split_files[split_name] = file_name
        split_records.append(
            {
                "split": split_name,
                "split_id": V3_SPLIT_IDS[split_name],
                "sample_count": len(split_evens[split_name]),
                "dedup_hash": compute_dedup_hash(split_evens[split_name]),
                "sha256": _sha256_file(output_path),
                "file": file_name,
            }
        )

    manifest = {
        "format": "v3",
        "version": VERSION_V3,
        "total_samples": total_samples,
        "bucket_targets": bucket_targets,
        "splits": split_records,
        "seed": seed,
        "assignment_seed": V3_ASSIGNMENT_SEED,
        "workers": workers,
        "chunk_size": chunk_size,
        "candidate_prime_count": CANDIDATE_PRIME_COUNT,
        "min_training_even": MIN_TRAINING_EVEN,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_commit": _git_commit(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_path = output_dir / "goldbach_manifest_v3.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print()
    print("v3 generation complete.")
    print(f"  manifest: {manifest_path}")
    for record in split_records:
        print(f"  {record['split']}: {record['sample_count']:,} -> {record['file']}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IGOT v3 train/val/test datasets")
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
    )
    parser.add_argument("--chunk-size", type=int, default=5_000)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument(
        "--total-samples",
        type=int,
        default=V3_TOTAL,
        help=f"Total samples across splits (default {V3_TOTAL:,}; use small value for smoke)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Shortcut for --total-samples 1000",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total_samples = 1_000 if args.smoke else args.total_samples
    if total_samples < 100:
        print("ERROR: total-samples must be >= 100", file=sys.stderr)
        return 1
    try:
        generate_v3(
            output_dir=args.output_dir,
            seed=args.seed,
            workers=args.workers,
            chunk_size=args.chunk_size,
            progress_every=args.progress_every,
            total_samples=total_samples,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
