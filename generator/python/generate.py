#!/usr/bin/env python3
"""Goldbach dataset generator (Phase 0 / 1 / 2)."""

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

from buckets import bucket_range, collect_bucket_samples, sample_even_log_uniform  # noqa: F401
from constants import (
    CANDIDATE_PRIME_COUNT,
    MIN_TRAINING_EVEN,
    PHASE0_BUCKET_TARGETS,
    PHASE0_TOTAL,
    PHASE1_BUCKET_TARGETS,
    PHASE1_TOTAL,
    PHASE2_BUCKET_TARGETS,
    PHASE2_HARD_CASE_COUNT,
    PHASE2_TOTAL,
    SAMPLE_SIZE,
    SAMPLE_SIZE_V2,
    VERSION,
    VERSION_V2,
)
from format_io import build_header, make_sample, make_sample_v2
from hard_cases import collect_bucket18_evens
from labeling import find_min_p1
from worker import label_even_chunk, label_even_chunk_v2


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strategy_for_phase0_bucket(bucket_id: int) -> str:
    return "enumerate" if bucket_id <= 3 else "log_uniform"


def _bucket_strategy(phase: str, bucket_id: int) -> str:
    if phase == "phase0":
        return _strategy_for_phase0_bucket(bucket_id)
    if phase == "phase1":
        return "enumerate"
    if phase == "phase2":
        return "enumerate" if bucket_id <= 6 else "log_uniform"
    raise ValueError(f"Unknown phase: {phase}")


def _phase_targets(phase: str) -> dict[int, int]:
    if phase == "phase0":
        return PHASE0_BUCKET_TARGETS
    if phase == "phase1":
        return PHASE1_BUCKET_TARGETS
    if phase == "phase2":
        return PHASE2_BUCKET_TARGETS
    raise ValueError(f"Unknown phase: {phase}")


def _prepare_v2_bucket_evens(
    evens: list[int],
    bucket_id: int,
    target: int,
    seed: int,
) -> list[int]:
    """Drop evens below MIN_TRAINING_EVEN; top up when the bucket range has spare evens."""
    import random

    filtered = [even for even in evens if even >= MIN_TRAINING_EVEN]
    if len(filtered) == target:
        return filtered
    if len(filtered) > target:
        raise RuntimeError(
            f"Bucket {bucket_id}: got {len(filtered)} evens after filtering, expected {target}"
        )

    low, high = bucket_range(bucket_id)
    rng = random.Random(seed + bucket_id * 9_999_983)
    seen = set(filtered)
    need = target - len(filtered)
    low_bound = max(low, MIN_TRAINING_EVEN)
    try:
        extras = sample_even_log_uniform(low_bound, high, need, rng, seen=seen)
    except RuntimeError:
        return filtered
    filtered.extend(extra for extra in extras if extra >= MIN_TRAINING_EVEN)
    return filtered


def _expected_sample_total(phase: str, dataset_version: int) -> int:
    totals = {
        "phase0": PHASE0_TOTAL,
        "phase1": PHASE1_TOTAL,
        "phase2": PHASE2_TOTAL,
    }
    total = totals[phase]
    if phase == "phase0" and dataset_version == VERSION_V2:
        # Bucket 1 enumerates every even in [4, 100); removing N=4 frees no replacement slot.
        return total - 1
    return total


def _collect_bucket_evens_for_phase(
    phase: str,
    bucket_id: int,
    target: int,
    seed: int,
) -> list[int]:
    import random

    low, high = bucket_range(bucket_id)
    strategy = _bucket_strategy(phase, bucket_id)
    if phase == "phase2" and bucket_id == 18:
        return collect_bucket18_evens(target, seed)
    rng = random.Random(seed + bucket_id * 1_000_003)
    return collect_bucket_samples(bucket_id, target, rng, strategy)


def _chunk_list(items: list[int], chunk_size: int) -> list[list[int]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _label_sequential_to_handle(
    handle,
    evens: list[int],
    progress_every: int,
    processed_offset: int,
    total_target: int,
    started: float,
    *,
    dataset_version: int,
) -> tuple[int, list[str]]:
    errors: list[str] = []
    written = 0
    for index, even_number in enumerate(evens, start=1):
        try:
            if dataset_version == VERSION_V2:
                payload = make_sample_v2(even_number).to_bytes()
            else:
                p1 = find_min_p1(even_number)
                payload = make_sample(even_number, p1).to_bytes()
            handle.write(payload)
            written += 1
        except (RuntimeError, ValueError) as exc:
            errors.append(f"even={even_number}: {exc}")

        global_index = processed_offset + index
        if global_index % progress_every == 0 or global_index == total_target:
            elapsed = time.perf_counter() - started
            rate = global_index / elapsed if elapsed > 0 else 0.0
            print(f"  processed {global_index:,}/{total_target:,} ({rate:,.0f}/s)")

    return written, errors


def _label_parallel_to_handle(
    handle,
    evens: list[int],
    workers: int,
    chunk_size: int,
    progress_every: int,
    processed_offset: int,
    total_target: int,
    started: float,
    *,
    dataset_version: int,
) -> tuple[int, list[str]]:
    sample_size = SAMPLE_SIZE_V2 if dataset_version == VERSION_V2 else SAMPLE_SIZE
    label_chunk = label_even_chunk_v2 if dataset_version == VERSION_V2 else label_even_chunk
    chunks = _chunk_list(evens, chunk_size)
    errors: list[str] = []
    written = 0
    processed = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk_index, (payload, chunk_errors) in enumerate(pool.map(label_chunk, chunks), start=1):
            handle.write(payload)
            errors.extend(chunk_errors)
            processed += len(chunks[chunk_index - 1])
            written += len(payload) // sample_size
            global_index = processed_offset + processed
            if global_index % progress_every < chunk_size or global_index == total_target:
                elapsed = time.perf_counter() - started
                rate = global_index / elapsed if elapsed > 0 else 0.0
                print(f"  processed {global_index:,}/{total_target:,} ({rate:,.0f}/s)")

    return written, errors


def generate_dataset(
    phase: str,
    output_path: Path,
    manifest_path: Path,
    seed: int,
    workers: int,
    chunk_size: int,
    progress_every: int,
    dataset_version: int = VERSION,
) -> None:
    targets = _phase_targets(phase)
    expected_total = _expected_sample_total(phase, dataset_version)
    nominal_total = {
        "phase0": PHASE0_TOTAL,
        "phase1": PHASE1_TOTAL,
        "phase2": PHASE2_TOTAL,
    }[phase]
    if sum(targets.values()) != nominal_total:
        raise RuntimeError(f"{phase} bucket targets do not sum to expected total")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_records: list[dict] = []
    started = time.perf_counter()
    total_written = 0
    all_errors: list[str] = []

    format_label = "v2" if dataset_version == VERSION_V2 else "v1"
    use_parallel_workers = workers > 1 and (
        phase in {"phase1", "phase2"} or dataset_version == VERSION_V2
    )
    print(f"Generating {phase} {format_label} dataset ({expected_total:,} samples)...")

    with output_path.open("wb") as handle:
        handle.write(build_header(0, version=dataset_version))

        for bucket_id in sorted(targets):
            target = targets[bucket_id]
            low, high = bucket_range(bucket_id)
            strategy = _bucket_strategy(phase, bucket_id)
            print(f"Bucket {bucket_id:2d}: {strategy:12s} target={target:,}")

            evens = _collect_bucket_evens_for_phase(phase, bucket_id, target, seed)
            if dataset_version == VERSION_V2:
                evens = _prepare_v2_bucket_evens(evens, bucket_id, target, seed)
            bucket_records.append(
                {"id": bucket_id, "range": [low, high], "count": len(evens), "strategy": strategy}
            )

            if use_parallel_workers:
                written, errors = _label_parallel_to_handle(
                    handle,
                    evens,
                    workers,
                    chunk_size,
                    progress_every,
                    total_written,
                    expected_total,
                    started,
                    dataset_version=dataset_version,
                )
            else:
                written, errors = _label_sequential_to_handle(
                    handle,
                    evens,
                    progress_every,
                    total_written,
                    expected_total,
                    started,
                    dataset_version=dataset_version,
                )

            total_written += written
            all_errors.extend(errors)

    if all_errors:
        error_log = output_path.with_suffix(".errors.log")
        error_log.write_text("\n".join(all_errors), encoding="utf-8")
        raise RuntimeError(f"Labeling failed for {len(all_errors)} evens; see {error_log}")

    if total_written != expected_total:
        raise RuntimeError(
            f"Write count mismatch: wrote {total_written}, expected {expected_total} "
            f"(nominal phase target={nominal_total})"
        )

    with output_path.open("r+b") as handle:
        handle.seek(0)
        handle.write(build_header(total_written, version=dataset_version))

    digest = _sha256_file(output_path)
    sample_size = SAMPLE_SIZE_V2 if dataset_version == VERSION_V2 else SAMPLE_SIZE
    manifest = {
        "phase": phase,
        "version": dataset_version,
        "sample_count": total_written,
        "sample_size_bytes": sample_size,
        "file": output_path.name,
        "sha256": digest,
        "seed": seed,
        "workers": workers if use_parallel_workers else 1,
        "chunk_size": chunk_size if use_parallel_workers else None,
        "buckets": bucket_records,
        "hard_case_count": PHASE2_HARD_CASE_COUNT if phase == "phase2" else 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_commit": _git_commit(),
        "payload_offset": 64,
    }
    if dataset_version == VERSION_V2:
        manifest["candidate_prime_count"] = CANDIDATE_PRIME_COUNT
        manifest["min_training_even"] = MIN_TRAINING_EVEN
        manifest["coverage_report"] = "data/coverage_report_phase0.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - started
    file_size = output_path.stat().st_size
    print()
    print(f"{phase} generation complete.")
    print(f"  output:   {output_path}")
    print(f"  manifest: {manifest_path}")
    print(f"  samples:  {total_written:,}")
    print(f"  size:     {file_size:,} bytes ({file_size / (1024 * 1024):.2f} MiB)")
    print(f"  sha256:   {digest}")
    print(f"  elapsed:  {elapsed:.2f}s")


def _default_paths(
    phase: str,
    root: Path,
    dataset_version: int,
) -> tuple[Path, Path]:
    if phase == "phase0" and dataset_version == VERSION_V2:
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
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Generate Goldbach dataset")
    parser.add_argument(
        "--phase",
        choices=("phase0", "phase1", "phase2"),
        default="phase2",
        help="Generation phase (default: phase2)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output .bin path")
    parser.add_argument("--manifest", type=Path, default=None, help="Output manifest JSON path")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Worker processes for parallel labeling",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Evens per worker chunk",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Print progress every N samples",
    )
    parser.add_argument(
        "--format",
        choices=("v1", "v2"),
        default="v1",
        help="Dataset format: v1 (p1_label) or v2 (valid_mask)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_version = VERSION_V2 if args.format == "v2" else VERSION
    default_output, default_manifest = _default_paths(args.phase, _repo_root(), dataset_version)
    output_path = args.output or default_output
    manifest_path = args.manifest or default_manifest
    progress_every = args.progress_every
    if progress_every is None:
        progress_every = 10_000 if args.phase == "phase0" else 500_000
    if args.format == "v2" and args.phase != "phase0":
        print("ERROR: v2 format is currently supported for phase0 only", file=sys.stderr)
        return 1

    try:
        generate_dataset(
            phase=args.phase,
            output_path=output_path,
            manifest_path=manifest_path,
            seed=args.seed,
            workers=args.workers,
            chunk_size=args.chunk_size,
            progress_every=progress_every,
            dataset_version=dataset_version,
        )
    except Exception as exc:  # noqa: BLE001 - CLI entrypoint
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
