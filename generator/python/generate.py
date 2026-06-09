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

from buckets import bucket_range, collect_bucket_samples
from constants import (
    PHASE0_BUCKET_TARGETS,
    PHASE0_TOTAL,
    PHASE1_BUCKET_TARGETS,
    PHASE1_TOTAL,
    PHASE2_BUCKET_TARGETS,
    PHASE2_HARD_CASE_COUNT,
    PHASE2_TOTAL,
)
from format_io import build_header, make_sample
from constants import SAMPLE_SIZE
from hard_cases import collect_bucket18_evens
from labeling import find_min_p1
from worker import label_even_chunk


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
) -> tuple[int, list[str]]:
    errors: list[str] = []
    written = 0
    for index, even_number in enumerate(evens, start=1):
        try:
            p1 = find_min_p1(even_number)
            handle.write(make_sample(even_number, p1).to_bytes())
            written += 1
        except RuntimeError as exc:
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
) -> tuple[int, list[str]]:
    chunks = _chunk_list(evens, chunk_size)
    errors: list[str] = []
    written = 0
    processed = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk_index, (payload, chunk_errors) in enumerate(pool.map(label_even_chunk, chunks), start=1):
            handle.write(payload)
            errors.extend(chunk_errors)
            processed += len(chunks[chunk_index - 1])
            written += len(payload) // SAMPLE_SIZE
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
) -> None:
    targets = _phase_targets(phase)
    expected_total = {
        "phase0": PHASE0_TOTAL,
        "phase1": PHASE1_TOTAL,
        "phase2": PHASE2_TOTAL,
    }[phase]
    if sum(targets.values()) != expected_total:
        raise RuntimeError(f"{phase} bucket targets do not sum to expected total")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_records: list[dict] = []
    started = time.perf_counter()
    total_written = 0
    all_errors: list[str] = []

    print(f"Generating {phase} dataset ({expected_total:,} samples)...")

    with output_path.open("wb") as handle:
        handle.write(build_header(0))

        for bucket_id in sorted(targets):
            target = targets[bucket_id]
            low, high = bucket_range(bucket_id)
            strategy = _bucket_strategy(phase, bucket_id)
            print(f"Bucket {bucket_id:2d}: {strategy:12s} target={target:,}")

            evens = _collect_bucket_evens_for_phase(phase, bucket_id, target, seed)
            bucket_records.append(
                {"id": bucket_id, "range": [low, high], "count": len(evens), "strategy": strategy}
            )

            use_parallel = phase in {"phase1", "phase2"} and workers > 1
            if use_parallel:
                written, errors = _label_parallel_to_handle(
                    handle,
                    evens,
                    workers,
                    chunk_size,
                    progress_every,
                    total_written,
                    expected_total,
                    started,
                )
            else:
                written, errors = _label_sequential_to_handle(
                    handle,
                    evens,
                    progress_every,
                    total_written,
                    expected_total,
                    started,
                )

            total_written += written
            all_errors.extend(errors)

    if all_errors:
        error_log = output_path.with_suffix(".errors.log")
        error_log.write_text("\n".join(all_errors), encoding="utf-8")
        raise RuntimeError(f"Labeling failed for {len(all_errors)} evens; see {error_log}")

    if total_written != expected_total:
        raise RuntimeError(f"Write count mismatch: wrote {total_written}, expected {expected_total}")

    with output_path.open("r+b") as handle:
        handle.seek(0)
        handle.write(build_header(total_written))

    digest = _sha256_file(output_path)
    manifest = {
        "phase": phase,
        "version": 1,
        "sample_count": total_written,
        "file": output_path.name,
        "sha256": digest,
        "seed": seed,
        "workers": workers if phase in {"phase1", "phase2"} else 1,
        "chunk_size": chunk_size if phase in {"phase1", "phase2"} else None,
        "buckets": bucket_records,
        "hard_case_count": PHASE2_HARD_CASE_COUNT if phase == "phase2" else 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_commit": _git_commit(),
        "payload_offset": 64,
    }
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


def _default_paths(phase: str, root: Path) -> tuple[Path, Path]:
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
        default=500_000,
        help="Print progress every N samples",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_output, default_manifest = _default_paths(args.phase, _repo_root())
    output_path = args.output or default_output
    manifest_path = args.manifest or default_manifest

    try:
        generate_dataset(
            phase=args.phase,
            output_path=output_path,
            manifest_path=manifest_path,
            seed=args.seed,
            workers=args.workers,
            chunk_size=args.chunk_size,
            progress_every=args.progress_every,
        )
    except Exception as exc:  # noqa: BLE001 - CLI entrypoint
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
