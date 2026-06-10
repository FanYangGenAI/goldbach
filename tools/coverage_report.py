#!/usr/bin/env python3
"""Compute IGOT candidate vocabulary coverage statistics."""

from __future__ import annotations

import argparse
import json
import random
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
    HEADER_SIZE,
    MIN_TRAINING_EVEN,
    PHASE0_TOTAL,
)
from coverage import (  # noqa: E402
    analyze_even,
    build_coverage_report,
    collect_phase0_evens,
    select_uncovered_examples,
)
from format_io import (  # noqa: E402
    GoldbachSample,
    GoldbachSampleV2,
    sample_size_for_version,
    verify_header,
)
from primes import first_k_odd_primes  # noqa: E402


def _parse_k_values(raw: str | None, k: int) -> list[int]:
    if raw is None:
        return [k]
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("k-sweep must contain at least one integer")
    return values


def _chunk_list(items: list[int], chunk_size: int) -> list[list[int]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _analyze_chunk(args: tuple[list[int], int]) -> list:
    evens, k = args
    candidates = first_k_odd_primes(k)
    return [analyze_even(even_number, candidates) for even_number in evens]


def _analyze_evens_parallel(
    even_numbers: list[int],
    k: int,
    workers: int,
    chunk_size: int,
) -> list:
    if workers <= 1 or len(even_numbers) <= chunk_size:
        return _analyze_chunk((even_numbers, k))

    chunks = _chunk_list(even_numbers, chunk_size)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk_rows in pool.map(_analyze_chunk, [(chunk, k) for chunk in chunks]):
            results.extend(chunk_rows)
    return results


def _load_evens_from_dataset(
    data_path: Path,
    sample_limit: int | None,
    seed: int,
) -> list[int]:
    with data_path.open("rb") as handle:
        header = verify_header(handle.read(HEADER_SIZE))
        sample_size = sample_size_for_version(header.version)
        indices = list(range(header.sample_count))
        if sample_limit is not None and sample_limit < header.sample_count:
            rng = random.Random(seed)
            indices = rng.sample(indices, sample_limit)

        evens: list[int] = []
        for index in indices:
            handle.seek(HEADER_SIZE + index * sample_size)
            payload = handle.read(sample_size)
            if header.version == 1:
                evens.append(GoldbachSample.from_bytes(payload).even_number)
            else:
                evens.append(GoldbachSampleV2.from_bytes(payload).even_number)
    return evens


def _collect_random_evens(phase: str, seed: int) -> tuple[list[int], str]:
    if phase != "phase0":
        raise ValueError(f"Unsupported random phase: {phase}")
    pairs = collect_phase0_evens(seed)
    evens = [even_number for even_number, _bucket_id in pairs]
    return evens, f"phase0_random:{PHASE0_TOTAL}"


def run_coverage_report(args: argparse.Namespace) -> dict:
    if args.mode == "random":
        even_numbers, source = _collect_random_evens(args.phase, args.seed)
    elif args.mode == "dataset":
        if args.data is None:
            raise ValueError("--data is required for dataset mode")
        even_numbers = _load_evens_from_dataset(args.data, args.sample_size, args.seed)
        source = str(args.data)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    if args.max_samples is not None and args.max_samples < len(even_numbers):
        rng = random.Random(args.seed)
        even_numbers = rng.sample(even_numbers, args.max_samples)

    even_numbers = [even for even in even_numbers if even >= MIN_TRAINING_EVEN]

    k_values = _parse_k_values(args.k_sweep, args.k)
    started = time.perf_counter()
    reports: dict[int, dict] = {}

    for k in k_values:
        rows = _analyze_evens_parallel(
            even_numbers,
            k,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )
        report = build_coverage_report(
            rows,
            k=k,
            seed=args.seed,
            mode=args.mode,
            source=source,
        )
        report["uncovered_samples"] = select_uncovered_examples(
            rows,
            limit=args.max_uncovered_examples,
        )
        reports[k] = report

    elapsed = time.perf_counter() - started
    if len(k_values) == 1:
        payload = reports[k_values[0]]
    else:
        payload = {
            "mode": args.mode,
            "source": source,
            "seed": args.seed,
            "k_sweep": reports,
        }
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["elapsed_seconds"] = round(elapsed, 3)
    payload["sample_count"] = len(even_numbers)
    return payload


def _default_output_path(root: Path, k_values: list[int]) -> Path:
    if len(k_values) == 1:
        return root / "data" / f"coverage_report_k{k_values[0]}.json"
    return root / "data" / "coverage_report_k_sweep.json"


def _print_summary(payload: dict) -> None:
    if "k_sweep" in payload:
        print("Coverage sweep summary:")
        for k in sorted(payload["k_sweep"]):
            summary = payload["k_sweep"][k]["global"]
            print(
                f"  k={k:4d}  coverage={summary['coverage_at_k']:.6f}  "
                f"uncovered={summary['uncovered_count']:,}/{summary['sample_count']:,}"
            )
        return

    summary = payload["global"]
    print(f"Coverage@{payload['k']}: {summary['coverage_at_k']:.6f}")
    print(
        f"Samples: {summary['sample_count']:,}  "
        f"covered: {summary['covered_count']:,}  "
        f"uncovered: {summary['uncovered_count']:,}"
    )
    print(f"Valid count mean: {summary['valid_count']['mean']:.3f}")
    rank_mean = summary["min_valid_rank"]["mean"]
    if rank_mean is not None:
        print(f"Min valid rank mean: {rank_mean:.3f}")
    print("Per-bucket coverage:")
    for bucket in payload["buckets"]:
        print(
            f"  bucket {bucket['id']:2d}  "
            f"coverage={bucket['coverage_at_k']:.6f}  "
            f"samples={bucket['sample_count']:,}"
        )
    if payload.get("uncovered_samples"):
        print("Uncovered examples:")
        for example in payload["uncovered_samples"][:5]:
            print(f"  N={example['even_number']}  bucket={example['bucket_id']}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Compute IGOT coverage@K statistics")
    parser.add_argument(
        "--mode",
        choices=("random", "dataset"),
        default="random",
        help="Sample source: Phase 0 random evens or existing dataset",
    )
    parser.add_argument(
        "--phase",
        choices=("phase0",),
        default="phase0",
        help="Random sampling profile (random mode only)",
    )
    parser.add_argument("--data", type=Path, default=None, help="Dataset .bin path")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Random subset size when reading a dataset",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap total evens analyzed (after collection/load)",
    )
    parser.add_argument("--k", type=int, default=CANDIDATE_PRIME_COUNT, help="Candidate count")
    parser.add_argument(
        "--k-sweep",
        type=str,
        default=None,
        help="Comma-separated K values, e.g. 500,1000,2000,5000",
    )
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    parser.add_argument("--chunk-size", type=int, default=256, help="Evens per worker chunk")
    parser.add_argument(
        "--max-uncovered-examples",
        type=int,
        default=20,
        help="Number of uncovered N examples to include in JSON",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    k_values = _parse_k_values(args.k_sweep, args.k)

    try:
        payload = run_coverage_report(args)
    except (ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    output_path = args.output or _default_output_path(root, k_values)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_summary(payload)
    print(f"Report: {output_path}")
    print(f"Elapsed: {payload['elapsed_seconds']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
