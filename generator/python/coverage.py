"""Coverage statistics for IGOT candidate vocabulary."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from buckets import bucket_range, collect_bucket_samples
from constants import BUCKET_BOUNDS, PHASE0_BUCKET_TARGETS
from labeling import compute_valid_mask
from primes import first_k_odd_primes


@dataclass(frozen=True)
class EvenCoverage:
    even_number: int
    bucket_id: int
    valid_count: int
    covered: bool
    min_valid_rank: int | None


def bucket_id_for_even(even_number: int) -> int:
    for bucket_id, low, high in BUCKET_BOUNDS:
        if low <= even_number < high:
            return bucket_id
    raise ValueError(f"even_number {even_number} is outside known bucket ranges")


def analyze_even(even_number: int, candidate_primes: Sequence[int]) -> EvenCoverage:
    mask = compute_valid_mask(even_number, candidate_primes)
    valid_count = sum(mask)
    min_valid_rank = next((index for index, value in enumerate(mask) if value), None)
    return EvenCoverage(
        even_number=even_number,
        bucket_id=bucket_id_for_even(even_number),
        valid_count=valid_count,
        covered=valid_count > 0,
        min_valid_rank=min_valid_rank,
    )


def analyze_evens(even_numbers: Iterable[int], k: int) -> list[EvenCoverage]:
    candidates = first_k_odd_primes(k)
    return [analyze_even(even_number, candidates) for even_number in even_numbers]


def _strategy_for_phase0_bucket(bucket_id: int) -> str:
    return "enumerate" if bucket_id <= 3 else "log_uniform"


def collect_phase0_evens(seed: int) -> list[tuple[int, int]]:
    """Return (even_number, bucket_id) pairs using Phase 0 bucket targets."""
    results: list[tuple[int, int]] = []
    for bucket_id in sorted(PHASE0_BUCKET_TARGETS):
        target = PHASE0_BUCKET_TARGETS[bucket_id]
        strategy = _strategy_for_phase0_bucket(bucket_id)
        rng = random.Random(seed + bucket_id * 1_000_003)
        evens = collect_bucket_samples(bucket_id, target, rng, strategy)
        results.extend((even_number, bucket_id) for even_number in evens)
    return results


def _histogram(values: Sequence[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _summarize_valid_count(rows: Sequence[EvenCoverage]) -> dict:
    counts = [row.valid_count for row in rows]
    if not counts:
        return {"mean": 0.0, "min": 0, "max": 0, "histogram": {}}
    return {
        "mean": sum(counts) / len(counts),
        "min": min(counts),
        "max": max(counts),
        "histogram": _histogram(counts),
    }


def _summarize_min_valid_rank(rows: Sequence[EvenCoverage]) -> dict:
    covered_rows = [row for row in rows if row.min_valid_rank is not None]
    ranks = [row.min_valid_rank for row in covered_rows if row.min_valid_rank is not None]
    if not ranks:
        return {"mean": None, "histogram": {}, "uncovered": len(rows)}
    return {
        "mean": sum(ranks) / len(ranks),
        "histogram": _histogram(ranks),
        "uncovered": len(rows) - len(covered_rows),
    }


def _summarize_rows(rows: Sequence[EvenCoverage]) -> dict:
    covered_count = sum(1 for row in rows if row.covered)
    sample_count = len(rows)
    coverage = covered_count / sample_count if sample_count else 0.0
    return {
        "sample_count": sample_count,
        "covered_count": covered_count,
        "uncovered_count": sample_count - covered_count,
        "coverage_at_k": coverage,
        "valid_count": _summarize_valid_count(rows),
        "min_valid_rank": _summarize_min_valid_rank(rows),
    }


def build_coverage_report(
    rows: Sequence[EvenCoverage],
    *,
    k: int,
    seed: int | None,
    mode: str,
    source: str,
) -> dict:
    global_summary = _summarize_rows(rows)
    buckets: list[dict] = []
    for bucket_id, low, high in BUCKET_BOUNDS:
        bucket_rows = [row for row in rows if row.bucket_id == bucket_id]
        if not bucket_rows:
            continue
        bucket_summary = _summarize_rows(bucket_rows)
        buckets.append(
            {
                "id": bucket_id,
                "range": [low, high],
                **bucket_summary,
            }
        )

    uncovered = [row for row in rows if not row.covered]
    return {
        "k": k,
        "mode": mode,
        "source": source,
        "seed": seed,
        "global": global_summary,
        "buckets": buckets,
        "uncovered_count": len(uncovered),
    }


def select_uncovered_examples(
    rows: Sequence[EvenCoverage],
    limit: int,
) -> list[dict]:
    examples: list[dict] = []
    for row in rows:
        if row.covered:
            continue
        examples.append(
            {
                "even_number": row.even_number,
                "bucket_id": row.bucket_id,
            }
        )
        if len(examples) >= limit:
            break
    return examples
