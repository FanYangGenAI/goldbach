"""Bucket sampling helpers."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from typing import Callable

from constants import BUCKET_BOUNDS


def bucket_range(bucket_id: int) -> tuple[int, int]:
    for current_id, low, high in BUCKET_BOUNDS:
        if current_id == bucket_id:
            return low, high
    raise ValueError(f"Unknown bucket id: {bucket_id}")


def enumerate_evens(low: int, high: int) -> Iterator[int]:
    start = low if low % 2 == 0 else low + 1
    for value in range(start, high, 2):
        yield value


def _round_to_even_in_range(value: float, low: int, high: int) -> int | None:
    candidate = int(round(value))
    if candidate < low:
        candidate = low if low % 2 == 0 else low + 1
    if candidate >= high:
        candidate = high - 1 if (high - 1) % 2 == 0 else high - 2
    if candidate % 2 == 1:
        if candidate + 1 < high:
            candidate += 1
        elif candidate - 1 >= low:
            candidate -= 1
        else:
            return None
    if candidate < low or candidate >= high:
        return None
    return candidate


def sample_even_log_uniform(
    low: int,
    high: int,
    count: int,
    rng: random.Random,
    seen: set[int] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[int]:
    if count <= 0:
        return []

    local_seen: set[int] = set() if seen is None else seen
    results: list[int] = []
    log_low = math.log(low)
    log_high = math.log(high)
    attempts = 0
    max_attempts = max(count * 200, 10_000)

    while len(results) < count and attempts < max_attempts:
        attempts += 1
        draw = math.exp(rng.uniform(log_low, log_high))
        candidate = _round_to_even_in_range(draw, low, high)
        if candidate is None or candidate in local_seen:
            continue
        local_seen.add(candidate)
        results.append(candidate)
        if progress is not None:
            progress(len(results), count)

    if len(results) < count:
        raise RuntimeError(
            f"Failed to sample {count} unique evens in [{low}, {high}); got {len(results)}"
        )
    return results


def collect_bucket_samples(
    bucket_id: int,
    target_count: int,
    rng: random.Random,
    strategy: str,
) -> list[int]:
    low, high = bucket_range(bucket_id)
    if strategy == "enumerate":
        values = list(enumerate_evens(low, high))
        if len(values) != target_count:
            raise RuntimeError(
                f"Bucket {bucket_id} enumerate mismatch: expected {target_count}, got {len(values)}"
            )
        return values
    if strategy == "log_uniform":
        return sample_even_log_uniform(low, high, target_count, rng)
    raise ValueError(f"Unknown strategy: {strategy}")
