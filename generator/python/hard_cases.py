"""Hard-case even sampling for bucket 18."""

from __future__ import annotations

import random

from buckets import bucket_range, sample_even_log_uniform


def generate_hard_cases(
    low: int,
    high: int,
    count: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed ^ 0x9E3779B97F4A7C15)
    seen: set[int] = set()
    values: list[int] = []

    step = 2 * 3 * 5 * 7 * 11  # 2310
    start = ((low + step - 1) // step) * step
    offset = 0

    while len(values) < count and offset < count * 500:
        candidate = start + offset * step
        offset += 1
        if candidate >= high:
            candidate = low + (rng.randrange(0, (high - low) // 2) * 2)
        if candidate < low or candidate >= high or candidate % 2 != 0:
            continue
        if (candidate // 2) % 1155 != 0 and offset % 3 != 0:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        values.append(candidate)

    if len(values) < count:
        extras = sample_even_log_uniform(low, high, count - len(values) + 1000, rng, seen)
        for value in extras:
            if value not in seen:
                seen.add(value)
                values.append(value)
            if len(values) >= count:
                break

    if len(values) < count:
        raise RuntimeError(f"Failed to generate {count} hard cases, got {len(values)}")
    return values[:count]


def collect_bucket18_evens(target_count: int, seed: int) -> list[int]:
    low, high = bucket_range(18)
    hard_count = min(100_000, target_count)
    hard = generate_hard_cases(low, high, hard_count, seed)
    seen = set(hard)
    evens = list(hard)

    if target_count > hard_count:
        rng = random.Random(seed + 180017)
        need = target_count - hard_count
        extras = sample_even_log_uniform(low, high, need, rng, seen)
        evens.extend(extras)

    if len(evens) != target_count:
        raise RuntimeError(f"Bucket 18 target mismatch: expected {target_count}, got {len(evens)}")
    return evens
