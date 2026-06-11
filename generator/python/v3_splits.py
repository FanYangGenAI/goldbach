"""v3 dataset bucket targets and deterministic N -> split assignment."""

from __future__ import annotations

import hashlib
import random
from buckets import bucket_range, enumerate_evens, sample_even_log_uniform
from constants import (
    MIN_TRAINING_EVEN,
    PHASE0_BUCKET_TARGETS,
    PHASE0_TOTAL,
    V3_ASSIGNMENT_SEED,
    V3_SPLIT_COUNTS,
    V3_SPLIT_NAMES,
    V3_SPLIT_RATIOS,
    V3_TOTAL,
)


def count_unique_evens_in_bucket(bucket_id: int, min_even: int = MIN_TRAINING_EVEN) -> int:
    low, high = bucket_range(bucket_id)
    low_bound = max(low, min_even)
    if low_bound >= high:
        return 0
    span = high - low_bound
    if span > 20_000_000:
        return span // 2
    return sum(1 for _ in enumerate_evens(low_bound, high))


def build_v3_bucket_targets(total_samples: int = V3_TOTAL) -> dict[int, int]:
    """Scale phase0 bucket mix to ``total_samples``, respecting finite bucket capacity."""
    scale = total_samples / PHASE0_TOTAL
    caps = {
        bucket_id: count_unique_evens_in_bucket(bucket_id)
        for bucket_id in PHASE0_BUCKET_TARGETS
    }
    targets: dict[int, int] = {}
    for bucket_id, base_count in PHASE0_BUCKET_TARGETS.items():
        want = int(round(base_count * scale))
        targets[bucket_id] = min(want, caps[bucket_id])

    deficit = total_samples - sum(targets.values())
    if deficit > 0:
        receivers = [
            bucket_id
            for bucket_id in targets
            if targets[bucket_id] < caps[bucket_id]
        ]
        if not receivers:
            raise RuntimeError(
                f"Cannot reach {total_samples} samples; capped total={sum(targets.values())}"
            )
        weight_sum = sum(PHASE0_BUCKET_TARGETS[bucket_id] for bucket_id in receivers)
        allocated = 0
        for bucket_id in receivers:
            room = caps[bucket_id] - targets[bucket_id]
            extra = min(room, int(round(deficit * PHASE0_BUCKET_TARGETS[bucket_id] / weight_sum)))
            targets[bucket_id] += extra
            allocated += extra
        remainder = deficit - allocated
        order = sorted(
            receivers,
            key=lambda bucket_id: PHASE0_BUCKET_TARGETS[bucket_id],
            reverse=True,
        )
        for bucket_id in order:
            if remainder <= 0:
                break
            room = caps[bucket_id] - targets[bucket_id]
            add = min(room, remainder)
            targets[bucket_id] += add
            remainder -= add
        if remainder > 0:
            raise RuntimeError(f"Could not allocate {remainder} samples after redistribution")

    surplus = sum(targets.values()) - total_samples
    if surplus > 0:
        order = sorted(targets, key=lambda bucket_id: targets[bucket_id], reverse=True)
        for bucket_id in order:
            if surplus <= 0:
                break
            removable = targets[bucket_id] - 1
            remove = min(removable, surplus)
            targets[bucket_id] -= remove
            surplus -= remove

    if sum(targets.values()) != total_samples:
        raise RuntimeError(
            f"v3 bucket targets sum to {sum(targets.values())}, expected {total_samples}"
        )
    return targets


def allocate_bucket_counts_to_splits(
    bucket_counts: dict[int, int],
    *,
    assignment_seed: int = V3_ASSIGNMENT_SEED,
) -> dict[str, dict[int, int]]:
    """Largest-remainder allocation of each bucket count across train/val/test."""
    split_targets: dict[str, dict[int, int]] = {name: {} for name in V3_SPLIT_NAMES}
    for bucket_id, count in bucket_counts.items():
        if bucket_id <= 3:
            low, high = bucket_range(bucket_id)
            low_bound = max(low, MIN_TRAINING_EVEN)
            pool = [even for even in enumerate_evens(low_bound, high) if even >= MIN_TRAINING_EVEN]
            hash_counts = {name: 0 for name in V3_SPLIT_NAMES}
            for even_number in pool:
                hash_counts[assign_split_for_even(even_number, assignment_seed)] += 1
            split_targets_for_bucket = _largest_remainder_allocate(
                count,
                hash_counts,
            )
            for name in V3_SPLIT_NAMES:
                split_targets[name][bucket_id] = split_targets_for_bucket[name]
            continue

        raw = {name: count * V3_SPLIT_RATIOS[name] for name in V3_SPLIT_NAMES}
        floors = {name: int(raw[name]) for name in V3_SPLIT_NAMES}
        remainder = count - sum(floors.values())
        order = sorted(V3_SPLIT_NAMES, key=lambda name: raw[name] - floors[name], reverse=True)
        for index in range(remainder):
            floors[order[index % len(order)]] += 1
        for name in V3_SPLIT_NAMES:
            split_targets[name][bucket_id] = floors[name]
    return split_targets


def _largest_remainder_allocate(
    total: int,
    caps: dict[str, int],
) -> dict[str, int]:
    """Allocate ``total`` slots across splits without exceeding hash caps per bucket."""
    if total > sum(caps.values()):
        raise RuntimeError(f"Cannot allocate {total} samples with caps {caps}")
    raw = {name: total * V3_SPLIT_RATIOS[name] for name in V3_SPLIT_NAMES}
    floors = {name: min(int(raw[name]), caps[name]) for name in V3_SPLIT_NAMES}
    assigned = sum(floors.values())
    order = sorted(
        V3_SPLIT_NAMES,
        key=lambda name: raw[name] - int(raw[name]),
        reverse=True,
    )
    while assigned < total:
        progressed = False
        for name in order:
            if floors[name] < caps[name]:
                floors[name] += 1
                assigned += 1
                progressed = True
                if assigned >= total:
                    break
        if not progressed:
            raise RuntimeError(f"Hash caps exhausted while allocating {total}: {caps}")
    return floors


def assign_split_for_even(even_number: int, assignment_seed: int = V3_ASSIGNMENT_SEED) -> str:
    """Deterministic hash assignment: same N always maps to the same split."""
    digest = hashlib.sha256(f"{assignment_seed}:v3:{even_number}".encode()).digest()
    unit = int.from_bytes(digest[:4], "big") / 2**32
    cumulative = 0.0
    for name in V3_SPLIT_NAMES:
        cumulative += V3_SPLIT_RATIOS[name]
        if unit < cumulative:
            return name
    return V3_SPLIT_NAMES[-1]


def collect_v3_split_evens(
    *,
    total_samples: int,
    seed: int,
    assignment_seed: int = V3_ASSIGNMENT_SEED,
) -> dict[str, list[int]]:
    """Sample unique evens per split with per-bucket quotas and hash-based split routing."""
    bucket_targets = build_v3_bucket_targets(total_samples)
    per_split_bucket = allocate_bucket_counts_to_splits(
        bucket_targets,
        assignment_seed=assignment_seed,
    )
    expected_split_counts = {
        name: sum(per_split_bucket[name].values()) for name in V3_SPLIT_NAMES
    }
    bucket_remaining: dict[str, dict[int, int]] = {
        name: dict(per_split_bucket[name]) for name in V3_SPLIT_NAMES
    }
    split_remaining = dict(expected_split_counts)
    split_evens: dict[str, list[int]] = {name: [] for name in V3_SPLIT_NAMES}
    global_seen: set[int] = set()

    for bucket_id in sorted(bucket_targets):
        rng = random.Random(seed + bucket_id * 1_000_003)
        low, high = bucket_range(bucket_id)
        low_bound = max(low, MIN_TRAINING_EVEN)
        need_total = sum(bucket_remaining[name][bucket_id] for name in V3_SPLIT_NAMES)
        if need_total == 0:
            continue
        bucket_cap = count_unique_evens_in_bucket(bucket_id)

        if need_total >= bucket_cap:
            pool = sample_even_log_uniform(
                low_bound,
                high,
                bucket_cap,
                rng,
                seen=global_seen,
            )
            assigned_counts = {name: 0 for name in V3_SPLIT_NAMES}
            rng.shuffle(pool)
            for even_number in pool:
                split_name = assign_split_for_even(even_number, assignment_seed)
                global_seen.add(even_number)
                split_evens[split_name].append(even_number)
                assigned_counts[split_name] += 1
            for split_name in V3_SPLIT_NAMES:
                split_remaining[split_name] -= assigned_counts[split_name]
                bucket_remaining[split_name][bucket_id] = 0
            continue

        if bucket_id <= 3:
            pool = [
                even for even in enumerate_evens(low_bound, high) if even >= MIN_TRAINING_EVEN
            ]
            rng.shuffle(pool)
            by_preferred = {name: [] for name in V3_SPLIT_NAMES}
            for even_number in pool:
                by_preferred[assign_split_for_even(even_number, assignment_seed)].append(
                    even_number
                )
            for split_name in V3_SPLIT_NAMES:
                need = bucket_remaining[split_name][bucket_id]
                if need == 0:
                    continue
                available = by_preferred[split_name]
                if len(available) < need:
                    raise RuntimeError(
                        f"Bucket {bucket_id}: split={split_name} needs {need} evens, "
                        f"hash-assigned pool has {len(available)}"
                    )
                for even_number in available[:need]:
                    global_seen.add(even_number)
                    split_evens[split_name].append(even_number)
                    bucket_remaining[split_name][bucket_id] -= 1
                    split_remaining[split_name] -= 1
            remaining = sum(bucket_remaining[name][bucket_id] for name in V3_SPLIT_NAMES)
            if remaining > 0:
                raise RuntimeError(
                    f"Bucket {bucket_id}: could not fill {remaining} remaining slots"
                )
            continue

        candidates: list[int] = []
        extra = max(64, need_total // 20)
        while True:
            remaining_in_bucket = bucket_cap - len(candidates)
            if remaining_in_bucket <= 0:
                break
            batch_size = min(
                max(need_total + extra - len(candidates), extra),
                remaining_in_bucket,
            )
            batch = sample_even_log_uniform(
                low_bound,
                high,
                batch_size,
                rng,
                seen=global_seen,
            )
            candidates.extend(batch)
            by_preferred = {name: [] for name in V3_SPLIT_NAMES}
            for even_number in candidates:
                by_preferred[assign_split_for_even(even_number, assignment_seed)].append(
                    even_number
                )
            if all(
                len(by_preferred[name]) >= bucket_remaining[name][bucket_id]
                for name in V3_SPLIT_NAMES
                if bucket_remaining[name][bucket_id] > 0
            ):
                break
            if len(candidates) > need_total * 3:
                raise RuntimeError(
                    f"Bucket {bucket_id}: hash pool still insufficient after "
                    f"{len(candidates)} samples"
                )

        rng.shuffle(candidates)
        by_preferred: dict[str, list[int]] = {name: [] for name in V3_SPLIT_NAMES}
        for even_number in candidates:
            preferred = assign_split_for_even(even_number, assignment_seed)
            by_preferred[preferred].append(even_number)

        for split_name in V3_SPLIT_NAMES:
            need = bucket_remaining[split_name][bucket_id]
            if need == 0:
                continue
            available = by_preferred[split_name]
            if len(available) < need:
                raise RuntimeError(
                    f"Bucket {bucket_id}: split={split_name} needs {need} evens, "
                    f"hash-assigned pool has {len(available)} (increase oversample)"
                )
            for even_number in available[:need]:
                global_seen.add(even_number)
                split_evens[split_name].append(even_number)
                bucket_remaining[split_name][bucket_id] -= 1
                split_remaining[split_name] -= 1

        remaining = sum(bucket_remaining[name][bucket_id] for name in V3_SPLIT_NAMES)
        if remaining > 0:
            raise RuntimeError(
                f"Bucket {bucket_id}: could not fill {remaining} remaining slots"
            )

    total_assigned = sum(len(values) for values in split_evens.values())
    if total_assigned != total_samples:
        raise RuntimeError(
            f"v3 split total {total_assigned} != requested {total_samples}"
        )
    return split_evens


def compute_dedup_hash(even_numbers: list[int]) -> int:
    """Stable 64-bit fingerprint of sorted unique N values in a split file."""
    payload = ",".join(str(value) for value in sorted(set(even_numbers))).encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little")
