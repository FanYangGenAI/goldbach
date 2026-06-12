#!/usr/bin/env python3
"""Evaluate F-4 pre-registered bucket rules (section 2.1 three-tier test)."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


LARGE_BUCKETS = tuple(range(7, 19))
PHASES = ("phase1", "phase2", "phase3")
N_CMP = len(LARGE_BUCKETS) * len(PHASES)
SOFT_QUOTA = math.floor(0.10 * N_CMP)


@dataclass(frozen=True)
class BucketComparison:
    phase: str
    bucket: int
    top1_new: float
    top1_ref: float
    sample_count: int

    @property
    def delta(self) -> float:
        return self.top1_new - self.top1_ref

    @property
    def se(self) -> float:
        p = max(min((self.top1_new + self.top1_ref) / 2.0, 1.0 - 1e-12), 1e-12)
        n = max(self.sample_count, 1)
        return math.sqrt(p * (1.0 - p) / n)

    @property
    def abs_delta(self) -> float:
        return abs(self.delta)


def _binom_two_sided_pvalue(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    obs = min(successes, trials - successes)
    lower_tail = sum(
        math.comb(trials, k) * (0.5**trials) for k in range(obs + 1)
    )
    return min(1.0, 2.0 * lower_tail)


def sign_test_pvalue(deltas: list[float]) -> float:
    nonzero = [delta for delta in deltas if delta != 0.0]
    trials = len(nonzero)
    if trials == 0:
        return 1.0
    positives = sum(1 for delta in nonzero if delta > 0.0)
    return _binom_two_sided_pvalue(positives, trials)


def bucket_top1(metrics: dict, bucket: int) -> tuple[float, int]:
    bucket_metrics = metrics.get("by_bucket", {}).get(str(bucket), {})
    if not bucket_metrics:
        bucket_metrics = metrics.get("by_bucket", {}).get(bucket, {})
    top1 = float(bucket_metrics.get("top1_valid_rate", 0.0))
    count = int(bucket_metrics.get("sample_count", 0))
    return top1, count


def build_comparisons(
    new_by_phase: dict[str, dict],
    ref_by_phase: dict[str, dict],
) -> list[BucketComparison]:
    comparisons: list[BucketComparison] = []
    for phase in PHASES:
        new_metrics = new_by_phase[phase]
        ref_metrics = ref_by_phase[phase]
        for bucket in LARGE_BUCKETS:
            top1_new, _ = bucket_top1(new_metrics, bucket)
            top1_ref, sample_count = bucket_top1(ref_metrics, bucket)
            comparisons.append(
                BucketComparison(
                    phase=phase,
                    bucket=bucket,
                    top1_new=top1_new,
                    top1_ref=top1_ref,
                    sample_count=sample_count,
                )
            )
    return comparisons


def evaluate_three_tier_rule(comparisons: list[BucketComparison]) -> dict:
    hard_failures = [
        {
            "phase": item.phase,
            "bucket": item.bucket,
            "delta": item.delta,
            "abs_delta": item.abs_delta,
            "se": item.se,
            "threshold_3se": 3 * item.se,
        }
        for item in comparisons
        if item.abs_delta > 3 * item.se
    ]
    soft_zone = [
        item
        for item in comparisons
        if (2 * item.se) < item.abs_delta <= (3 * item.se)
    ]
    sign_p = sign_test_pvalue([item.delta for item in comparisons])
    passed = (
        len(hard_failures) == 0
        and len(soft_zone) <= SOFT_QUOTA
        and sign_p >= 0.01
    )
    rows = [
        {
            "phase": item.phase,
            "bucket": item.bucket,
            "top1_new": item.top1_new,
            "top1_ref": item.top1_ref,
            "delta": item.delta,
            "se": item.se,
            "abs_delta": item.abs_delta,
            "within_2se": item.abs_delta <= 2 * item.se,
            "soft_zone": (2 * item.se) < item.abs_delta <= (3 * item.se),
            "hard_fail": item.abs_delta > 3 * item.se,
        }
        for item in comparisons
    ]
    return {
        "n_cmp": len(comparisons),
        "soft_quota_max": SOFT_QUOTA,
        "soft_zone_count": len(soft_zone),
        "hard_failures": hard_failures,
        "sign_test_pvalue": sign_p,
        "pass": passed,
        "comparisons": rows,
    }


def load_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" in payload:
        return payload["metrics"]
    history = payload.get("history", [])
    if history:
        return history[-1]
    raise ValueError(f"No metrics found in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate F-4 section 2.1 bucket rules")
    parser.add_argument("--new-phase1", type=Path, required=True)
    parser.add_argument("--new-phase2", type=Path, required=True)
    parser.add_argument("--new-phase3", type=Path, required=True)
    parser.add_argument("--ref-phase1", type=Path, required=True)
    parser.add_argument("--ref-phase2", type=Path, required=True)
    parser.add_argument("--ref-phase3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", type=str, default="4a-E")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    new_by_phase = {
        "phase1": load_metrics(args.new_phase1),
        "phase2": load_metrics(args.new_phase2),
        "phase3": load_metrics(args.new_phase3),
    }
    ref_by_phase = {
        "phase1": load_metrics(args.ref_phase1),
        "phase2": load_metrics(args.ref_phase2),
        "phase3": load_metrics(args.ref_phase3),
    }
    comparisons = build_comparisons(new_by_phase, ref_by_phase)
    rule = evaluate_three_tier_rule(comparisons)
    report = {
        "label": args.label,
        "rule": rule,
        "new_sources": {
            "phase1": str(args.new_phase1),
            "phase2": str(args.new_phase2),
            "phase3": str(args.new_phase3),
        },
        "ref_sources": {
            "phase1": str(args.ref_phase1),
            "phase2": str(args.ref_phase2),
            "phase3": str(args.ref_phase3),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{args.label}: {'PASS' if rule['pass'] else 'FAIL'}")
    print(f"  hard_failures={len(rule['hard_failures'])} soft_zone={rule['soft_zone_count']}/{SOFT_QUOTA}")
    print(f"  sign_test_p={rule['sign_test_pvalue']:.6f}")
    print(f"  report: {args.output}")
    return 0 if rule["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
