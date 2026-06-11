#!/usr/bin/env python3
"""v3 data acceptance gate: appendix C items 1, 2, 5, 6 (WP-F F-4b prerequisite)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_GENERATOR_PYTHON = _ROOT / "generator" / "python"
if str(_GENERATOR_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GENERATOR_PYTHON))

from constants import (  # noqa: E402
    CANDIDATE_PRIME_COUNT,
    HEADER_SIZE,
    V3_ASSIGNMENT_SEED,
    V3_SPLIT_IDS,
    V3_SPLIT_NAMES,
)
from format_io import (  # noqa: E402
    GoldbachSampleV2,
    decode_valid_mask,
    encode_valid_mask,
    sample_size_for_version,
    verify_header,
)
from v3_splits import assign_split_for_even  # noqa: E402


def _read_all_evens(path: Path) -> list[int]:
    with path.open("rb") as handle:
        header = verify_header(handle.read(HEADER_SIZE))
    record_size = sample_size_for_version(header.version)
    evens: list[int] = []
    with path.open("rb") as handle:
        handle.seek(HEADER_SIZE)
        for _ in range(header.sample_count):
            row = handle.read(record_size)
            if len(row) != record_size:
                raise ValueError(f"Truncated record in {path}")
            evens.append(int.from_bytes(row[0:8], "little"))
    return evens


def _audit_split_leakage(split_evens: dict[str, list[int]]) -> dict:
    overlap: list[dict] = []
    split_sets = {name: set(values) for name, values in split_evens.items()}
    names = list(split_evens)
    for left_index in range(len(names)):
        for right_index in range(left_index + 1, len(names)):
            left_name = names[left_index]
            right_name = names[right_index]
            shared = split_sets[left_name] & split_sets[right_name]
            if shared:
                overlap.append(
                    {
                        "splits": [left_name, right_name],
                        "count": len(shared),
                        "examples": sorted(shared)[:5],
                    }
                )
    return {
        "id": "C1_split_leakage",
        "pass": len(overlap) == 0,
        "overlap": overlap,
    }


def _audit_intra_split_duplicates(split_evens: dict[str, list[int]]) -> dict:
    failures: list[dict] = []
    for split_name, evens in split_evens.items():
        seen: set[int] = set()
        duplicates: list[int] = []
        for even in evens:
            if even in seen:
                duplicates.append(even)
            seen.add(even)
        if duplicates:
            failures.append(
                {
                    "split": split_name,
                    "duplicate_count": len(duplicates),
                    "examples": sorted(set(duplicates))[:5],
                }
            )
    return {
        "id": "C2_intra_split_duplicates",
        "pass": len(failures) == 0,
        "failures": failures,
    }


def _audit_assignment_rng(split_evens: dict[str, list[int]]) -> dict:
    mismatches: list[dict] = []
    for split_name, evens in split_evens.items():
        for even in evens:
            assigned = assign_split_for_even(even, V3_ASSIGNMENT_SEED)
            if assigned != split_name:
                mismatches.append(
                    {"even": even, "expected": split_name, "assigned": assigned}
                )
                if len(mismatches) >= 20:
                    break
        if len(mismatches) >= 20:
            break
    return {
        "id": "C5_assignment_rng",
        "pass": len(mismatches) == 0,
        "assignment_seed": V3_ASSIGNMENT_SEED,
        "mismatches": mismatches,
    }


def _audit_bitset_roundtrip() -> dict:
    cases: list[tuple[str, tuple[int, ...]]] = [
        ("all_zero", tuple(0 for _ in range(CANDIDATE_PRIME_COUNT))),
        ("all_one", tuple(1 for _ in range(CANDIDATE_PRIME_COUNT))),
        (
            "alternating",
            tuple(1 if index % 2 == 0 else 0 for index in range(CANDIDATE_PRIME_COUNT)),
        ),
    ]
    rng = random.Random(7)
    random_case = tuple(rng.randint(0, 1) for _ in range(CANDIDATE_PRIME_COUNT))
    cases.append(("random_1000_seed7", random_case))

    failures: list[str] = []
    for label, mask in cases:
        encoded = encode_valid_mask(mask)
        decoded = decode_valid_mask(encoded)
        if decoded != mask:
            failures.append(label)

    sample = GoldbachSampleV2(
        even_number=100,
        modulo_features=tuple(0 for _ in range(10)),
        prime_density=0.0,
        valid_mask=random_case,
    )
    roundtrip = GoldbachSampleV2.from_bytes(sample.to_bytes())
    if roundtrip.valid_mask != random_case:
        failures.append("sample_v2_roundtrip")

    return {
        "id": "C6_bitset_roundtrip",
        "pass": len(failures) == 0,
        "failures": failures,
    }


def run_v3_audit(
    *,
    data_dir: Path,
    manifest_path: Path | None,
    output_path: Path,
) -> dict:
    split_paths = {
        name: data_dir / f"goldbach_train_v3_{name}.bin"
        for name in V3_SPLIT_NAMES
    }
    for path in split_paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")

    split_evens = {name: _read_all_evens(path) for name, path in split_paths.items()}
    checks = [
        _audit_split_leakage(split_evens),
        _audit_intra_split_duplicates(split_evens),
        _audit_assignment_rng(split_evens),
        _audit_bitset_roundtrip(),
    ]

    header_checks: list[dict] = []
    for split_name, path in split_paths.items():
        with path.open("rb") as handle:
            header = verify_header(handle.read(HEADER_SIZE))
        expected_id = V3_SPLIT_IDS[split_name]
        header_ok = header.split_id == expected_id
        header_checks.append(
            {
                "split": split_name,
                "split_id": header.split_id,
                "expected_split_id": expected_id,
                "dedup_hash": header.dedup_hash,
                "pass": header_ok,
            }
        )

    report = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path) if manifest_path else None,
        "split_files": {name: str(path) for name, path in split_paths.items()},
        "checks": checks,
        "header_checks": header_checks,
        "pass": all(item["pass"] for item in checks)
        and all(item["pass"] for item in header_checks),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 data acceptance gate audit")
    parser.add_argument("--data-dir", type=Path, default=_ROOT / "data")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_ROOT / "data" / "goldbach_manifest_v3.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "results" / "wpf_regress" / "v3_audit_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_v3_audit(
            data_dir=args.data_dir,
            manifest_path=args.manifest if args.manifest.exists() else None,
            output_path=args.output,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    for check in report["checks"]:
        status = "PASS" if check["pass"] else "FAIL"
        print(f"{status}: {check['id']}")
    for check in report["header_checks"]:
        status = "PASS" if check["pass"] else "FAIL"
        print(f"{status}: header split={check['split']}")
    print(f"Overall: {'PASS' if report['pass'] else 'FAIL'}")
    print(f"Report: {args.output}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
