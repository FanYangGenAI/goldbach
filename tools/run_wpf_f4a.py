#!/usr/bin/env python3
"""Run WP-F F-4a v2 control batch: historical reval + retrain + criteria."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

HISTORICAL_CHECKPOINTS = {
    "phase1": _REPO_ROOT / "results" / "phase1_gpu" / "best.pt",
    "phase2": _REPO_ROOT / "results" / "phase2_bs2048" / "best.pt",
    "phase3": _REPO_ROOT / "results" / "phase3_flash_bs2048" / "best.pt",
}


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or _REPO_ROOT, check=True)


def reval_historical(output_dir: Path) -> dict[str, Path]:
    reval_dir = output_dir / "historical_reval"
    reval_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for phase, checkpoint in HISTORICAL_CHECKPOINTS.items():
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing historical checkpoint: {checkpoint}")
        out = reval_dir / f"{phase}_reval_report.json"
        batch = 512 if phase == "phase1" else 2048
        _run(
            [
                PYTHON,
                "tools/reval_checkpoint.py",
                "--checkpoint",
                str(checkpoint),
                "--phase",
                phase,
                "--batch-size",
                str(batch),
                "--output",
                str(out),
            ]
        )
        paths[phase] = out
    return paths


def train_f4a(output_dir: Path, *, seed: int) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    configs = [
        ("phase1", ["--phase", "phase1", "--batch-size", "512", "--lr", "1e-2"]),
        ("phase2", ["--phase", "phase2", "--batch-size", "2048", "--lr", "1e-2"]),
        (
            "phase3",
            [
                "--phase",
                "phase3",
                "--batch-size",
                "2048",
                "--lr",
                "0.0025",
            ],
        ),
    ]
    for phase, extra in configs:
        run_dir = output_dir / phase / f"seed_{seed}"
        cmd = [
            PYTHON,
            "train/train.py",
            *extra,
            "--epochs",
            "10",
            "--seed",
            str(seed),
            "--select-metric",
            "val_bce",
            "--output-dir",
            str(run_dir),
        ]
        if phase == "phase2":
            cmd.extend(
                [
                    "--baseline-report",
                    str(output_dir / "phase1" / f"seed_{seed}" / "train_report.json"),
                ]
            )
        if phase == "phase3":
            cmd.extend(
                [
                    "--phase2-report",
                    str(output_dir / "phase2" / f"seed_{seed}" / "train_report.json"),
                ]
            )
        _run(cmd)
        reval_best = run_dir / "best_reval_report.json"
        _run(
            [
                PYTHON,
                "tools/reval_checkpoint.py",
                "--checkpoint",
                str(run_dir / "best.pt"),
                "--phase",
                phase,
                "--batch-size",
                "512" if phase == "phase1" else "2048",
                "--output",
                str(reval_best),
            ]
        )
        runs[phase] = reval_best
    return runs


def evaluate_criteria(
    *,
    new_reports: dict[str, Path],
    ref_reports: dict[str, Path],
    output_dir: Path,
) -> dict:
    criteria_path = output_dir / "f4a_criteria_report.json"
    cmd = [
        PYTHON,
        "tools/eval_f4_criteria.py",
        "--label",
        "4a-E",
        "--new-phase1",
        str(new_reports["phase1"]),
        "--new-phase2",
        str(new_reports["phase2"]),
        "--new-phase3",
        str(new_reports["phase3"]),
        "--ref-phase1",
        str(ref_reports["phase1"]),
        "--ref-phase2",
        str(ref_reports["phase2"]),
        "--ref-phase3",
        str(ref_reports["phase3"]),
        "--output",
        str(criteria_path),
    ]
    result = subprocess.run(cmd, cwd=_REPO_ROOT, check=False)
    if result.returncode not in (0, 1):
        result.check_returncode()

    criteria = json.loads(criteria_path.read_text(encoding="utf-8"))
    new_p1 = json.loads(new_reports["phase1"].read_text(encoding="utf-8"))["metrics"]
    ref_p1 = json.loads(ref_reports["phase1"].read_text(encoding="utf-8"))["metrics"]
    bce_delta = new_p1["bce_loss"] - ref_p1["bce_loss"]
    pass_4af = bce_delta <= 0.001
    summary = {
        "4a-E": {
            "pass": criteria["rule"]["pass"],
            "hard_failures": len(criteria["rule"]["hard_failures"]),
            "soft_zone_count": criteria["rule"]["soft_zone_count"],
            "sign_test_pvalue": criteria["rule"]["sign_test_pvalue"],
        },
        "4a-F": {
            "new_bce": new_p1["bce_loss"],
            "ref_bce": ref_p1["bce_loss"],
            "delta": bce_delta,
            "threshold": 0.001,
            "pass": pass_4af,
        },
        "4a-B_bucket1_checks": new_p1["by_bucket"].get("1", {}).get("avg_checks_to_success"),
        "criteria_report": str(criteria_path),
    }
    summary_path = output_dir / "f4a_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F-4a v2 control batch")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "wpf_regress_f4a",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reval-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref_reports = reval_historical(args.output_dir)
    if args.reval_only:
        return 0
    if not args.skip_train:
        new_reports = train_f4a(args.output_dir, seed=args.seed)
    else:
        new_reports = {
            phase: args.output_dir / phase / f"seed_{args.seed}" / "best_reval_report.json"
            for phase in HISTORICAL_CHECKPOINTS
        }
    evaluate_criteria(
        new_reports=new_reports,
        ref_reports=ref_reports,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
