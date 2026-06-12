#!/usr/bin/env python3
"""Run WP-F F-4b formal regression + P1b test evaluation on v3."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DATA = _REPO_ROOT / "data"
NULL_MODEL_COMMIT = "v1.0"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)


def train_matrix(output_dir: Path, seeds: list[int]) -> None:
    runs = [
        ("f4-p1", "phase1", ["--batch-size", "2048", "--lr", "1e-2"]),
        ("f4-p2", "phase2", ["--batch-size", "2048", "--lr", "1e-2"]),
        (
            "f4-p3",
            "phase3",
            [
                "--batch-size",
                "2048",
                "--lr",
                "0.0025",
                "--cross-attn-layers",
                "2",
                "--self-attn-layers",
                "2",
                "--stack-mode",
                "sequential",
                "--lr-schedule",
                "cosine",
                "--epochs",
                "50",
            ],
        ),
    ]
    for run_id, phase, extra in runs:
        for seed in seeds:
            run_path = output_dir / run_id / f"seed_{seed}"
            cmd = [
                PYTHON,
                "train/train.py",
                "--dataset",
                "v3",
                "--phase",
                phase,
                *extra,
                "--seeds",
                str(seed),
                "--select-metric",
                "val_bce",
                "--output-dir",
                str(run_path),
            ]
            if phase == "phase2":
                cmd.extend(
                    [
                        "--baseline-report",
                        str(output_dir / "f4-p1" / f"seed_{seed}" / "train_report.json"),
                    ]
                )
            if phase == "phase3":
                cmd.extend(
                    [
                        "--phase2-report",
                        str(output_dir / "f4-p2" / f"seed_{seed}" / "train_report.json"),
                    ]
                )
            _run(cmd)
            _run(
                [
                    PYTHON,
                    "tools/reval_checkpoint.py",
                    "--checkpoint",
                    str(run_path / "best.pt"),
                    "--phase",
                    phase,
                    "--val-data",
                    str(DATA / "goldbach_train_v3_val.bin"),
                    "--batch-size",
                    "2048",
                    "--output",
                    str(run_path / "best_reval_report.json"),
                ]
            )


def eval_p1b(output_dir: Path, seeds: list[int]) -> None:
    log = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "null0_ref": NULL_MODEL_COMMIT,
        "seeds": seeds,
        "runs": [],
    }
    for seed in seeds:
        run_path = output_dir / "f4-p3" / f"seed_{seed}"
        out = run_path / "p1b_test_eval.json"
        _run(
            [
                PYTHON,
                "tools/eval_p1b.py",
                "--checkpoint",
                str(run_path / "best.pt"),
                "--output",
                str(out),
                "--null0-ref",
                NULL_MODEL_COMMIT,
            ]
        )
        log["runs"].append(json.loads(out.read_text(encoding="utf-8")))
    log_path = output_dir / "p1b_eval_log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"P1b log: {log_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F-4b + P1b on v3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "wpf_regress",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--p1b-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.p1b_only:
        train_matrix(args.output_dir, seeds)
    eval_p1b(args.output_dir, seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
