#!/usr/bin/env python3
"""Plot train/val curves from train_report.json.

从 train_report.json 生成训练/验证损失曲线及排序指标曲线。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_FIGURES = _REPO / "docs" / "figures"


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc
    return plt


def _load_history(report_path: Path) -> tuple[dict, list[dict]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    history = report.get("history", [])
    if not history:
        raise ValueError(f"No history found in {report_path}")
    return report, history


def plot_bce_curve(
    history: list[dict],
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    """Plot train vs validation BCE loss by epoch."""
    plt = _require_matplotlib()

    epochs = [row["epoch"] for row in history]
    train_bce = [row["train_bce_loss"] for row in history]
    val_bce = [row["bce_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_bce, marker="o", markersize=3, label="train BCE")
    ax.plot(epochs, val_bce, marker="o", markersize=3, label="val BCE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ranking_curve(
    history: list[dict],
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    """Plot top1 / top5 valid rates by epoch."""
    plt = _require_matplotlib()

    epochs = [row["epoch"] for row in history]
    top1 = [row["top1_valid_rate"] * 100 for row in history]
    top5 = [row["top5_valid_rate"] * 100 for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, top1, marker="o", markersize=3, label="top1 valid %")
    ax.plot(epochs, top5, marker="o", markersize=3, label="top5 valid %")
    ax.set_xlabel("epoch")
    ax.set_ylabel("valid rate (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_report(
    report_path: Path,
    output_dir: Path,
    *,
    stem: str | None = None,
    title: str | None = None,
    include_ranking: bool = True,
) -> list[Path]:
    """Generate curve PNGs for one train_report.json."""
    report, history = _load_history(report_path)
    stem = stem or report_path.parent.name
    phase = report.get("phase", stem)
    plot_title = title or phase

    outputs: list[Path] = []
    bce_path = output_dir / f"{stem}_train_val_bce.png"
    plot_bce_curve(history, bce_path, title=plot_title)
    outputs.append(bce_path)

    if include_ranking:
        rank_path = output_dir / f"{stem}_top1_top5_vs_epoch.png"
        plot_ranking_curve(history, rank_path, title=plot_title)
        outputs.append(rank_path)

    return outputs


def plot_baseline_bundle(repo_root: Path, figures_dir: Path) -> list[Path]:
    """Retroactively plot Phase 1/2/3 baseline curves without retraining."""
    baselines = [
        (
            repo_root / "results" / "phase1_gpu" / "train_report.json",
            "phase1",
            "Phase 1 sieve baseline",
        ),
        (
            repo_root / "results" / "phase2_bs2048" / "train_report.json",
            "phase2",
            "Phase 2 sieve+residual (batch=2048)",
        ),
        (
            repo_root / "results" / "phase3_flash_bs2048" / "train_report.json",
            "phase3_baseline",
            "Phase 3 IGOT baseline (SDPA, batch=2048)",
        ),
    ]

    outputs: list[Path] = []
    for report_path, stem, title in baselines:
        if not report_path.exists():
            print(f"SKIP (missing): {report_path}", file=sys.stderr)
            continue
        paths = plot_report(report_path, figures_dir, stem=stem, title=title)
        outputs.extend(paths)
        print(f"Wrote {len(paths)} figure(s) from {report_path}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from train_report.json")
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        help="Path to train_report.json (omit with --baseline-bundle)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_FIGURES,
        help="Directory for PNG outputs",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Output filename stem (default: parent dir name of report)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title",
    )
    parser.add_argument(
        "--bce-only",
        action="store_true",
        help="Only plot BCE curve (skip top1/top5)",
    )
    parser.add_argument(
        "--baseline-bundle",
        action="store_true",
        help="Plot Phase 1/2/3 baseline reports into docs/figures/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.baseline_bundle:
        plot_baseline_bundle(_REPO, args.output_dir)
        return 0

    if args.report is None:
        print("Provide a report path or use --baseline-bundle", file=sys.stderr)
        return 1

    report_path = args.report if args.report.is_absolute() else _REPO / args.report
    if not report_path.exists():
        print(f"Report not found: {report_path}", file=sys.stderr)
        return 1

    paths = plot_report(
        report_path,
        args.output_dir,
        stem=args.stem,
        title=args.title,
        include_ranking=not args.bce_only,
    )
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
