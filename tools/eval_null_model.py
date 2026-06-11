#!/usr/bin/env python3
"""Evaluate NULL-0 / NULL-1 on the IGOT validation split (P0 acceptance)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

_REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (_REPO_ROOT / "model", _REPO_ROOT / "train", _REPO_ROOT / "generator" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constants import CANDIDATE_PRIME_COUNT  # noqa: E402
from dataset import GoldbachBinDataset  # noqa: E402
from metrics import compute_ranking_metrics  # noqa: E402
from null_model import NullModel, NullModelCalibrated  # noqa: E402


def _repo_root() -> Path:
    return _REPO_ROOT


def _build_val_subset(
    dataset: GoldbachBinDataset,
    *,
    val_fraction: float,
    seed: int,
) -> Subset:
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    _, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return val_set


def _build_train_subset(
    dataset: GoldbachBinDataset,
    *,
    val_fraction: float,
    seed: int,
) -> Subset:
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    train_set, _ = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_set


@torch.no_grad()
def evaluate_null_logits(
    even_numbers: torch.Tensor,
    ranking_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    bce_logits: torch.Tensor | None = None,
) -> dict:
    """Evaluate ranking from `ranking_logits`; BCE from `bce_logits` when provided."""
    loss_fn = nn.BCEWithLogitsLoss()
    bce_source = ranking_logits if bce_logits is None else bce_logits
    bce = loss_fn(bce_source, valid_mask).item()
    metrics = compute_ranking_metrics(
        ranking_logits,
        valid_mask,
        bce_loss=bce,
        even_numbers=even_numbers,
        top_k=CANDIDATE_PRIME_COUNT,
    )
    return metrics.to_dict()


def collect_split_tensors(subset: Subset, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    even_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    for batch in loader:
        even_batches.append(batch["even"])
        mask_batches.append(batch["valid_mask"])
    return torch.cat(even_batches, dim=0), torch.cat(mask_batches, dim=0)


def evaluate_variant(
    name: str,
    even_numbers: torch.Tensor,
    valid_mask: torch.Tensor,
    ranking_logits: torch.Tensor,
    *,
    bce_logits: torch.Tensor | None = None,
) -> dict:
    metrics = evaluate_null_logits(
        even_numbers,
        ranking_logits,
        valid_mask,
        bce_logits=bce_logits,
    )
    return {"name": name, "metrics": metrics}


def load_reference_metrics(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if "best_val_metrics" in report:
        return report["best_val_metrics"]
    if "final_val_metrics" in report:
        return report["final_val_metrics"]
    history = report.get("history", [])
    if not history:
        raise ValueError(f"No metrics found in {path}")
    best = max(history, key=lambda row: row["top1_valid_rate"])
    return {key: best[key] for key in best if key != "epoch"}


def compare_to_reference(null_metrics: dict, reference: dict) -> dict[str, float]:
    return {
        "top1_valid_rate": null_metrics["top1_valid_rate"] - reference["top1_valid_rate"],
        "top5_valid_rate": null_metrics["top5_valid_rate"] - reference["top5_valid_rate"],
        "avg_checks_to_success": null_metrics["avg_checks_to_success"] - reference["avg_checks_to_success"],
        "bce_loss": null_metrics["bce_loss"] - reference["bce_loss"],
    }


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Evaluate NULL-0 / NULL-1 (WP-N P0)")
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "goldbach_train_v2_phase0.bin",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "null",
    )
    parser.add_argument(
        "--phase1-report",
        type=Path,
        default=root / "results" / "phase1_min6" / "train_report.json",
    )
    parser.add_argument(
        "--phase2-report",
        type=Path,
        default=root / "results" / "phase2_bs2048" / "train_report.json",
    )
    parser.add_argument(
        "--phase3-report",
        type=Path,
        default=root / "results" / "phase3_depth" / "stageII" / "B_C2_L2" / "train_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = GoldbachBinDataset(args.data)
    val_subset = _build_val_subset(dataset, val_fraction=args.val_fraction, seed=args.seed)
    train_subset = _build_train_subset(dataset, val_fraction=args.val_fraction, seed=args.seed)

    val_even, val_mask = collect_split_tensors(val_subset, args.batch_size)
    train_even, train_mask = collect_split_tensors(train_subset, args.batch_size)

    null0 = NullModel()
    null0_logits = torch.from_numpy(
        null0.predict_logits_batch(val_even.numpy())
    ).to(dtype=torch.float32)
    null0_result = evaluate_variant("NULL-0", val_even, val_mask, null0_logits)

    null1 = NullModelCalibrated(null0)
    null1.fit(train_even.numpy(), train_mask.numpy())
    null1_bce_logits = torch.from_numpy(
        null1.predict_calibrated_logits_batch(val_even.numpy())
    ).to(dtype=torch.float32)
    null1_ranking_logits = torch.from_numpy(
        null1.ranking_logits_batch(val_even.numpy())
    ).to(dtype=torch.float32)
    null1_result = evaluate_variant(
        "NULL-1",
        val_even,
        val_mask,
        null1_ranking_logits,
        bce_logits=null1_bce_logits,
    )
    null1_legacy_logits = torch.from_numpy(
        null1.predict_calibrated_logits_batch(val_even.numpy())
    ).to(dtype=torch.float32)
    null1_legacy_ranking = evaluate_null_logits(val_even, null1_legacy_logits, val_mask)

    references: dict[str, dict] = {}
    deltas: dict[str, dict] = {}
    for label, report_path in (
        ("phase1_min6", args.phase1_report),
        ("phase2_bs2048", args.phase2_report),
        ("phase3_B_C2_L2", args.phase3_report),
    ):
        if report_path.exists():
            ref_metrics = load_reference_metrics(report_path)
            references[label] = ref_metrics
            deltas[label] = compare_to_reference(null0_result["metrics"], ref_metrics)

    p0 = {
        "top1_within_1pp": all(abs(delta["top1_valid_rate"]) <= 0.01 for delta in deltas.values()),
        "top5_within_1pp": all(abs(delta["top5_valid_rate"]) <= 0.01 for delta in deltas.values()),
        "checks_within_0_1": all(abs(delta["avg_checks_to_success"]) <= 0.1 for delta in deltas.values()),
        "null1_bce_le_0_127": null1_result["metrics"]["bce_loss"] <= 0.127,
    }
    p0_pass = all(p0.values()) if deltas else False

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "val_sample_count": int(val_even.shape[0]),
        "train_sample_count": int(train_even.shape[0]),
        "variants": {
            "NULL-0": null0_result,
            "NULL-1": null1_result,
        },
        "null1_ranking_inheritance": {
            "top1_null0": null0_result["metrics"]["top1_valid_rate"],
            "top1_null1_corrected": null1_result["metrics"]["top1_valid_rate"],
            "top1_null1_uncorrected": null1_legacy_ranking["top1_valid_rate"],
            "top1_match_after_fix": abs(
                null0_result["metrics"]["top1_valid_rate"]
                - null1_result["metrics"]["top1_valid_rate"]
            )
            < 1e-9,
        },
        "reference_models": references,
        "null0_minus_reference": deltas,
        "p0_checks": p0,
        "p0_pass": p0_pass,
    }

    report_path = args.output_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"NULL-0 val metrics: {json.dumps(null0_result['metrics'], indent=2)}")
    print(f"NULL-1 val metrics: {json.dumps(null1_result['metrics'], indent=2)}")
    print(f"P0 checks: {json.dumps(p0, indent=2)}")
    print(f"P0 pass: {p0_pass}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
