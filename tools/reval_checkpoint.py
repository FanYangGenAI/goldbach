#!/usr/bin/env python3
"""Re-evaluate a historical checkpoint with current masked metrics (F-4a)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

_REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (_REPO_ROOT / "model", _REPO_ROOT / "train"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset import GoldbachBinDataset  # noqa: E402
from igot import IGOTModel  # noqa: E402
from metrics import compute_ranking_metrics  # noqa: E402
from train import (  # noqa: E402
    _forward_logits,
    _masked_bce_loss,
    build_model,
)


def _remap_legacy_phase3_state(state_dict: dict) -> dict:
    if "prime_mod_cross_attn.attention.q_proj.weight" not in state_dict:
        return state_dict
    remapped: dict = {}
    for key, value in state_dict.items():
        if key.startswith("prime_mod_cross_attn."):
            remapped["cross_attn_layers.0." + key[len("prime_mod_cross_attn.") :]] = value
        elif key.startswith("prime_prime_self_attn."):
            remapped["self_attn_layers.0." + key[len("prime_prime_self_attn.") :]] = value
        else:
            remapped[key] = value
    return remapped


def _load_checkpoint(path: Path, phase: str, device: torch.device) -> nn.Module:
    payload = torch.load(path, map_location=device, weights_only=False)
    state_dict = _remap_legacy_phase3_state(payload["model_state_dict"])
    model = build_model(
        phase,
        cross_attn_layers=int(payload.get("cross_attn_layers") or 1),
        self_attn_layers=int(payload.get("self_attn_layers") or 1),
        stack_mode=str(payload.get("stack_mode") or "sequential"),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_checkpoint(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    phase: str,
    top_k: int,
) -> dict:
    total_loss = 0.0
    total_samples = 0
    logit_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    even_batches: list[torch.Tensor] = []

    for batch in loader:
        even = batch["even"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        device_batch = {key: value.to(device) for key, value in batch.items()}
        outputs = _forward_logits(model, device_batch, phase)
        logits = outputs["final_logits"]
        eligibility = outputs["eligibility_mask"]
        loss = _masked_bce_loss(logits, valid_mask, eligibility)

        batch_size = even.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        logit_batches.append(logits.cpu())
        mask_batches.append(valid_mask.cpu())
        even_batches.append(even.cpu())

    metrics = compute_ranking_metrics(
        torch.cat(logit_batches, dim=0),
        torch.cat(mask_batches, dim=0),
        bce_loss=total_loss / max(total_samples, 1),
        even_numbers=torch.cat(even_batches, dim=0),
        top_k=top_k,
    )
    return metrics.to_dict()


def _build_val_loader(
    data_path: Path,
    *,
    val_fraction: float,
    seed: int,
    batch_size: int,
    val_data: Path | None = None,
) -> tuple[DataLoader, int]:
    if val_data is not None:
        dataset = GoldbachBinDataset(val_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return loader, dataset.candidate_count or 500

    dataset = GoldbachBinDataset(data_path)
    top_k = dataset.candidate_count or 500
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    _, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, top_k


def reval_checkpoint(
    *,
    checkpoint: Path,
    phase: str,
    data_path: Path,
    val_fraction: float,
    seed: int,
    batch_size: int,
    device_name: str,
    val_data: Path | None = None,
) -> dict:
    device = torch.device(device_name)
    loader, top_k = _build_val_loader(
        data_path,
        val_fraction=val_fraction,
        seed=seed,
        batch_size=batch_size,
        val_data=val_data,
    )
    model = _load_checkpoint(checkpoint, phase, device)
    metrics = evaluate_checkpoint(model, loader, device, phase=phase, top_k=top_k)
    return {
        "checkpoint": str(checkpoint),
        "phase": phase,
        "data": str(data_path),
        "val_fraction": val_fraction,
        "seed": seed,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    root = _REPO_ROOT
    parser = argparse.ArgumentParser(description="Re-evaluate checkpoint with masked metrics")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase", choices=("phase1", "phase2", "phase3"), required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "goldbach_train_v2_phase0.bin",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--val-data",
        type=Path,
        default=None,
        help="Evaluate on this file directly (no random split)",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if args.val_data is None and not args.data.exists():
        print(f"Dataset not found: {args.data}", file=sys.stderr)
        return 1
    if args.val_data is not None and not args.val_data.exists():
        print(f"Val dataset not found: {args.val_data}", file=sys.stderr)
        return 1

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"

    report = reval_checkpoint(
        checkpoint=args.checkpoint,
        phase=args.phase,
        data_path=args.data,
        val_fraction=args.val_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
        device_name=device_name,
        val_data=args.val_data,
    )
    output_path = args.output or args.checkpoint.parent / "reval_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    metrics = report["metrics"]
    print(f"Re-evaluated {args.checkpoint}")
    print(f"  masked BCE: {metrics['bce_loss']:.4f}")
    print(f"  top1:       {metrics['top1_valid_rate']:.4f}")
    print(f"  report:     {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
