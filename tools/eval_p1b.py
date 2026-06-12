#!/usr/bin/env python3
"""P1b champion retest: Phase 3 C=2 L=2 on v3 test with NULL-0_ref (same session)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (_REPO_ROOT / "model", _REPO_ROOT / "train"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset import GoldbachBinDataset  # noqa: E402
from metrics import compute_ranking_metrics  # noqa: E402
from null_model import NullModel  # noqa: E402
from train import _forward_logits, _masked_bce_loss, build_model  # noqa: E402


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


def _load_model(checkpoint: Path, device: torch.device) -> nn.Module:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = _remap_legacy_phase3_state(payload["model_state_dict"])
    model = build_model(
        "phase3",
        cross_attn_layers=int(payload.get("cross_attn_layers") or 2),
        self_attn_layers=int(payload.get("self_attn_layers") or 2),
        stack_mode=str(payload.get("stack_mode") or "sequential"),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
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
        outputs = _forward_logits(model, device_batch, "phase3")
        logits = outputs["final_logits"]
        eligibility = outputs["eligibility_mask"]
        loss = _masked_bce_loss(logits, valid_mask, eligibility)
        batch_size = even.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        logit_batches.append(logits.cpu())
        mask_batches.append(valid_mask.cpu())
        even_batches.append(even.cpu())

    return compute_ranking_metrics(
        torch.cat(logit_batches, dim=0),
        torch.cat(mask_batches, dim=0),
        bce_loss=total_loss / max(total_samples, 1),
        even_numbers=torch.cat(even_batches, dim=0),
        top_k=500,
    ).to_dict()


@torch.no_grad()
def evaluate_null0(loader: DataLoader) -> dict:
    null0 = NullModel()
    logit_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    even_batches: list[torch.Tensor] = []
    for batch in loader:
        even = batch["even"]
        valid_mask = batch["valid_mask"]
        logits = torch.from_numpy(null0.predict_logits_batch(even.numpy())).float()
        logit_batches.append(logits)
        mask_batches.append(valid_mask)
        even_batches.append(even)
    even_all = torch.cat(even_batches, dim=0)
    logits_all = torch.cat(logit_batches, dim=0)
    mask_all = torch.cat(mask_batches, dim=0)
    loss_fn = nn.BCEWithLogitsLoss()
    bce = loss_fn(logits_all, mask_all).item()
    return compute_ranking_metrics(
        logits_all,
        mask_all,
        bce_loss=bce,
        even_numbers=even_all,
        top_k=500,
    ).to_dict()


def parse_args() -> argparse.Namespace:
    root = _REPO_ROOT
    parser = argparse.ArgumentParser(description="P1b test evaluation on v3")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--test-data",
        type=Path,
        default=root / "data" / "goldbach_train_v3_test.bin",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--null0-ref", type=str, default="v1.0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device_name = args.device
    device = torch.device(device_name)

    dataset = GoldbachBinDataset(args.test_data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = _load_model(args.checkpoint, device)

    null0_metrics = evaluate_null0(loader)
    model_metrics = evaluate_loader(model, loader, device)
    delta_top1 = model_metrics["top1_valid_rate"] - null0_metrics["top1_valid_rate"]
    threshold = 0.003
    p1b_pass = abs(delta_top1) <= threshold

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "null0_ref": args.null0_ref,
        "checkpoint": str(args.checkpoint),
        "test_data": str(args.test_data),
        "sample_count": model_metrics["sample_count"],
        "null0_test": null0_metrics,
        "model_test": model_metrics,
        "delta_top1_pp": delta_top1 * 100.0,
        "threshold_pp": threshold * 100.0,
        "p1b_pass": p1b_pass,
        "legacy_champion_top1": 0.4463,
        "legacy_reproduced": abs(model_metrics["top1_valid_rate"] - 0.4463) <= 0.005,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"P1b test top1 model={model_metrics['top1_valid_rate']:.4f} null0={null0_metrics['top1_valid_rate']:.4f}")
    print(f"  delta={delta_top1*100:.3f}pp  pass={p1b_pass}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
