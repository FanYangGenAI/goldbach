#!/usr/bin/env python3
"""Train IGOT Phase 1 explicit sieve baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "model") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "model"))
if str(_REPO_ROOT / "train") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "train"))

from dataset import GoldbachBinDataset  # noqa: E402
from metrics import compute_ranking_metrics  # noqa: E402
from sieve_baseline import ExplicitSieveModule  # noqa: E402


def _repo_root() -> Path:
    return _REPO_ROOT


def _default_data_path(root: Path) -> Path:
    return root / "data" / "goldbach_train_v2_phase0.bin"


@torch.no_grad()
def evaluate(
    model: ExplicitSieveModule,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    top_k: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    logit_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    even_batches: list[torch.Tensor] = []

    for batch in loader:
        even = batch["even"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        outputs = model(even)
        logits = outputs["sieve_logits"]
        loss = loss_fn(logits, valid_mask)
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


def train_epoch(
    model: ExplicitSieveModule,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        even = batch["even"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(even)["sieve_logits"]
        loss = loss_fn(logits, valid_mask)
        loss.backward()
        optimizer.step()
        batch_size = even.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def save_checkpoint(
    path: Path,
    model: ExplicitSieveModule,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "modulus_weights": model.modulus_weight_table(),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Train explicit sieve baseline (IGOT Phase 1)")
    parser.add_argument("--data", type=Path, default=root / "data" / "goldbach_train_v2_phase0.bin")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "phase1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> int:
    args = parse_args()
    if not args.data.exists():
        print(f"Dataset not found: {args.data}", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dataset = GoldbachBinDataset(args.data)
    top_k = dataset.candidate_count or 500

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = ExplicitSieveModule().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    history: list[dict] = []
    best_topk = -1.0
    started = time.perf_counter()

    print(f"Training explicit sieve baseline on {device}")
    print(f"  train samples: {train_size:,}")
    print(f"  val samples:   {val_size:,}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device, top_k=top_k)
        epoch_record = {
            "epoch": epoch,
            "train_bce_loss": train_loss,
            **val_metrics,
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch:02d}  "
            f"train_bce={train_loss:.4f}  "
            f"val_bce={val_metrics['bce_loss']:.4f}  "
            f"top1={val_metrics['top1_valid_rate']:.4f}  "
            f"top5={val_metrics['top5_valid_rate']:.4f}  "
            f"topK={val_metrics['topk_valid_rate']:.4f}  "
            f"avg_checks={val_metrics['avg_checks_to_success']:.2f}"
        )

        if val_metrics["topk_valid_rate"] >= best_topk:
            best_topk = val_metrics["topk_valid_rate"]
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, val_metrics)

    elapsed = time.perf_counter() - started
    save_checkpoint(args.output_dir / "last.pt", model, optimizer, args.epochs, history[-1])

    report = {
        "phase": "phase1_sieve_baseline",
        "data": str(args.data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "device": str(device),
        "elapsed_seconds": round(elapsed, 3),
        "history": history,
        "final_modulus_weights": model.modulus_weight_table(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "train_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("Training complete.")
    print(f"  best checkpoint: {args.output_dir / 'best.pt'}")
    print(f"  report:          {report_path}")
    print(f"  elapsed:         {elapsed:.1f}s")
    print("Top modulus weights:")
    for modulus, weight in sorted(model.modulus_weight_table(), key=lambda item: item[1])[:8]:
        print(f"  r={modulus:3d}  w={weight:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
