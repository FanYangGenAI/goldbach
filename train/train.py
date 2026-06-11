#!/usr/bin/env python3
"""Train IGOT Phase 1/2/3 models.

训练 IGOT Phase 1/2/3 模型。

Phase 1: final_logit = sieve_logit
Phase 2: final_logit = sieve_logit + residual_logit
Phase 3: final_logit = sieve_logit + attention_logit + residual_logit
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "model") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "model"))
if str(_REPO_ROOT / "train") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "train"))

from dataset import GoldbachBinDataset  # noqa: E402
from metrics import compute_ranking_metrics  # noqa: E402
from igot import IGOTModel  # noqa: E402
from sieve_baseline import ExplicitSieveModule  # noqa: E402
from sieve_residual import SieveResidualModel  # noqa: E402


class GoldbachModel(Protocol):
    """Minimal interface shared by Phase 1/2/3 models."""

    def forward(self, *args, **kwargs): ...

    def modulus_weight_table(self) -> list[tuple[int, float]]: ...


def _repo_root() -> Path:
    return _REPO_ROOT


def _forward_logits(model: nn.Module, batch: dict[str, torch.Tensor], phase: str) -> dict[str, torch.Tensor]:
    """Run model forward and normalize output dict across phases.

    统一 Phase 1/2/3 的前向输出格式，训练循环只依赖 final_logits。
    """
    if phase == "phase1":
        # Phase 1 only needs even_number N.
        # Phase 1 只需要偶数 N
        outputs = model(batch["even"])
        return {
            "final_logits": outputs["sieve_logits"],
            "sieve_logits": outputs["sieve_logits"],
            "residual_logits": None,
        }

    # Phase 2/3 need full feature batch: N, bits, mods, density.
    # Phase 2/3 需要完整特征：N、位流、模特征、density
    return model(batch["even"], batch["bits"], batch["mods"], batch["density"])


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    top_k: int,
    phase: str,
) -> dict:
    """Evaluate on validation set: BCE + ranking metrics.

    在验证集上评估：BCE 损失 + 排序指标（top1/top5/topK/avg_checks）。
    """
    model.eval()
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
        loss = loss_fn(logits, valid_mask)

        batch_size = even.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Accumulate on CPU to reduce GPU memory during full-val pass.
        # 在 CPU 上累积，避免验证阶段占用过多显存
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
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    phase: str,
    scheduler: LambdaLR | None = None,
) -> tuple[float, float]:
    """One training epoch with multi-label BCE loss.

    训练一个 epoch，使用多标签 BCE 损失。
    Returns (mean_train_bce, learning_rate_at_epoch_end).
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        even = batch["even"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        device_batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        logits = _forward_logits(model, device_batch, phase)["final_logits"]
        loss = loss_fn(logits, valid_mask)
        loss.backward()
        # Gradient clip stabilizes Phase 3 (large batch + attention can NaN).
        # 梯度裁剪：稳定 Phase 3 大 batch + 注意力训练，防止 NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = even.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_lr = optimizer.param_groups[0]["lr"]
    return total_loss / max(total_samples, 1), epoch_lr


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    lr_schedule: str,
    total_steps: int,
    warmup_steps: int,
    eta_min: float,
) -> LambdaLR | None:
    """Build per-step LR scheduler: none (fixed) or warmup + cosine decay."""
    if lr_schedule == "none":
        return None

    if lr_schedule != "cosine":
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

    warmup_steps = max(0, warmup_steps)
    decay_steps = max(total_steps - warmup_steps, 1)
    eta_min_ratio = eta_min / base_lr

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(max(step - warmup_steps, 0) / decay_steps, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    phase: str,
) -> None:
    """Save model, optimizer, and validation metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if hasattr(model, "modulus_weight_table"):
        payload["modulus_weights"] = model.modulus_weight_table()
    torch.save(payload, path)


def load_baseline_metrics(path: Path | None) -> dict | None:
    """Load final-epoch metrics from a baseline report for gain computation.

    从基线报告读取最后一个 epoch 的指标，用于 residual_gain / attention_gain。
    """
    if path is None or not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    history = report.get("history", [])
    if not history:
        return None
    return history[-1]


def load_phase1_baseline_metrics(path: Path | None) -> dict | None:
    return load_baseline_metrics(path)


def compute_metric_gain(current: dict, baseline: dict | None) -> dict | None:
    """current - baseline on key metrics (positive top1 = improvement).

    计算当前指标相对基线的差值（top1 为正表示排序质量提升）。
    """
    if baseline is None:
        return None
    keys = (
        "bce_loss",
        "top1_valid_rate",
        "top5_valid_rate",
        "topk_valid_rate",
        "avg_checks_to_success",
    )
    gain = {key: current[key] - baseline[key] for key in keys}
    gain["failure_rate_under_k"] = (
        current["failure_rate_under_k"] - baseline["failure_rate_under_k"]
    )
    return gain


def compute_residual_gain(phase2_metrics: dict, phase1_metrics: dict | None) -> dict | None:
    return compute_metric_gain(phase2_metrics, phase1_metrics)


def compute_attention_gain(phase3_metrics: dict, phase2_metrics: dict | None) -> dict | None:
    """Phase 3 - Phase 2: quantifies attention branch incremental value.

    attention_gain = metrics(Phase3) - metrics(Phase2 末 epoch)。
    """
    return compute_metric_gain(phase3_metrics, phase2_metrics)


def build_model(
    phase: str,
    *,
    cross_attn_layers: int = 1,
    self_attn_layers: int = 1,
    stack_mode: str = "sequential",
) -> nn.Module:
    """Factory for Phase 1/2/3 models.

    phase1 -> ExplicitSieveModule
    phase2 -> SieveResidualModel
    phase3 -> IGOTModel (sieve + cross/self-attn + residual)
    """
    if phase == "phase1":
        return ExplicitSieveModule()
    if phase == "phase2":
        return SieveResidualModel()
    if phase == "phase3":
        return IGOTModel(
            cross_attn_layers=cross_attn_layers,
            self_attn_layers=self_attn_layers,
            stack_mode=stack_mode,  # type: ignore[arg-type]
        )
    raise ValueError(f"Unknown phase: {phase}")


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Train IGOT Phase 1/2/3 models")
    parser.add_argument(
        "--phase",
        choices=("phase1", "phase2", "phase3"),
        default="phase1",
        help="phase1: sieve; phase2: sieve+residual; phase3: full IGOT",
    )
    parser.add_argument("--data", type=Path, default=root / "data" / "goldbach_train_v2_phase0.bin")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=root / "results" / "phase1_gpu" / "train_report.json",
        help="Phase 1 report for residual_gain (phase2) or unused in phase3",
    )
    parser.add_argument(
        "--phase2-report",
        type=Path,
        default=root / "results" / "phase2_bs2048" / "train_report.json",
        help="Phase 2 report used to compute attention_gain in phase3",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--cross-attn-layers",
        type=int,
        default=1,
        help="Phase 3: number of Prime-Modulus cross-attention layers",
    )
    parser.add_argument(
        "--self-attn-layers",
        type=int,
        default=1,
        help="Phase 3: number of Prime-Prime self-attention layers",
    )
    parser.add_argument(
        "--stack-mode",
        choices=("sequential", "paired"),
        default="sequential",
        help="Phase 3: sequential (all cross then all self) or paired (cross->self units)",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("none", "cosine"),
        default="none",
        help="Learning rate schedule: fixed (none) or warmup+cosine",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="LR warmup steps; default max(500, steps_per_epoch) when cosine",
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=None,
        help="Minimum LR for cosine decay (default: lr * 0.01)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    """auto -> cuda if available, else cpu."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> int:
    args = parse_args()
    if not args.data.exists():
        print(f"Dataset not found: {args.data}", file=sys.stderr)
        return 1

    if args.output_dir is None:
        args.output_dir = _repo_root() / "results" / args.phase

    # Fix random seed for reproducible train/val split and initialization.
    # 固定随机种子，保证划分与初始化可复现
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    dataset = GoldbachBinDataset(args.data)
    top_k = dataset.candidate_count or 500

    # 90/10 train-validation split (same seed as Phase 1 for fair comparison).
    # 90/10 训练验证划分（与 Phase 1 相同 seed，便于公平对比）
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

    model = build_model(
        args.phase,
        cross_attn_layers=args.cross_attn_layers,
        self_attn_layers=args.self_attn_layers,
        stack_mode=args.stack_mode,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    phase1_metrics = load_phase1_baseline_metrics(args.baseline_report)
    phase2_metrics = load_baseline_metrics(args.phase2_report)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_steps
    if warmup_steps is None and args.lr_schedule == "cosine":
        warmup_steps = max(500, steps_per_epoch)
    elif warmup_steps is None:
        warmup_steps = 0
    eta_min = args.eta_min if args.eta_min is not None else args.lr * 0.01
    scheduler = build_lr_scheduler(
        optimizer,
        base_lr=args.lr,
        lr_schedule=args.lr_schedule,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        eta_min=eta_min,
    )

    history: list[dict] = []
    best_score = -1.0
    started = time.perf_counter()

    print(f"Training {args.phase} on {device}", flush=True)
    print(f"  train samples: {train_size:,}", flush=True)
    print(f"  val samples:   {val_size:,}", flush=True)
    print(f"  train batches/epoch: {steps_per_epoch}", flush=True)
    print(f"  val batches/epoch:   {len(val_loader)}", flush=True)
    if args.phase == "phase3":
        print(
            f"  attention depth: cross={args.cross_attn_layers} "
            f"self={args.self_attn_layers} stack={args.stack_mode}",
            flush=True,
        )
    if args.lr_schedule == "cosine":
        print(
            f"  lr schedule: cosine  warmup_steps={warmup_steps}  "
            f"eta_min={eta_min:g}  total_steps={total_steps}",
            flush=True,
        )

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.perf_counter()
        train_loss, epoch_lr = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            args.phase,
            scheduler,
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device, top_k, args.phase)
        epoch_elapsed = time.perf_counter() - epoch_t0

        epoch_record = {
            "epoch": epoch,
            "train_bce_loss": train_loss,
            "learning_rate": epoch_lr,
            **val_metrics,
        }
        if args.phase == "phase2":
            epoch_record["residual_gain"] = compute_residual_gain(val_metrics, phase1_metrics)
        # Phase 3: compare against Phase 2 final epoch (--phase2-report).
        # Phase 3：相对 Phase 2 末 epoch 计算 attention_gain
        if args.phase == "phase3":
            epoch_record["attention_gain"] = compute_attention_gain(val_metrics, phase2_metrics)
        history.append(epoch_record)

        gain_text = ""
        if epoch_record.get("residual_gain"):
            gain = epoch_record["residual_gain"]
            gain_text = (
                f"  d_top1={gain['top1_valid_rate']:+.4f}"
                f"  d_top5={gain['top5_valid_rate']:+.4f}"
                f"  d_checks={gain['avg_checks_to_success']:+.3f}"
            )
        if epoch_record.get("attention_gain"):
            gain = epoch_record["attention_gain"]
            gain_text = (
                f"  attn_d_top1={gain['top1_valid_rate']:+.4f}"
                f"  attn_d_top5={gain['top5_valid_rate']:+.4f}"
                f"  attn_d_checks={gain['avg_checks_to_success']:+.3f}"
            )
        lr_text = f"  lr={epoch_lr:.2e}" if args.lr_schedule == "cosine" else ""
        print(
            f"epoch {epoch:02d}  "
            f"train_bce={train_loss:.4f}  "
            f"val_bce={val_metrics['bce_loss']:.4f}  "
            f"top1={val_metrics['top1_valid_rate']:.4f}  "
            f"top5={val_metrics['top5_valid_rate']:.4f}  "
            f"topK={val_metrics['topk_valid_rate']:.4f}  "
            f"avg_checks={val_metrics['avg_checks_to_success']:.2f}  "
            f"epoch_time={epoch_elapsed:.1f}s"
            f"{lr_text}"
            f"{gain_text}",
            flush=True,
        )

        # Checkpoint by top1_valid_rate (primary ranking quality metric).
        # 以 top1_valid_rate 作为最优模型选择标准
        score = val_metrics["top1_valid_rate"]
        if score >= best_score:
            best_score = score
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, val_metrics, args.phase)

    elapsed = time.perf_counter() - started
    final_metrics = history[-1]
    save_checkpoint(args.output_dir / "last.pt", model, optimizer, args.epochs, final_metrics, args.phase)

    phase_labels = {
        "phase1": "phase1_sieve_baseline",
        "phase2": "phase2_sieve_residual",
        "phase3": "phase3_igot_transformer",
    }
    report = {
        "phase": phase_labels[args.phase],
        "model_phase": args.phase,
        "data": str(args.data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "warmup_steps": warmup_steps if args.lr_schedule == "cosine" else 0,
        "eta_min": eta_min if args.lr_schedule == "cosine" else None,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "device": str(device),
        "elapsed_seconds": round(elapsed, 3),
        "baseline_report": str(args.baseline_report) if args.phase == "phase2" else None,
        "phase2_report": str(args.phase2_report) if args.phase == "phase3" else None,
        "history": history,
    }
    if args.phase == "phase3":
        report["cross_attn_layers"] = args.cross_attn_layers
        report["self_attn_layers"] = args.self_attn_layers
        report["stack_mode"] = args.stack_mode
    if hasattr(model, "modulus_weight_table"):
        report["final_modulus_weights"] = model.modulus_weight_table()
    if args.phase == "phase2":
        report["final_residual_gain"] = compute_residual_gain(final_metrics, phase1_metrics)
    if args.phase == "phase3":
        report["final_attention_gain"] = compute_attention_gain(final_metrics, phase2_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "train_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("Training complete.")
    print(f"  best checkpoint: {args.output_dir / 'best.pt'}")
    print(f"  report:          {report_path}")
    print(f"  elapsed:         {elapsed:.1f}s")
    if args.phase == "phase2" and report.get("final_residual_gain"):
        gain = report["final_residual_gain"]
        print("Residual gain vs Phase 1:")
        print(f"  top1:   {gain['top1_valid_rate']:+.4f}")
        print(f"  top5:   {gain['top5_valid_rate']:+.4f}")
        print(f"  topK:   {gain['topk_valid_rate']:+.4f}")
        print(f"  checks: {gain['avg_checks_to_success']:+.3f}")
        print(f"  bce:    {gain['bce_loss']:+.4f}")
    if args.phase == "phase3" and report.get("final_attention_gain"):
        gain = report["final_attention_gain"]
        print("Attention gain vs Phase 2:")
        print(f"  top1:   {gain['top1_valid_rate']:+.4f}")
        print(f"  top5:   {gain['top5_valid_rate']:+.4f}")
        print(f"  topK:   {gain['topk_valid_rate']:+.4f}")
        print(f"  checks: {gain['avg_checks_to_success']:+.3f}")
        print(f"  bce:    {gain['bce_loss']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
