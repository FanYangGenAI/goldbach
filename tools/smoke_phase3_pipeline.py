#!/usr/bin/env python3
"""Step-by-step Phase 3 pipeline smoke test with timing and flushed logs.

Phase 3 链路分步冒烟测试：逐步计时并立即 flush 日志，定位卡住位置。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "model"))
sys.path.insert(0, str(_REPO / "train"))

from dataset import GoldbachBinDataset  # noqa: E402
from igot import IGOTModel  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def timed(label: str):
    class _Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            log(f"[START] {label}")
            return self

        def __exit__(self, *args):
            log(f"[DONE]  {label}  ({time.perf_counter() - self.t0:.2f}s)")

    return _Timer()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 pipeline smoke test")
    parser.add_argument("batch_size", type=int, nargs="?", default=2048)
    parser.add_argument(
        "--cross-layers",
        type=int,
        default=1,
        help="Number of cross-attention layers",
    )
    parser.add_argument(
        "--self-layers",
        type=int,
        default=1,
        help="Number of self-attention layers",
    )
    parser.add_argument(
        "--stack-mode",
        choices=("sequential", "paired"),
        default="sequential",
        help="Attention stack mode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = _REPO / "data" / "goldbach_train_v2_phase0.bin"

    log(
        f"PyTorch {torch.__version__}  device={device}  batch_size={args.batch_size}  "
        f"cross={args.cross_layers}  self={args.self_layers}  stack={args.stack_mode}"
    )
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name()}")
        log(f"flash_sdp={torch.backends.cuda.flash_sdp_enabled()}")

    with timed("load dataset"):
        dataset = GoldbachBinDataset(data_path)
        val_size = max(1, int(len(dataset) * 0.1))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(7),
        )
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    log(f"  train_batches={len(train_loader)}  val_batches={len(val_loader)}")

    with timed("build model + move to GPU"):
        model = IGOTModel(
            cross_attn_layers=args.cross_layers,
            self_attn_layers=args.self_layers,
            stack_mode=args.stack_mode,
        ).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)
        loss_fn = torch.nn.BCEWithLogitsLoss()

    with timed("first train batch (forward+backward)"):
        batch = next(iter(train_loader))
        device_batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            device_batch["even"],
            device_batch["bits"],
            device_batch["mods"],
            device_batch["density"],
        )
        loss = loss_fn(outputs["final_logits"], device_batch["valid_mask"].float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        log(f"  loss={loss.item():.4f}  logits_shape={tuple(outputs['final_logits'].shape)}")

    with timed("second train batch (post-warmup timing)"):
        batch = next(iter(train_loader))
        device_batch = {k: v.to(device) for k, v in batch.items()}
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            device_batch["even"],
            device_batch["bits"],
            device_batch["mods"],
            device_batch["density"],
        )
        loss = loss_fn(outputs["final_logits"], device_batch["valid_mask"].float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        log(f"  step_time={(time.perf_counter() - t0) * 1000:.1f}ms")

    with timed("one val batch"):
        batch = next(iter(val_loader))
        model.eval()
        with torch.no_grad():
            device_batch = {k: v.to(device) for k, v in batch.items()}
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(
                device_batch["even"],
                device_batch["bits"],
                device_batch["mods"],
                device_batch["density"],
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            log(f"  val_forward={(time.perf_counter() - t0) * 1000:.1f}ms")

    log("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
