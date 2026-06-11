#!/usr/bin/env python3
"""Survivor-order shuffle analysis for NULL-0 (WP-C ruler 3 preview, N-2 closeout)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

_REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (_REPO_ROOT / "model", _REPO_ROOT / "train", _REPO_ROOT / "generator" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constants import CANDIDATE_PRIME_COUNT  # noqa: E402
from dataset import GoldbachBinDataset  # noqa: E402
from metrics import compute_ranking_metrics  # noqa: E402
from null_model import NullModel  # noqa: E402
def _build_val_subset(dataset: GoldbachBinDataset, val_fraction: float, seed: int) -> Subset:
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    _, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return val_set


def collect_tensors(subset: Subset, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    evens: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for batch in loader:
        evens.append(batch["even"])
        masks.append(batch["valid_mask"])
    return torch.cat(evens, dim=0), torch.cat(masks, dim=0)


def survivor_mask(null_model: NullModel, even_numbers: np.ndarray) -> np.ndarray:
    """Eligible candidates with sieve row all-zero (z_ij = 0 for all j)."""
    offsets = even_numbers[:, np.newaxis] - null_model.cands[np.newaxis, :]
    eligible = offsets >= null_model.min_offset
    divisible = np.zeros(offsets.shape, dtype=bool)
    for modulus in null_model.mods:
        divisible |= offsets % int(modulus) == 0
    return eligible & ~divisible


def top1_after_survivor_shuffle(
    base_ranked: np.ndarray,
    valid_mask: np.ndarray,
    survivors: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Shuffle survivor slots in a fixed base ranking; return batch top1 rate."""
    batch_size = base_ranked.shape[0]
    hits = 0.0
    for row in range(batch_size):
        order = base_ranked[row].copy()
        survivor_positions = [pos for pos, cand in enumerate(order) if survivors[row, cand]]
        if len(survivor_positions) >= 2:
            survivor_indices = order[survivor_positions]
            shuffled_indices = survivor_indices[rng.permutation(len(survivor_indices))]
            order[survivor_positions] = shuffled_indices
        hits += valid_mask[row, order[0]]
    return hits / batch_size


def parse_args() -> argparse.Namespace:
    root = _REPO_ROOT
    parser = argparse.ArgumentParser(description="NULL-0 survivor shuffle top1 distribution")
    parser.add_argument("--data", type=Path, default=root / "data" / "goldbach_train_v2_phase0.bin")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--shuffles", type=int, default=100)
    parser.add_argument("--phase3-top1", type=float, default=0.44634464383125305)
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "null")
    parser.add_argument("--figure", type=Path, default=root / "docs" / "figures" / "null_shuffle_top1_dist.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)

    dataset = GoldbachBinDataset(args.data)
    val_subset = _build_val_subset(dataset, args.val_fraction, args.seed)
    val_even, val_mask = collect_tensors(val_subset, args.batch_size)

    null0 = NullModel()
    even_np = val_even.numpy()
    mask_np = val_mask.numpy()
    null0_logits = torch.from_numpy(null0.predict_logits_batch(even_np)).float()
    survivors = survivor_mask(null0, even_np)

    baseline = compute_ranking_metrics(null0_logits, val_mask, even_numbers=val_even, top_k=CANDIDATE_PRIME_COUNT)
    baseline_top1 = baseline.top1_valid_rate
    base_ranked = np.argsort(-null0_logits.numpy(), axis=1)

    rng = np.random.default_rng(args.seed + 1000)
    shuffle_top1: list[float] = []
    for _ in range(args.shuffles):
        shuffle_top1.append(
            top1_after_survivor_shuffle(base_ranked, mask_np, survivors, rng)
        )

    samples = np.asarray(shuffle_top1, dtype=np.float64)
    phase3 = args.phase3_top1
    percentile = float((samples < phase3).mean() * 100.0)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_null0_top1": baseline_top1,
        "shuffle_count": args.shuffles,
        "shuffle_top1_mean": float(samples.mean()),
        "shuffle_top1_std": float(samples.std()),
        "shuffle_top1_min": float(samples.min()),
        "shuffle_top1_max": float(samples.max()),
        "phase3_reference_top1": phase3,
        "phase3_percentile_among_shuffles": percentile,
        "interpretation": (
            f"Phase 3 top1 ({phase3:.4f}) exceeds {percentile:.1f}% of survivor-shuffle draws."
        ),
    }

    json_path = args.output_dir / "shuffle_survivor_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(samples * 100.0, bins=20, color="#4C72B0", edgecolor="white", alpha=0.9)
    ax.axvline(baseline_top1 * 100.0, color="#55A868", linestyle="--", linewidth=2, label=f"NULL-0 ({baseline_top1*100:.2f}%)")
    ax.axvline(phase3 * 100.0, color="#C44E52", linestyle="-", linewidth=2, label=f"Phase 3 best ({phase3*100:.2f}%)")
    ax.set_xlabel("top1 valid rate (%)")
    ax.set_ylabel("count")
    ax.set_title("NULL-0 survivor-order shuffle (100 draws)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figure, dpi=150)
    plt.close(fig)

    print(json.dumps(report, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {args.figure}")


if __name__ == "__main__":
    main()
