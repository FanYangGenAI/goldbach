#!/usr/bin/env python3
"""Smoke-check eligibility mask on validation bucket 1 (WP-F F-1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split

_REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (_REPO_ROOT / "model", _REPO_ROOT / "train"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset import GoldbachBinDataset  # noqa: E402
from metrics import bucket_ids_for_evens, compute_ranking_metrics  # noqa: E402
from sieve_residual import SieveResidualModel  # noqa: E402


def main() -> None:
    data = _REPO_ROOT / "data" / "goldbach_train_v2_phase0.bin"
    dataset = GoldbachBinDataset(data)
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    _, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(7),
    )
    loader = DataLoader(val_set, batch_size=2048, shuffle=False)

    model = SieveResidualModel()
    model.eval()

    logits_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    even_batches: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["even"], batch["bits"], batch["mods"], batch["density"])
            logits_batches.append(outputs["final_logits"])
            mask_batches.append(batch["valid_mask"])
            even_batches.append(batch["even"])

    logits = torch.cat(logits_batches, dim=0)
    masks = torch.cat(mask_batches, dim=0)
    evens = torch.cat(even_batches, dim=0)
    metrics = compute_ranking_metrics(logits, masks, even_numbers=evens, top_k=500)

    bucket_ids = bucket_ids_for_evens(evens)
    bucket1 = metrics.by_bucket.get(1, {})
    report = {
        "bucket1_avg_checks": bucket1.get("avg_checks_to_success"),
        "bucket1_sample_count": bucket1.get("sample_count"),
        "global_avg_checks": metrics.avg_checks_to_success,
        "global_top1": metrics.top1_valid_rate,
        "acceptance_bucket1_le_6": bucket1.get("avg_checks_to_success", 999) <= 6,
    }
    out = _REPO_ROOT / "results" / "wpf" / "eligibility_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
