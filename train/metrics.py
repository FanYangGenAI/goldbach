"""Evaluation metrics for IGOT candidate ranking."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

_GENERATOR_PYTHON = Path(__file__).resolve().parents[1] / "generator" / "python"
if str(_GENERATOR_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GENERATOR_PYTHON))

from constants import BUCKET_BOUNDS  # noqa: E402


@dataclass
class RankingMetrics:
    bce_loss: float
    top1_valid_rate: float
    top5_valid_rate: float
    topk_valid_rate: float
    avg_checks_to_success: float
    failure_rate_under_k: float
    sample_count: int
    by_bucket: dict[int, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "bce_loss": self.bce_loss,
            "top1_valid_rate": self.top1_valid_rate,
            "top5_valid_rate": self.top5_valid_rate,
            "topk_valid_rate": self.topk_valid_rate,
            "avg_checks_to_success": self.avg_checks_to_success,
            "failure_rate_under_k": self.failure_rate_under_k,
            "sample_count": self.sample_count,
            "by_bucket": self.by_bucket,
        }


def bucket_ids_for_evens(even_numbers: torch.Tensor) -> torch.Tensor:
    bucket_ids = torch.zeros_like(even_numbers)
    for bucket_id, low, high in BUCKET_BOUNDS:
        mask = (even_numbers >= low) & (even_numbers < high)
        bucket_ids[mask] = bucket_id
    return bucket_ids


def _checks_to_first_success(ranked_mask: torch.Tensor) -> torch.Tensor:
    """Return 1-based index of first valid candidate, or 0 if none."""
    has_valid = ranked_mask.any(dim=1)
    first_hit = ranked_mask.to(dtype=torch.float32).argmax(dim=1) + 1
    return torch.where(has_valid, first_hit, torch.zeros_like(first_hit))


def compute_ranking_metrics(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    bce_loss: float | None = None,
    even_numbers: torch.Tensor | None = None,
    top_k: int | None = None,
) -> RankingMetrics:
    if logits.shape != valid_mask.shape:
        raise ValueError("logits and valid_mask must have the same shape")
    if top_k is None:
        top_k = logits.shape[1]

    ranked_indices = torch.argsort(logits, dim=1, descending=True)
    ranked_valid = torch.gather(valid_mask, 1, ranked_indices)

    top1 = ranked_valid[:, 0].float().mean().item()
    top5 = ranked_valid[:, : min(5, top_k)].any(dim=1).float().mean().item()
    topk = ranked_valid[:, :top_k].any(dim=1).float().mean().item()

    checks = _checks_to_first_success(ranked_valid[:, :top_k])
    covered = checks > 0
    if covered.any():
        avg_checks = checks[covered].float().mean().item()
    else:
        avg_checks = float(top_k)
    failure_rate = 1.0 - topk

    by_bucket: dict[int, dict[str, float]] = {}
    if even_numbers is not None:
        bucket_ids = bucket_ids_for_evens(even_numbers)
        for bucket_id in torch.unique(bucket_ids).tolist():
            bucket_id = int(bucket_id)
            mask = bucket_ids == bucket_id
            bucket_ranked = ranked_valid[mask][:, :top_k]
            bucket_topk = bucket_ranked.any(dim=1).float().mean().item()
            bucket_checks = _checks_to_first_success(bucket_ranked)
            bucket_covered = bucket_checks > 0
            if bucket_covered.any():
                bucket_avg_checks = bucket_checks[bucket_covered].float().mean().item()
            else:
                bucket_avg_checks = float(top_k)
            by_bucket[bucket_id] = {
                "topk_valid_rate": bucket_topk,
                "avg_checks_to_success": bucket_avg_checks,
                "sample_count": int(mask.sum().item()),
            }

    return RankingMetrics(
        bce_loss=0.0 if bce_loss is None else bce_loss,
        top1_valid_rate=top1,
        top5_valid_rate=top5,
        topk_valid_rate=topk,
        avg_checks_to_success=avg_checks,
        failure_rate_under_k=failure_rate,
        sample_count=int(logits.shape[0]),
        by_bucket=by_bucket,
    )
