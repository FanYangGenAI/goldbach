"""Evaluation metrics for IGOT candidate ranking.

IGOT 候选排序评估指标：top1/top5/topK、平均检查次数、按 bucket 分层统计。
"""

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
    """Aggregated ranking metrics for one evaluation pass.

    一次验证/评估的汇总指标。
    """

    bce_loss: float
    top1_valid_rate: float          # fraction where rank-1 candidate is valid / top1 命中率
    top5_valid_rate: float          # fraction with any valid in top-5 / top5 命中率
    topk_valid_rate: float          # fraction with any valid in top-K / topK 覆盖率
    avg_checks_to_success: float    # mean 1-based rank of first valid / 平均检查次数
    failure_rate_under_k: float     # 1 - topk_valid_rate / topK 内未覆盖比例
    sample_count: int
    by_bucket: dict[int, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly dict for train_report.json."""
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
    """Assign each even N to a log-scale bucket for stratified metrics.

    按偶数 N 的对数区间分配 bucket id，用于分层统计。
    """
    bucket_ids = torch.zeros_like(even_numbers)
    for bucket_id, low, high in BUCKET_BOUNDS:
        mask = (even_numbers >= low) & (even_numbers < high)
        bucket_ids[mask] = bucket_id
    return bucket_ids


def _checks_to_first_success(ranked_mask: torch.Tensor) -> torch.Tensor:
    """Return 1-based index of first valid candidate, or 0 if none.

    返回第一个有效候选的 1-based 排名；若无有效候选则返回 0。
    """
    has_valid = ranked_mask.any(dim=1)
    # argmax on float mask returns index of first 1 (since only 0/1 values).
    # 在 0/1 掩码上 argmax 等价于找第一个 1 的位置
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
    """Compute ranking metrics from raw logits and multi-hot valid_mask.

    从 logits 和多标签 valid_mask 计算排序指标。

    Args:
        logits:       [B, K] unnormalized scores per candidate
        valid_mask:   [B, K] 1.0 where (N-p_i, p_i) is a valid Goldbach pair
        bce_loss:     optional pre-aggregated BCE (passed from evaluate loop)
        even_numbers: optional [B] for per-bucket breakdown
        top_k:        ranking cutoff K (default: full candidate count)
    """
    if logits.shape != valid_mask.shape:
        raise ValueError("logits and valid_mask must have the same shape")
    if top_k is None:
        top_k = logits.shape[1]

    # --- Step 1: sort candidates by descending logit ---
    # --- 步骤 1：按 logit 降序排列候选 ---
    ranked_indices = torch.argsort(logits, dim=1, descending=True)

    # Reorder valid_mask to match ranked order.
    # 按排序后的顺序重排 valid_mask
    ranked_valid = torch.gather(valid_mask, 1, ranked_indices)

    # --- Step 2: top1 / top5 / topK hit rates ---
    # --- 步骤 2：计算 top1 / top5 / topK 命中率 ---
    top1 = ranked_valid[:, 0].float().mean().item()
    top5 = ranked_valid[:, : min(5, top_k)].any(dim=1).float().mean().item()
    topk = ranked_valid[:, :top_k].any(dim=1).float().mean().item()

    # --- Step 3: average checks until first valid candidate ---
    # --- 步骤 3：平均需要检查多少个候选才能找到第一个有效对 ---
    checks = _checks_to_first_success(ranked_valid[:, :top_k])
    covered = checks > 0
    if covered.any():
        avg_checks = checks[covered].float().mean().item()
    else:
        # No sample has valid in top-K; use K as pessimistic fallback.
        # 若 topK 内全无有效对，用 K 作为悲观上界
        avg_checks = float(top_k)
    failure_rate = 1.0 - topk

    # --- Step 4: optional per-bucket breakdown ---
    # --- 步骤 4：可选的按 N 区间分层统计 ---
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
