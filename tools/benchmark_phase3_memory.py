#!/usr/bin/env python3
"""Benchmark Phase 3 forward+backward peak memory and step time by batch size.

Phase 3 显存与单步耗时基准测试。
用于验证 SDPA（FlashAttention）是否允许在 K=500 自注意力下使用更大 batch。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "model"))

from igot import IGOTModel  # noqa: E402


def peak_mb() -> float:
    """Peak CUDA memory allocated since last reset, in MiB.

    自上次 reset 以来 CUDA 峰值已分配显存（MiB）。
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def run_step(batch_size: int, device: torch.device, warmup: int, repeats: int) -> dict:
    """Run one training step benchmark: forward + BCE + backward + optimizer.

    模拟一个完整训练步：前向 + BCE + 反向 + 优化器更新。
    """
    model = IGOTModel().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Synthetic batch matching real training tensor shapes.
    # 合成 batch，形状与真实训练一致
    even = torch.randint(100, 10_000, (batch_size,), dtype=torch.int64, device=device)
    even = even - (even % 2)  # force even N
    bits = torch.rand(batch_size, 64, device=device)
    mods = torch.randint(0, 3, (batch_size, 10), dtype=torch.int64, device=device)
    density = torch.full((batch_size,), 0.2, device=device)
    target = torch.zeros(batch_size, 500, device=device)

    # Warmup: trigger CUDA kernel compilation / cuDNN autotune before timing.
    # 预热：触发 CUDA 编译 / cuDNN 自动调优，避免污染计时
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(even, bits, mods, density)
        loss = loss_fn(outputs["final_logits"], target)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    timings: list[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(even, bits, mods, density)
        loss = loss_fn(outputs["final_logits"], target)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)

    return {
        "batch_size": batch_size,
        "mean_step_ms": 1000.0 * sum(timings) / len(timings),
        "peak_memory_mb": peak_mb(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark IGOT Phase 3 step time and peak GPU memory",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--repeats", type=int, default=5, help="timed steps after warmup")
    parser.add_argument("--warmup", type=int, default=5, help="untimed warmup steps (use >=5 for batch 2048)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
        print(f"SDPA flash enabled: {torch.backends.cuda.flash_sdp_enabled()}", flush=True)
        print(f"SDPA mem-efficient enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}", flush=True)

    results = []
    for batch_size in args.batch_sizes:
        try:
            result = run_step(batch_size, device, args.warmup, args.repeats)
            results.append(result)
            print(
                f"batch={result['batch_size']:4d}  "
                f"step={result['mean_step_ms']:.1f}ms  "
                f"peak_mem={result['peak_memory_mb']:.0f}MB",
                flush=True,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"batch={batch_size:4d}  OOM", flush=True)
            results.append({
                "batch_size": batch_size,
                "mean_step_ms": None,
                "peak_memory_mb": None,
                "oom": True,
            })

    out = Path(_REPO / "results" / "phase3_flash_benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "results": results}, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
