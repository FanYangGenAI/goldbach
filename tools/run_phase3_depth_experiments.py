#!/usr/bin/env python3
"""Autonomous runner for Phase 3 depth experiments (Stages I–IV).

Runs experiments sequentially, plots curves, updates docs/phase3_deep_experiment_plan.md
after each sub-experiment, and logs issues to results/phase3_depth/experiment_log.md.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PYTHON = _REPO / ".venv" / "Scripts" / "python.exe"
if not _PYTHON.exists():
    _PYTHON = Path(sys.executable)

PLAN_MD = _REPO / "docs" / "phase3_deep_experiment_plan.md"
FIGURES_DIR = _REPO / "docs" / "figures"
DEPTH_ROOT = _REPO / "results" / "phase3_depth"
STATE_PATH = DEPTH_ROOT / "experiment_state.json"
LOG_PATH = DEPTH_ROOT / "experiment_log.md"
PHASE2_REPORT = _REPO / "results" / "phase2_bs2048" / "train_report.json"

TRAIN_SAMPLES = 90_000
BASE_LR = 0.0025
EPOCHS = 50
GPU_MEM_LIMIT_MB = 22_000  # RTX 4090 24GB, leave headroom


@dataclass
class RunConfig:
    run_id: str
    stage: str
    cross_attn_layers: int
    self_attn_layers: int
    stack_mode: str
    suggested_batch: int
    output_dir: Path
    figure_stem: str


@dataclass
class RunResult:
    run_id: str
    stage: str
    config: dict
    status: str  # ok | failed | retried
    batch_size: int = 0
    epochs: int = EPOCHS
    elapsed_seconds: float = 0.0
    best_epoch: int = 0
    best_top1: float = 0.0
    best_top5: float = 0.0
    best_bce: float = 0.0
    best_checks: float = 0.0
    final_top1: float = 0.0
    final_top5: float = 0.0
    has_nan: bool = False
    report_path: str = ""
    figure_bce: str = ""
    figure_rank: str = ""
    notes: list[str] = field(default_factory=list)


def log_event(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"- **{timestamp}**: {message}\n"
    DEPTH_ROOT.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text("# Phase 3 Depth Experiment Log\n\n", encoding="utf-8")
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"runs": {}, "stage_decisions": {}}


def save_state(state: dict) -> None:
    DEPTH_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_cmd(cmd: list[str], label: str) -> int:
    import os

    log_event(f"START `{label}`: {' '.join(cmd)}")
    started = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(cmd, cwd=_REPO, env=env)
    elapsed = time.perf_counter() - started
    log_event(f"DONE `{label}` exit={proc.returncode} elapsed={elapsed:.1f}s")
    return proc.returncode


def warmup_steps_for(batch_size: int) -> int:
    steps = math.ceil(TRAIN_SAMPLES / batch_size)
    return max(500, steps)


def pick_batch_size(
    config: RunConfig,
    candidates: list[int],
) -> tuple[int, list[dict]]:
    """Benchmark candidate batch sizes; return largest that fits GPU."""
    sys.path.insert(0, str(_REPO / "model"))
    sys.path.insert(0, str(_REPO / "tools"))
    import torch

    from benchmark_phase3_memory import run_step  # type: ignore  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench_results: list[dict] = []
    chosen = candidates[-1]

    for batch_size in sorted(candidates):
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            result = run_step(
                batch_size,
                device,
                warmup=3,
                repeats=2,
                cross_attn_layers=config.cross_attn_layers,
                self_attn_layers=config.self_attn_layers,
                stack_mode=config.stack_mode,
            )
            result["oom"] = False
            bench_results.append(result)
            if result["peak_memory_mb"] <= GPU_MEM_LIMIT_MB:
                chosen = batch_size
            else:
                log_event(
                    f"batch={batch_size} peak={result['peak_memory_mb']:.0f}MB exceeds limit; skip"
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bench_results.append({"batch_size": batch_size, "oom": True})
            log_event(f"OOM at batch={batch_size} for {config.run_id}")

    bench_path = config.output_dir / "benchmark.json"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    bench_path.write_text(json.dumps(bench_results, indent=2), encoding="utf-8")
    return chosen, bench_results


def smoke_test(config: RunConfig, batch_size: int) -> bool:
    code = run_cmd(
        [
            str(_PYTHON),
            str(_REPO / "tools" / "smoke_phase3_pipeline.py"),
            str(batch_size),
            "--cross-layers",
            str(config.cross_attn_layers),
            "--self-layers",
            str(config.self_attn_layers),
            "--stack-mode",
            config.stack_mode,
        ],
        f"smoke {config.run_id}",
    )
    return code == 0


def train_run(config: RunConfig, batch_size: int, lr: float = BASE_LR) -> int:
    warmup = warmup_steps_for(batch_size)
    return run_cmd(
        [
            str(_PYTHON),
            str(_REPO / "train" / "train.py"),
            "--phase",
            "phase3",
            "--cross-attn-layers",
            str(config.cross_attn_layers),
            "--self-attn-layers",
            str(config.self_attn_layers),
            "--stack-mode",
            config.stack_mode,
            "--epochs",
            str(EPOCHS),
            "--batch-size",
            str(batch_size),
            "--lr",
            str(lr),
            "--lr-schedule",
            "cosine",
            "--warmup-steps",
            str(warmup),
            "--device",
            "cuda",
            "--output-dir",
            str(config.output_dir),
            "--phase2-report",
            str(PHASE2_REPORT),
        ],
        f"train {config.run_id} bs={batch_size} lr={lr}",
    )


def analyze_report(report_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    history = report["history"]
    has_nan = any(
        math.isnan(row.get("train_bce_loss", 0))
        or math.isnan(row.get("bce_loss", 0))
        for row in history
    )
    best = max(history, key=lambda row: row["top1_valid_rate"])
    final = history[-1]
    return {
        "has_nan": has_nan,
        "best_epoch": best["epoch"],
        "best_top1": best["top1_valid_rate"],
        "best_top5": best["top5_valid_rate"],
        "best_bce": best["bce_loss"],
        "best_checks": best["avg_checks_to_success"],
        "final_top1": final["top1_valid_rate"],
        "final_top5": final["top5_valid_rate"],
        "elapsed_seconds": report.get("elapsed_seconds", 0),
        "batch_size": report.get("batch_size", 0),
    }


def plot_run(config: RunConfig, report_path: Path) -> tuple[str, str]:
    run_cmd(
        [
            str(_PYTHON),
            str(_REPO / "tools" / "plot_train_curves.py"),
            str(report_path),
            "--stem",
            config.figure_stem,
            "--output-dir",
            str(FIGURES_DIR),
        ],
        f"plot {config.run_id}",
    )
    bce = f"docs/figures/{config.figure_stem}_train_val_bce.png"
    rank = f"docs/figures/{config.figure_stem}_top1_top5_vs_epoch.png"
    return bce, rank


def execute_run(config: RunConfig, candidates: list[int]) -> RunResult:
    state = load_state()
    if config.run_id in state["runs"] and state["runs"][config.run_id].get("status") == "ok":
        log_event(f"SKIP {config.run_id} (already completed)")
        saved = state["runs"][config.run_id]
        return RunResult(**{k: saved[k] for k in RunResult.__annotations__ if k in saved})

    result = RunResult(
        run_id=config.run_id,
        stage=config.stage,
        config=asdict(config) | {"output_dir": str(config.output_dir)},
        status="failed",
    )

    batch_size, bench = pick_batch_size(config, candidates)
    if not any(r.get("batch_size") == batch_size and not r.get("oom") for r in bench):
        batch_size = config.suggested_batch
        result.notes.append(f"benchmark failed; fallback to suggested batch {batch_size}")

    if not smoke_test(config, batch_size):
        result.notes.append("smoke failed on first attempt")
        batch_size = min(candidates)
        if not smoke_test(config, batch_size):
            result.notes.append("smoke failed after batch downgrade")
            save_run_result(result)
            return result

    exit_code = train_run(config, batch_size)
    report_path = config.output_dir / "train_report.json"

    if exit_code != 0 or not report_path.exists():
        result.notes.append(f"train exit={exit_code}")
        save_run_result(result)
        return result

    metrics = analyze_report(report_path)
    result.has_nan = metrics["has_nan"]
    result.batch_size = metrics["batch_size"]
    result.elapsed_seconds = metrics["elapsed_seconds"]
    result.best_epoch = metrics["best_epoch"]
    result.best_top1 = metrics["best_top1"]
    result.best_top5 = metrics["best_top5"]
    result.best_bce = metrics["best_bce"]
    result.best_checks = metrics["best_checks"]
    result.final_top1 = metrics["final_top1"]
    result.final_top5 = metrics["final_top5"]
    result.report_path = str(report_path.relative_to(_REPO)).replace("\\", "/")

    if result.has_nan:
        log_event(f"NaN detected in {config.run_id}; retry with lr={BASE_LR/2}")
        result.notes.append("NaN detected; retry lr/2")
        result.status = "retried"
        exit_code = train_run(config, batch_size, lr=BASE_LR / 2)
        if exit_code == 0 and report_path.exists():
            metrics = analyze_report(report_path)
            result.has_nan = metrics["has_nan"]
            result.best_epoch = metrics["best_epoch"]
            result.best_top1 = metrics["best_top1"]
            result.best_top5 = metrics["best_top5"]
            result.best_bce = metrics["best_bce"]
            result.best_checks = metrics["best_checks"]
            result.final_top1 = metrics["final_top1"]
            result.final_top5 = metrics["final_top5"]
            result.elapsed_seconds = metrics["elapsed_seconds"]

    if not result.has_nan:
        result.status = "ok"
    else:
        result.notes.append("still NaN after retry")

    bce_fig, rank_fig = plot_run(config, report_path)
    result.figure_bce = bce_fig
    result.figure_rank = rank_fig
    save_run_result(result)
    update_plan_markdown()
    return result


def save_run_result(result: RunResult) -> None:
    state = load_state()
    state["runs"][result.run_id] = asdict(result)
    save_state(state)


def stage_i_configs() -> list[RunConfig]:
    specs = [
        ("A_L1", 1, 2048),
        ("A_L2", 2, 2048),
        ("A_L3", 3, 1024),
        ("A_L4", 4, 1024),
        ("A_L5", 5, 512),
        ("A_L6", 6, 512),
    ]
    return [
        RunConfig(
            run_id=f"p3-{name}",
            stage="I",
            cross_attn_layers=1,
            self_attn_layers=layers,
            stack_mode="sequential",
            suggested_batch=bs,
            output_dir=DEPTH_ROOT / "stageI" / name,
            figure_stem=f"phase3_stageI_{name}",
        )
        for name, layers, bs in specs
    ]


def best_run(runs: dict[str, dict], prefix: str, key: str = "best_top1") -> dict | None:
    matched = [r for rid, r in runs.items() if rid.startswith(prefix) and r.get("status") == "ok"]
    if not matched:
        return None
    return max(matched, key=lambda r: r[key])


def stage_ii_configs(l_self_star: int) -> list[RunConfig]:
    configs = []
    for cross in (1, 2, 3):
        name = f"B_C{cross}_L{l_self_star}"
        configs.append(
            RunConfig(
                run_id=f"p3-{name}",
                stage="II",
                cross_attn_layers=cross,
                self_attn_layers=l_self_star,
                stack_mode="sequential",
                suggested_batch=1024 if cross > 1 or l_self_star >= 4 else 2048,
                output_dir=DEPTH_ROOT / "stageII" / name,
                figure_stem=f"phase3_stageII_{name}",
            )
        )
    return configs


def stage_iii_configs() -> list[RunConfig]:
    specs = [
        ("C_L1", 1, 2048),
        ("C_L2", 2, 1024),
        ("C_L3", 3, 1024),
        ("C_L4", 4, 512),
        ("C_L5", 5, 512),
        ("C_L6", 6, 512),
    ]
    return [
        RunConfig(
            run_id=f"p3-{name}",
            stage="III",
            cross_attn_layers=depth,
            self_attn_layers=depth,
            stack_mode="paired",
            suggested_batch=bs,
            output_dir=DEPTH_ROOT / "stageIII" / name,
            figure_stem=f"phase3_stageIII_{name}",
        )
        for name, depth, bs in specs
    ]


def decide_stage_iv(state: dict) -> list[RunConfig]:
    runs = state["runs"]
    dec = state.get("stage_decisions", {})
    l_self = dec.get("L_self_star", 3)
    c_cross = dec.get("C_cross_star", 1)

    decoupled = None
    for rid, r in runs.items():
        if r.get("stage") == "II" and r.get("status") == "ok":
            if r["config"]["cross_attn_layers"] == c_cross:
                decoupled = r
                break
    tied_best = best_run(runs, "p3-C_L", "best_top1")

    if decoupled and tied_best:
        margin = tied_best["best_top1"] - decoupled["best_top1"]
        if c_cross == 1 and margin <= 0.002:
            log_event(
                "SKIP Stage IV: C_cross*=1 and tied depth does not beat decoupled optimal"
            )
            state["stage_decisions"]["stage_iv"] = "skipped"
            state["stage_decisions"]["stage_iv_reason"] = (
                "C_cross*=1; tied-depth best top1 does not exceed decoupled optimal"
            )
            save_state(state)
            return []

    # Local grid around hopeful region
    grid: list[tuple[int, int, int]] = []
    if l_self >= 3:
        grid.extend([(2, l_self, 1024), (3, l_self, 1024)])
    if l_self >= 4:
        grid.append((2, 4, 1024))
        grid.append((3, 4, 512))
    # dedupe
    seen: set[tuple[int, int]] = set()
    configs: list[RunConfig] = []
    for cross, self_l, bs in grid[:6]:
        key = (cross, self_l)
        if key in seen:
            continue
        seen.add(key)
        # skip if already covered by stage I/II best
        if cross == c_cross and self_l == l_self:
            continue
        name = f"D_C{cross}_L{self_l}"
        configs.append(
            RunConfig(
                run_id=f"p3-{name}",
                stage="IV",
                cross_attn_layers=cross,
                self_attn_layers=self_l,
                stack_mode="sequential",
                suggested_batch=bs,
                output_dir=DEPTH_ROOT / "stageIV" / name,
                figure_stem=f"phase3_stageIV_{name}",
            )
        )
    if not configs:
        state["stage_decisions"]["stage_iv"] = "skipped"
        state["stage_decisions"]["stage_iv_reason"] = "No ambiguous region; grid empty"
        save_state(state)
    return configs


def plot_stage_summary(stage: str, run_ids: list[str], out_name: str, x_label: str, x_values: list) -> None:
    import matplotlib.pyplot as plt

    state = load_state()
    runs = state["runs"]
    top1 = []
    top5 = []
    for rid in run_ids:
        r = runs.get(rid, {})
        top1.append(r.get("best_top1", 0) * 100)
        top5.append(r.get("best_top5", 0) * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_values, top1, marker="o", label="best top1 %")
    ax.plot(x_values, top5, marker="s", label="best top5 %")
    ax.set_xlabel(x_label)
    ax.set_ylabel("valid rate (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_event(f"Wrote summary figure {out}")


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_stage_i(runs: dict) -> str:
    lines = [
        "### 13.1 Stage I — 方案 A（固定 Cross=1，扫描 Self L=1..6）",
        "",
        f"**完成时间**：{datetime.now(timezone.utc).strftime('%Y-%m-%d')}  ",
        f"**L_self\\***：{load_state().get('stage_decisions', {}).get('L_self_star', '待定')}（按 best val top1 选取）",
        "",
        "| Run | L_self | batch | best ep | best top1 | best top5 | best BCE | checks | 耗时 | 曲线 |",
        "|-----|--------|-------|---------|-----------|-----------|----------|--------|------|------|",
    ]
    for i in range(1, 7):
        rid = f"p3-A_L{i}"
        r = runs.get(rid, {})
        if r.get("status") != "ok":
            lines.append(f"| `{rid}` | {i} | — | — | — | — | — | — | **{r.get('status', 'pending')}** | — |")
            continue
        fig = r.get("figure_bce", "")
        lines.append(
            f"| `{rid}` | {i} | {r.get('batch_size')} | {r.get('best_epoch')} | "
            f"{fmt_pct(r['best_top1'])} | {fmt_pct(r['best_top5'])} | {r['best_bce']:.4f} | "
            f"{r['best_checks']:.2f} | {r.get('elapsed_seconds', 0):.0f}s | "
            f"[BCE]({fig}) |"
        )

    best = best_run(runs, "p3-A_L")
    if best:
        lines.extend([
            "",
            "![Stage I depth vs metrics](../figures/phase3_stageI_self_depth_metrics.png)",
            "",
            "**结论**：",
            f"- 最优 Self 深度 **L_self* = {best['config']['self_attn_layers']}**（top1={fmt_pct(best['best_top1'])}，top5={fmt_pct(best['best_top5'])}）",
            f"- 相对 Phase 3 基线（10ep 固定 LR，top1≈43.5%）：长训练 + cosine 后 top1={'提升' if best['best_top1'] > 0.435 else '未超越'}",
            "- 若 L≥4 后 top5 增益 <0.2pp，视为深度边际收益递减",
        ])
    return "\n".join(lines)


def render_stage_ii(runs: dict, l_self: int) -> str:
    lines = [
        f"### 13.2 Stage II — 方案 B（固定 L_self*={l_self}，扫描 Cross C=1..3）",
        "",
        "| Run | C_cross | L_self | batch | best top1 | best top5 | best BCE | 曲线 |",
        "|-----|---------|--------|-------|-----------|-----------|----------|------|",
    ]
    for c in range(1, 4):
        rid = f"p3-B_C{c}_L{l_self}"
        r = runs.get(rid, {})
        if r.get("status") != "ok":
            lines.append(f"| `{rid}` | {c} | {l_self} | — | — | — | — | **{r.get('status', 'pending')}** |")
            continue
        lines.append(
            f"| `{rid}` | {c} | {l_self} | {r.get('batch_size')} | "
            f"{fmt_pct(r['best_top1'])} | {fmt_pct(r['best_top5'])} | {r['best_bce']:.4f} | "
            f"[BCE]({r.get('figure_bce', '')}) |"
        )
    best = best_run(runs, "p3-B_C")
    dec = load_state().get("stage_decisions", {})
    c_star = dec.get("C_cross_star", "?")
    lines.extend([
        "",
        f"![Stage II cross depth at L_self*={l_self}](../figures/phase3_stageII_cross_depth_at_Lselfstar.png)",
        "",
        "**结论**：",
        f"- **C_cross* = {c_star}**",
    ])
    if best:
        lines.append(
            f"- 解耦最优配置 (C={best['config']['cross_attn_layers']}, L={l_self})："
            f"top1={fmt_pct(best['best_top1'])}，top5={fmt_pct(best['best_top5'])}"
        )
    return "\n".join(lines)


def render_stage_iii(runs: dict, decoupled: dict | None) -> str:
    lines = [
        "### 13.3 Stage III — 方案 C（绑深 L_cross=L_self=1..6，paired 模式）",
        "",
        "| Run | L | batch | best top1 | best top5 | best BCE | 曲线 |",
        "|-----|---|-------|-----------|-----------|----------|------|",
    ]
    for i in range(1, 7):
        rid = f"p3-C_L{i}"
        r = runs.get(rid, {})
        if r.get("status") != "ok":
            lines.append(f"| `{rid}` | {i} | — | — | — | — | **{r.get('status', 'pending')}** |")
            continue
        lines.append(
            f"| `{rid}` | {i} | {r.get('batch_size')} | {fmt_pct(r['best_top1'])} | "
            f"{fmt_pct(r['best_top5'])} | {r['best_bce']:.4f} | [BCE]({r.get('figure_bce', '')}) |"
        )
    tied_best = best_run(runs, "p3-C_L")
    lines.extend([
        "",
        "![Stage III tied depth curve](../figures/phase3_stageIII_tied_depth_curve.png)",
        "",
        "**结论**：",
    ])
    if tied_best:
        lines.append(
            f"- 绑深最优 **L_tied* = {tied_best['config']['cross_attn_layers']}**"
            f"（top1={fmt_pct(tied_best['best_top1'])}）"
        )
    if decoupled:
        lines.append(
            f"- 解耦最优（Stage II）：top1={fmt_pct(decoupled['best_top1'])}；"
            f"绑深 {'优于' if tied_best and tied_best['best_top1'] > decoupled['best_top1'] else '未优于'}解耦组合"
        )
    return "\n".join(lines)


def render_stage_iv(runs: dict) -> str:
    dec = load_state().get("stage_decisions", {})
    if dec.get("stage_iv") == "skipped":
        return (
            "### 13.4 Stage IV — 方案 D（局部网格）\n\n"
            f"**跳过**：{dec.get('stage_iv_reason', '见 stage_decisions')}\n"
        )
    lines = [
        "### 13.4 Stage IV — 方案 D（局部二维网格）",
        "",
        "| Run | C | L | batch | best top1 | best top5 | 曲线 |",
        "|-----|---|---|-------|-----------|-----------|------|",
    ]
    for rid, r in sorted(runs.items()):
        if r.get("stage") != "IV":
            continue
        cfg = r.get("config", {})
        if r.get("status") != "ok":
            lines.append(f"| `{rid}` | — | — | — | — | — | **{r.get('status')}** |")
            continue
        lines.append(
            f"| `{rid}` | {cfg.get('cross_attn_layers')} | {cfg.get('self_attn_layers')} | "
            f"{r.get('batch_size')} | {fmt_pct(r['best_top1'])} | {fmt_pct(r['best_top5'])} | "
            f"[BCE]({r.get('figure_bce', '')}) |"
        )
    return "\n".join(lines)


def render_global_best(state: dict) -> str:
    runs = state["runs"]
    ok_runs = [r for r in runs.values() if r.get("status") == "ok"]
    if not ok_runs:
        return "### 13.5 全局最优配置（Phase 4 backbone）\n\n_待全部实验完成后填写_\n"

    best = max(ok_runs, key=lambda r: (r["best_top1"], r["best_top5"]))
    cfg = best["config"]
    checkpoint = Path(cfg["output_dir"]) / "best.pt"
    return "\n".join([
        "### 13.5 全局最优配置（Phase 4 backbone）",
        "",
        "| 项目 | 定稿值 |",
        "|------|--------|",
        f"| Run ID | `{best['run_id']}` |",
        f"| C_cross | {cfg.get('cross_attn_layers')} |",
        f"| L_self | {cfg.get('self_attn_layers')} |",
        f"| stack_mode | {cfg.get('stack_mode')} |",
        f"| best top1 | {fmt_pct(best['best_top1'])} |",
        f"| best top5 | {fmt_pct(best['best_top5'])} |",
        f"| checkpoint | `{checkpoint}` |",
        "",
        "**框架极限结论（初稿）**：",
        f"- 深度实验全局最优 top1={fmt_pct(best['best_top1'])}，top5={fmt_pct(best['best_top5'])}",
        "- top5 ~87% 平台是否打破：见上表 best top5",
        f"- Phase 4 backbone 采用 `{best['run_id']}` 的 `best.pt`",
    ])


def update_plan_markdown() -> None:
    state = load_state()
    runs = state["runs"]
    dec = state.get("stage_decisions", {})

    sections = [
        render_stage_i(runs),
        render_stage_ii(runs, dec.get("L_self_star", 0)),
        render_stage_iii(
            runs,
            best_run(runs, "p3-B_C") if dec.get("C_cross_star") else None,
        ),
        render_stage_iv(runs),
        render_global_best(state),
    ]
    body = "\n\n---\n\n".join(sections)

    text = PLAN_MD.read_text(encoding="utf-8")
    start = text.find("## 13. 阶段结果（实施后填写）")
    end = text.find("\n---\n\n## 14.")
    if start < 0 or end < 0:
        log_event("Could not locate §13 markers in plan md")
        return

    new_section = (
        "## 13. 阶段结果（实施后填写）\n\n"
        f"> 自动更新于 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"{body}\n"
    )
    PLAN_MD.write_text(text[:start] + new_section + text[end:], encoding="utf-8")
    log_event("Updated docs/phase3_deep_experiment_plan.md §13")


def update_checklist(state: dict) -> None:
    text = PLAN_MD.read_text(encoding="utf-8")
    dec = state.get("stage_decisions", {})
    checks = {
        "Stage I（方案 A）完成；**L_self\\*** 已确定": dec.get("stage_i") == "done",
        "Stage II（方案 B）完成；**C_cross\\*** 已确定": dec.get("stage_ii") == "done",
        "Stage III（方案 C）完成；绑深 vs 解耦最优已对比": dec.get("stage_iii") == "done",
        "Stage IV：已执行或书面说明跳过理由": dec.get("stage_iv") in ("done", "skipped"),
        "全局汇总：最优配置、框架极限结论、Phase 4 backbone 选定": dec.get("all_done"),
    }
    for label, done in checks.items():
        old_unchecked = f"- [ ] {label}"
        old_checked = f"- [x] {label}"
        new = f"- [x] {label}" if done else f"- [ ] {label}"
        if old_checked in text:
            text = text.replace(old_checked, new)
        elif old_unchecked in text:
            text = text.replace(old_unchecked, new)
    # Update status line
    if dec.get("all_done"):
        text = text.replace(
            "> **状态**：**Stage 0 工程就绪**（2026-06-10）；Stage I 尚未开跑。",
            "> **状态**：**全部 Stage 完成**（见 §13）。",
        )
    PLAN_MD.write_text(text, encoding="utf-8")


def run_stage_i() -> None:
    log_event("=== Stage I start ===")
    candidates_map = {
        1: [512, 1024, 2048],
        2: [512, 1024, 2048],
        3: [512, 1024],
        4: [512, 1024],
        5: [512],
        6: [512],
    }
    for config in stage_i_configs():
        layers = config.self_attn_layers
        execute_run(config, candidates_map[layers])

    state = load_state()
    best = best_run(state["runs"], "p3-A_L")
    if best:
        state["stage_decisions"]["L_self_star"] = best["config"]["self_attn_layers"]
        state["stage_decisions"]["stage_i"] = "done"
        save_state(state)
        plot_stage_summary(
            "I",
            [f"p3-A_L{i}" for i in range(1, 7)],
            "phase3_stageI_self_depth_metrics.png",
            "L_self",
            list(range(1, 7)),
        )
    update_plan_markdown()
    log_event(f"=== Stage I done; L_self*={state['stage_decisions'].get('L_self_star')} ===")


def run_stage_ii() -> None:
    state = load_state()
    l_self = state["stage_decisions"].get("L_self_star", 3)
    log_event(f"=== Stage II start; L_self*={l_self} ===")
    for config in stage_ii_configs(l_self):
        total_depth = config.cross_attn_layers + config.self_attn_layers
        if total_depth <= 3:
            candidates = [512, 1024, 2048]
        elif total_depth <= 5:
            candidates = [512, 1024]
        else:
            candidates = [512]
        execute_run(config, candidates)

    state = load_state()
    best = best_run(state["runs"], "p3-B_C")
    if best:
        state["stage_decisions"]["C_cross_star"] = best["config"]["cross_attn_layers"]
        state["stage_decisions"]["stage_ii"] = "done"
        save_state(state)
        plot_stage_summary(
            "II",
            [f"p3-B_C{c}_L{l_self}" for c in range(1, 4)],
            "phase3_stageII_cross_depth_at_Lselfstar.png",
            "C_cross",
            [1, 2, 3],
        )
    update_plan_markdown()
    log_event(f"=== Stage II done; C_cross*={state['stage_decisions'].get('C_cross_star')} ===")


def run_stage_iii() -> None:
    log_event("=== Stage III start ===")
    candidates_map = {1: [1024, 2048], 2: [512, 1024], 3: [512, 1024], 4: [512], 5: [512], 6: [512]}
    for config in stage_iii_configs():
        execute_run(config, candidates_map[config.cross_attn_layers])

    state = load_state()
    state["stage_decisions"]["stage_iii"] = "done"
    save_state(state)
    plot_stage_summary(
        "III",
        [f"p3-C_L{i}" for i in range(1, 7)],
        "phase3_stageIII_tied_depth_curve.png",
        "tied depth L",
        list(range(1, 7)),
    )
    update_plan_markdown()
    log_event("=== Stage III done ===")


def run_stage_iv() -> None:
    state = load_state()
    configs = decide_stage_iv(state)
    if not configs:
        update_plan_markdown()
        return
    log_event(f"=== Stage IV start ({len(configs)} runs) ===")
    for config in configs:
        total = config.cross_attn_layers + config.self_attn_layers
        candidates = [512, 1024] if total <= 5 else [512]
        execute_run(config, candidates)
    state = load_state()
    state["stage_decisions"]["stage_iv"] = "done"
    save_state(state)
    update_plan_markdown()
    log_event("=== Stage IV done ===")


def finalize_docs() -> None:
    state = load_state()
    state["stage_decisions"]["all_done"] = True
    save_state(state)
    update_plan_markdown()
    update_checklist(state)

    # Update phase3_igot_transformer_experiment.md brief pointer
    exp_md = _REPO / "docs" / "phase3_igot_transformer_experiment.md"
    if exp_md.exists():
        text = exp_md.read_text(encoding="utf-8")
        old = "基线（1×cross + 1×self、10 epoch、**固定 LR**）已完成；**深度实验尚未开始**。"
        new = "基线已完成；**深度实验全部完成**，结论见 `docs/phase3_deep_experiment_plan.md` §13。"
        if old in text:
            exp_md.write_text(text.replace(old, new), encoding="utf-8")

    impl_md = _REPO / "docs" / "igot_implementation_plan.md"
    if impl_md.exists():
        text = impl_md.read_text(encoding="utf-8")
        text = text.replace(
            "Phase 3b Self-Attn 深度扫描 + 长训练  ← Stage 0 就绪，待 Stage I",
            "Phase 3b Self-Attn 深度扫描 + 长训练  ← **已完成**（见 phase3_deep_experiment_plan.md §13）",
        )
        impl_md.write_text(text, encoding="utf-8")

    log_event("=== ALL STAGES COMPLETE ===")


def main() -> int:
    DEPTH_ROOT.mkdir(parents=True, exist_ok=True)
    if not PHASE2_REPORT.exists():
        log_event(f"ERROR: missing {PHASE2_REPORT}")
        return 1

    run_stage_i()
    run_stage_ii()
    run_stage_iii()
    run_stage_iv()
    finalize_docs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
