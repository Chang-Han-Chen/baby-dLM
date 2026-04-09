#!/usr/bin/env python3
"""
run_lr_sweep.py — LR calibration for ClimbMix scaling experiments (§6.3).

Supports two optimizer families:
  - adamw:   1D sweep over a scalar learning_rate grid (unchanged from original)
  - normuon: 2D sweep over (adam_mult, matrix_mult) around Karpathy reference LRs

For each (optimizer_family, model_family, size) triple, trains for LR_SWEEP_STEPS
steps with a warmup-stable schedule, parses stdout for per-step train loss and
grad norm, selects the best stable config, and persists results.

Usage:
    # Full sweep (all optimizer × model × size combinations):
    python run_lr_sweep.py

    # Single optimizer family:
    python run_lr_sweep.py --optimizer adamw

    # Single (optimizer, model, size):
    python run_lr_sweep.py --optimizer normuon --model ar --size 50M

    # Dry-run (print commands without executing):
    python run_lr_sweep.py --dry-run
"""

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import experiment_config as ec

# ---------------------------------------------------------------
# Stdout parsing
# ---------------------------------------------------------------

# train.py prints lines like:
#   step 100 | tok_epoch 0.03 | loss 7.1234 | grad_norm 1.2345 | lr 0.001000
_STEP_RE = re.compile(
    r"step\s+(\d+)\s+\|.*?"
    r"loss\s+([\d.]+)\s+\|.*?"
    r"grad_norm\s+([\d.eE+\-]+)\s+\|.*?"
    r"lr\s+([\d.eE+\-]+)"
)


@dataclass
class SweepTrace:
    """Parsed training trace for a single run."""
    model: str
    size: str
    lr: float  # scalar LR for adamw, or dummy for normuon
    optimizer_family: str = "adamw"
    adam_mult: float = 1.0
    matrix_mult: float = 1.0
    steps: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)

    @property
    def final_loss(self) -> Optional[float]:
        return self.losses[-1] if self.losses else None

    @property
    def grad_norm_stable(self) -> bool:
        """
        Heuristic: grad norm is 'stable' if no single reading in the
        second half of training exceeds 5× the median of the first half.
        """
        if len(self.grad_norms) < 4:
            return True
        mid = len(self.grad_norms) // 2
        first_half = sorted(self.grad_norms[:mid])
        median_first = first_half[len(first_half) // 2]
        if median_first == 0:
            return True
        threshold = 5.0 * median_first
        return all(gn <= threshold for gn in self.grad_norms[mid:])


def parse_stdout(text: str, model: str, size: str, lr: float,
                 optimizer_family: str = "adamw",
                 adam_mult: float = 1.0,
                 matrix_mult: float = 1.0) -> SweepTrace:
    """Extract (step, loss, grad_norm) triples from train.py stdout."""
    trace = SweepTrace(
        model=model, size=size, lr=lr,
        optimizer_family=optimizer_family,
        adam_mult=adam_mult, matrix_mult=matrix_mult,
    )
    for m in _STEP_RE.finditer(text):
        trace.steps.append(int(m.group(1)))
        trace.losses.append(float(m.group(2)))
        trace.grad_norms.append(float(m.group(3)))
    return trace


# ---------------------------------------------------------------
# LR selection
# ---------------------------------------------------------------

def select_best_lr(traces: List[SweepTrace]) -> Tuple[float, SweepTrace]:
    """
    Pick the best LR (adamw): among stable runs, choose the one with the
    lowest final train loss.  If all runs are unstable, fall back to
    the smallest LR.

    Returns (chosen_lr, chosen_trace).
    """
    stable = [t for t in traces if t.grad_norm_stable and t.final_loss is not None]
    if stable:
        best = min(stable, key=lambda t: t.final_loss)
    else:
        print("  WARNING: no stable runs found, falling back to smallest LR")
        best = min(traces, key=lambda t: t.lr)
    return best.lr, best


def select_best_normuon(traces: List[SweepTrace]) -> Tuple[dict, SweepTrace]:
    """
    Pick the best (adam_mult, matrix_mult) for normuon: among stable runs,
    choose the one with the lowest final train loss.  If all unstable,
    fall back to the (1.0, 1.0) reference setting.

    Returns (config_dict, chosen_trace).
    """
    stable = [t for t in traces if t.grad_norm_stable and t.final_loss is not None]
    if stable:
        best = min(stable, key=lambda t: t.final_loss)
    else:
        print("  WARNING: no stable normuon runs, falling back to reference (1.0, 1.0)")
        # Find the reference run
        ref = [t for t in traces if t.adam_mult == 1.0 and t.matrix_mult == 1.0]
        best = ref[0] if ref else min(traces, key=lambda t: t.final_loss or float('inf'))
    config = {"adam_mult": best.adam_mult, "matrix_mult": best.matrix_mult}
    return config, best


# ---------------------------------------------------------------
# Run one sweep
# ---------------------------------------------------------------

def run_single(
    model: str,
    size: str,
    lr: float,
    out_dir: str,
    *,
    dry_run: bool = False,
    optimizer: str = "adamw",
    adam_mult: float = 1.0,
    matrix_mult: float = 1.0,
) -> Optional[SweepTrace]:
    """Launch train.py for one configuration and return parsed trace."""
    os.makedirs(out_dir, exist_ok=True)

    warmup_iters = max(1, int(ec.LR_SWEEP_STEPS * ec.LR_SWEEP_WARMUP_FRAC))

    cmd = ec.build_command(
        model=model,
        size=size,
        out_dir=out_dir,
        max_iters=ec.LR_SWEEP_STEPS,
        lr=lr,
        warmup_iters=warmup_iters,
        warmup_stable=True,
        skip_final_eval=True,
        skip_final_checkpoint=True,
        eval_interval=ec.LR_SWEEP_STEPS + 1,
        save_interval=0,
        num_final_samples=0,
        gpt2_eval_interval=0,
        sample_interval=0,
        optimizer=optimizer,
        adam_mult=adam_mult,
        matrix_mult=matrix_mult,
    )

    if optimizer == "normuon":
        label = f"adam_mult={adam_mult}, matrix_mult={matrix_mult}"
    else:
        label = f"LR={lr:.1e}"
    print(f"  {label}  cmd: {' '.join(cmd[:6])}...")

    if dry_run:
        print(f"    [dry-run] full command: {' '.join(cmd)}")
        return None

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    if result.returncode != 0:
        print(f"    FAILED (exit {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"    stderr: {line}")
        return None

    trace = parse_stdout(
        result.stdout, model, size, lr,
        optimizer_family=optimizer,
        adam_mult=adam_mult,
        matrix_mult=matrix_mult,
    )

    if not trace.steps:
        print(f"    FAILED: no step metrics parsed from stdout "
              f"({len(result.stdout)} chars of output)")
        return None

    status = "stable" if trace.grad_norm_stable else "UNSTABLE"
    print(f"    {status} | final_loss={trace.final_loss:.4f} | "
          f"{len(trace.steps)} data points")
    return trace


# ---------------------------------------------------------------
# AdamW sweep (1D)
# ---------------------------------------------------------------

def sweep_adamw(
    model: str,
    size: str,
    out_root: str,
    *,
    dry_run: bool = False,
) -> Optional[float]:
    """
    Run the full 1D LR grid for one (adamw, model, size) pair.
    Returns the selected LR, or None in dry-run mode.
    """
    grid = ec.LR_SWEEP_GRIDS.get(model, ec.LR_SWEEP_GRID_DEFAULT)
    print(f"\n{'='*60}")
    print(f"Sweeping adamw / {model} / {size}  |  grid: {grid}")
    print(f"{'='*60}")

    traces = []
    for lr in grid:
        run_dir = os.path.join(out_root, f"adamw_{model}_{size}", f"lr_{lr:.0e}")
        trace = run_single(model, size, lr, run_dir, dry_run=dry_run,
                           optimizer="adamw")
        if trace is not None:
            traces.append(trace)

    if not traces:
        if dry_run:
            print("  [dry-run] skipping selection")
            return None
        print("  ERROR: all runs failed")
        return None

    chosen_lr, chosen_trace = select_best_lr(traces)
    print(f"\n  >> Selected LR = {chosen_lr:.1e}  "
          f"(final_loss={chosen_trace.final_loss:.4f}, "
          f"stable={chosen_trace.grad_norm_stable})")

    ec.set_calibrated_lr(model, size, chosen_lr)
    print(f"  >> Saved to calibrated_lrs.json")

    traces_path = os.path.join(out_root, f"adamw_{model}_{size}", "sweep_traces.pkl")
    with open(traces_path, "wb") as f:
        pickle.dump(traces, f)
    print(f"  >> Traces saved to {traces_path}")

    return chosen_lr


# ---------------------------------------------------------------
# NorMuon sweep (2D)
# ---------------------------------------------------------------

def sweep_normuon(
    model: str,
    size: str,
    out_root: str,
    *,
    dry_run: bool = False,
) -> Optional[dict]:
    """
    Run the 2D (adam_mult, matrix_mult) grid for one (normuon, model, size) pair.
    Returns the selected config dict, or None in dry-run mode.
    """
    adam_grid = ec.NORMUON_ADAM_MULT_GRID
    matrix_grid = ec.NORMUON_MATRIX_MULT_GRID
    print(f"\n{'='*60}")
    print(f"Sweeping normuon / {model} / {size}")
    print(f"  adam_mult grid:   {adam_grid}")
    print(f"  matrix_mult grid: {matrix_grid}")
    print(f"{'='*60}")

    traces = []
    for am in adam_grid:
        for mm in matrix_grid:
            run_dir = os.path.join(
                out_root,
                f"normuon_{model}_{size}",
                f"am_{am:.1f}_mm_{mm:.1f}",
            )
            trace = run_single(
                model, size, lr=1.0, out_dir=run_dir,
                dry_run=dry_run, optimizer="normuon",
                adam_mult=am, matrix_mult=mm,
            )
            if trace is not None:
                traces.append(trace)

    if not traces:
        if dry_run:
            print("  [dry-run] skipping selection")
            return None
        print("  ERROR: all normuon runs failed")
        return None

    chosen_config, chosen_trace = select_best_normuon(traces)
    print(f"\n  >> Selected normuon config: adam_mult={chosen_config['adam_mult']}, "
          f"matrix_mult={chosen_config['matrix_mult']}  "
          f"(final_loss={chosen_trace.final_loss:.4f}, "
          f"stable={chosen_trace.grad_norm_stable})")

    ec.set_calibrated_normuon(
        model, size,
        chosen_config["adam_mult"],
        chosen_config["matrix_mult"],
    )
    print(f"  >> Saved to calibrated_normuon.json")

    traces_path = os.path.join(out_root, f"normuon_{model}_{size}", "sweep_traces.pkl")
    with open(traces_path, "wb") as f:
        pickle.dump(traces, f)
    print(f"  >> Traces saved to {traces_path}")

    return chosen_config


# ---------------------------------------------------------------
# Legacy compat: sweep_one_pair (delegates to sweep_adamw)
# ---------------------------------------------------------------

def sweep_one_pair(
    model: str,
    size: str,
    out_root: str,
    *,
    dry_run: bool = False,
) -> Optional[float]:
    """Backward-compatible wrapper; runs the adamw sweep only."""
    return sweep_adamw(model, size, out_root, dry_run=dry_run)


# ---------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LR calibration sweep for ClimbMix scaling experiments"
    )
    parser.add_argument(
        "--optimizer", type=str, default=None,
        choices=ec.OPTIMIZER_FAMILIES,
        help="Sweep a single optimizer family (default: all)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=ec.ALL_MODELS,
        help="Sweep a single model family (default: all)",
    )
    parser.add_argument(
        "--size", type=str, default=None,
        choices=ec.CLIMBMIX_SIZES,
        help="Sweep a single size (default: all ClimbMix sizes)",
    )
    parser.add_argument(
        "--out-root", type=str, default="runs/lr_sweep",
        help="Root directory for sweep outputs",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    optimizers = [args.optimizer] if args.optimizer else ec.OPTIMIZER_FAMILIES
    models = [args.model] if args.model else ec.ALL_MODELS
    sizes = [args.size] if args.size else ec.CLIMBMIX_SIZES

    adamw_results = {}
    normuon_results = {}

    for opt in optimizers:
        for model in models:
            for size in sizes:
                if opt == "adamw":
                    chosen = sweep_adamw(model, size, args.out_root,
                                         dry_run=args.dry_run)
                    if chosen is not None:
                        adamw_results[(model, size)] = chosen
                elif opt == "normuon":
                    chosen = sweep_normuon(model, size, args.out_root,
                                           dry_run=args.dry_run)
                    if chosen is not None:
                        normuon_results[(model, size)] = chosen

    # Print summary tables
    if adamw_results:
        print(f"\n{'='*60}")
        print("SUMMARY: Calibrated AdamW Learning Rates")
        print(f"{'='*60}")
        print(f"{'model':>8s}  {'size':>5s}  {'LR':>10s}")
        print(f"{'-'*8:>8s}  {'-'*5:>5s}  {'-'*10:>10s}")
        for (model, size), lr in sorted(adamw_results.items()):
            print(f"{model:>8s}  {size:>5s}  {lr:>10.1e}")

    if normuon_results:
        print(f"\n{'='*60}")
        print("SUMMARY: Calibrated NorMuon Configs")
        print(f"{'='*60}")
        print(f"{'model':>8s}  {'size':>5s}  {'adam_mult':>10s}  {'matrix_mult':>12s}")
        print(f"{'-'*8:>8s}  {'-'*5:>5s}  {'-'*10:>10s}  {'-'*12:>12s}")
        for (model, size), cfg in sorted(normuon_results.items()):
            print(f"{model:>8s}  {size:>5s}  {cfg['adam_mult']:>10.2f}  "
                  f"{cfg['matrix_mult']:>12.2f}")

    # Write combined JSON summary
    all_results = {}
    for (m, s), lr in adamw_results.items():
        all_results[f"adamw|{m}|{s}"] = {"lr": lr}
    for (m, s), cfg in normuon_results.items():
        all_results[f"normuon|{m}|{s}"] = cfg

    if all_results:
        summary_path = os.path.join(args.out_root, "lr_summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
