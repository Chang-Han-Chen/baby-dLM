#!/usr/bin/env python3
"""
evaluate.py — Post-training evaluation for scaling experiments (§6.6).

Loads a trained checkpoint and computes:
  - Primary: validation BPB (bits per byte) via prepare.py's evaluate_bpb
  - Secondary: generation quality samples for qualitative inspection

Outputs a per-run results row with all fields required by PLAN §6.6:
  optimizer family, curriculum id, model size, FLOP budget,
  selected LR setting, normuon multipliers and realized per-group LRs,
  final train / val metrics, and BPB.

Can evaluate a single checkpoint or scan a curriculum run directory
(from run_curriculum.py) to evaluate the final stage's checkpoint.

Usage:
    # Evaluate a single checkpoint:
    python evaluate.py --checkpoint runs/isoflop/baseline_ar_50M_1e+18_adamw/stage_0_ar/ckpt.pt

    # Evaluate a curriculum run (picks final stage checkpoint automatically):
    python evaluate.py --run-dir runs/curriculum/c1_geometric_50M_1e+18_adamw

    # Batch-evaluate all runs under an isoflop sweep:
    python evaluate.py --sweep-dir runs/isoflop

    # Skip BPB (faster, for debugging):
    python evaluate.py --checkpoint ckpt.pt --skip-bpb

    # Generate samples:
    python evaluate.py --checkpoint ckpt.pt --num-samples 5
"""

import argparse
import importlib
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import experiment_config as ec

# ---------------------------------------------------------------
# Result data structure
# ---------------------------------------------------------------

@dataclass
class EvalResult:
    """One row of the per-run results table (§6.6)."""
    # Identification
    checkpoint_path: str
    model_family: str
    size: str
    curriculum: str = ""
    flop_budget: float = 0.0

    # Optimizer / LR
    optimizer_family: str = "adamw"
    lr_config: dict = field(default_factory=dict)
    normuon_realized_lrs: Optional[dict] = None  # populated for normuon

    # Metrics
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    bpb: Optional[float] = None

    # Samples
    samples: List[str] = field(default_factory=list)

    # Metadata from checkpoint
    checkpoint_iter: Optional[int] = None

    def summary_dict(self) -> dict:
        d = {
            "checkpoint": self.checkpoint_path,
            "model_family": self.model_family,
            "size": self.size,
            "curriculum": self.curriculum,
            "flop_budget": self.flop_budget,
            "optimizer_family": self.optimizer_family,
            "lr_config": self.lr_config,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "bpb": self.bpb,
            "checkpoint_iter": self.checkpoint_iter,
            "num_samples": len(self.samples),
        }
        if self.normuon_realized_lrs is not None:
            d["normuon_realized_lrs"] = self.normuon_realized_lrs
        return d


# ---------------------------------------------------------------
# Checkpoint introspection
# ---------------------------------------------------------------

def load_checkpoint_metadata(ckpt_path: str) -> dict:
    """
    Load checkpoint metadata without loading model weights.

    Returns a dict with keys: args, iter, vocab_size, mask_token_id,
    optimizer_family, normuon_lrs (if applicable).
    """
    # Import torch lazily so pure-metadata tests don't need GPU
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return {
        "args": ckpt.get("args", {}),
        "iter": ckpt.get("iter"),
        "vocab_size": ckpt.get("vocab_size"),
        "mask_token_id": ckpt.get("mask_token_id"),
        "optimizer_family": ckpt.get("optimizer_family", "adamw"),
        "normuon_lrs": ckpt.get("normuon_lrs"),
    }


def infer_size_from_args(args: dict) -> Optional[str]:
    """Try to match checkpoint args to a known MODEL_SIZES entry."""
    n_embd = args.get("n_embd")
    n_layer = args.get("n_layer")
    if n_embd is None or n_layer is None:
        return None
    for label, (embd, layer, _, _) in ec.MODEL_SIZES.items():
        if embd == n_embd and layer == n_layer:
            return label
    return None


def infer_lr_config(args: dict, meta: dict) -> dict:
    """
    Reconstruct the LR config from checkpoint metadata.

    For adamw: {"lr": float}.
    For normuon: {"adam_mult": float, "matrix_mult": float}.
    """
    opt_family = meta.get("optimizer_family", args.get("optimizer", "adamw"))
    if opt_family == "normuon":
        return {
            "adam_mult": args.get("adam_mult", 1.0),
            "matrix_mult": args.get("matrix_mult", 1.0),
        }
    else:
        return {"lr": args.get("learning_rate", 0.0)}


# ---------------------------------------------------------------
# Model loading and BPB wrapper
# ---------------------------------------------------------------

def load_model_for_eval(ckpt_path: str, device: str = "cuda"):
    """
    Load a trained model from checkpoint, ready for evaluation.

    Returns (model, meta_dict, model_family_str).
    """
    import torch

    meta = load_checkpoint_metadata(ckpt_path)
    args = meta["args"]
    model_family = args.get("model", "ar")
    vocab_size = meta["vocab_size"]

    n_embd = args["n_embd"]
    n_head = args["n_head"]
    n_layer = args["n_layer"]
    head_dim = n_embd // n_head
    block_size = args.get("block_size", 2048)
    dropout = args.get("dropout", 0.1)
    block_len = args.get("block_len")

    module = importlib.import_module(ec.MODEL_MODULE_MAP[model_family])
    Model = module.Model

    model = Model(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        head_dim=head_dim,
        block_size=block_size,
        dropout=dropout,
        block_len=block_len,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return model, meta, model_family


class BPBModelWrapper:
    """
    Wraps a trained model to match the interface expected by
    prepare.evaluate_bpb: model(x, y, reduction='none') -> (B*T,) losses.

    evaluate_bpb calls:
        loss_flat = model(x, y, reduction='none').view(-1)
    where x and y are (B, T) token tensors from the val dataloader
    (next-token prediction format: x = tokens[:, :-1], y = tokens[:, 1:]).

    For AR models: compute logits from x, take CE against y.
    For diffusion/block models (MDLM, BD3-LM): same — the BPB metric
    measures next-token prediction quality regardless of training objective.
    """

    def __init__(self, model, model_family: str, vocab_size: int):
        self.model = model
        self.model_family = model_family
        self.vocab_size = vocab_size

    def __call__(self, x, y, reduction='none'):
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            if self.model_family == "ar":
                # AR forward: model(idx, targets) -> (logits, loss)
                # We need per-token losses, so compute manually.
                logits, _ = self.model(x)
                # logits: (B, T, V); predict position i -> target at i+1
                # x is already inputs (tokens[:, :-1]) and y is targets (tokens[:, 1:])
                # from prepare.make_dataloader, so logits align with y directly.
                per_token = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    y.reshape(-1),
                    reduction="none",
                )
            else:
                # Diffusion models (MDLM, BD3-LM): use backbone's forward
                # in next-token prediction mode.  The backbone accepts
                # (idx, targets, mask) and computes per-token CE internally.
                # For BPB we need per-token losses with all positions supervised.
                logits = self.model._forward_core(x, dual_stream=False)
                per_token = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    y.reshape(-1),
                    reduction="none",
                )

        if reduction == 'none':
            return per_token
        elif reduction == 'mean':
            return per_token.mean()
        else:
            return per_token.sum()

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()


# ---------------------------------------------------------------
# Generation wrapper
# ---------------------------------------------------------------

def generate_samples(
    model,
    model_family: str,
    meta: dict,
    num_samples: int,
    prompt_len: int = 16,
) -> List[str]:
    """Generate text samples from a loaded model."""
    import torch

    args = meta["args"]
    vocab_size = meta["vocab_size"]
    mask_token_id = meta["mask_token_id"]
    block_size = args.get("block_size", 2048)
    block_len = args.get("block_len")
    T = args.get("T", 100)

    module = importlib.import_module(ec.MODEL_MODULE_MAP[model_family])
    generate_from = module.generate_from

    # Set up tokenizer for decoding
    data_source = args.get("data", "climbmix")
    if data_source == "climbmix":
        from prepare import Tokenizer
        tokenizer = Tokenizer.from_directory()
        decode = tokenizer.decode
    else:
        # Legacy tiny data: try to load stoi/itos from checkpoint
        import torch as _torch
        ckpt = _torch.load(args.get("checkpoint_path", ""), map_location="cpu")
        itos = ckpt.get("itos", {})
        decode = lambda ids: "".join(itos.get(i, "?") for i in ids)

    # Get a prompt from validation data
    if data_source == "climbmix":
        from prepare import make_dataloader, Tokenizer as _Tok
        tok = _Tok.from_directory()
        loader = make_dataloader(tok, 1, block_size, "val")
        x0, _, _ = next(loader)
    else:
        # Fallback: random tokens
        import torch as _torch
        x0 = _torch.randint(0, vocab_size, (1, block_size), device="cuda")

    device = next(model.parameters()).device

    # Linear noise schedule for survival prob
    def survival_prob_scalar(t_frac):
        return 1.0 - t_frac

    samples = []
    for i in range(num_samples):
        prompt_mask = torch.zeros(1, block_size, dtype=torch.bool, device=device)
        prompt_mask[:, :prompt_len] = True
        x = x0.clone()

        kwargs = dict(
            T=T,
            block_size=block_size,
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            survival_prob_scalar=survival_prob_scalar,
            decode=decode,
        )
        if block_len is not None:
            kwargs["block_len"] = block_len

        sample_text = generate_from(model, x, prompt_mask, **kwargs)
        samples.append(sample_text)

    return samples


# ---------------------------------------------------------------
# Single checkpoint evaluation
# ---------------------------------------------------------------

def evaluate_checkpoint(
    ckpt_path: str,
    *,
    size: Optional[str] = None,
    curriculum: str = "",
    flop_budget: float = 0.0,
    skip_bpb: bool = False,
    num_samples: int = 0,
    batch_size: int = 32,
    device: str = "cuda",
) -> EvalResult:
    """
    Run full evaluation on a single checkpoint.

    Returns an EvalResult with BPB, samples, and all metadata.
    """
    import torch

    print(f"\nEvaluating: {ckpt_path}")

    # Load model
    model, meta, model_family = load_model_for_eval(ckpt_path, device=device)
    ckpt_args = meta["args"]

    # Infer size if not provided
    if size is None:
        size = infer_size_from_args(ckpt_args)
        if size is None:
            size = "unknown"
    print(f"  model={model_family}  size={size}  iter={meta['iter']}")

    # LR config
    lr_config = infer_lr_config(ckpt_args, meta)
    optimizer_family = meta.get("optimizer_family", ckpt_args.get("optimizer", "adamw"))

    # Normuon realized LRs (if applicable)
    normuon_realized = meta.get("normuon_lrs")

    # BPB evaluation
    bpb = None
    if not skip_bpb:
        data_source = ckpt_args.get("data", "climbmix")
        if data_source == "climbmix":
            from prepare import Tokenizer, evaluate_bpb
            tokenizer = Tokenizer.from_directory()
            wrapper = BPBModelWrapper(model, model_family, meta["vocab_size"])
            wrapper.eval()
            bpb = evaluate_bpb(wrapper, tokenizer, batch_size)
            print(f"  BPB = {bpb:.4f}")
        else:
            print(f"  BPB skipped (data_source={data_source}, not climbmix)")

    # Load final train/val loss from the loss log if available
    final_train_loss = None
    final_val_loss = None
    loss_log_path = ckpt_args.get("loss_log_path")
    if loss_log_path and os.path.exists(loss_log_path):
        with open(loss_log_path, "rb") as f:
            loss_data = pickle.load(f)
        if loss_data.get("train"):
            final_train_loss = loss_data["train"][-1][1]
        if loss_data.get("val"):
            final_val_loss = loss_data["val"][-1][1]

    # Generation samples
    samples = []
    if num_samples > 0:
        print(f"  Generating {num_samples} samples...")
        samples = generate_samples(
            model, model_family, meta,
            num_samples=num_samples,
            prompt_len=16,
        )
        for i, s in enumerate(samples):
            print(f"\n  --- Sample {i+1} ---")
            print(f"  {s[:200]}{'...' if len(s) > 200 else ''}")

    result = EvalResult(
        checkpoint_path=ckpt_path,
        model_family=model_family,
        size=size,
        curriculum=curriculum,
        flop_budget=flop_budget,
        optimizer_family=optimizer_family,
        lr_config=lr_config,
        normuon_realized_lrs=normuon_realized,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        bpb=bpb,
        samples=samples,
        checkpoint_iter=meta["iter"],
    )

    return result


# ---------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------

def find_final_checkpoint(run_dir: str) -> Optional[str]:
    """
    Find the final checkpoint in a curriculum run directory.

    Looks for the last stage_* subdirectory containing ckpt.pt.
    """
    if not os.path.isdir(run_dir):
        return None

    # Direct checkpoint
    direct = os.path.join(run_dir, "ckpt.pt")
    if os.path.exists(direct):
        return direct

    # Curriculum structure: stage_N_*/ckpt.pt
    stage_dirs = sorted(
        [d for d in os.listdir(run_dir) if d.startswith("stage_")],
        key=lambda d: int(d.split("_")[1]) if d.split("_")[1].isdigit() else -1,
    )
    for stage_dir in reversed(stage_dirs):
        ckpt = os.path.join(run_dir, stage_dir, "ckpt.pt")
        if os.path.exists(ckpt):
            return ckpt

    return None


def load_curriculum_metadata(run_dir: str) -> dict:
    """Load curriculum_summary.json if present."""
    json_path = os.path.join(run_dir, "curriculum_summary.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    return {}


def evaluate_run_dir(
    run_dir: str,
    *,
    skip_bpb: bool = False,
    num_samples: int = 0,
    batch_size: int = 32,
    device: str = "cuda",
) -> Optional[EvalResult]:
    """Evaluate the final checkpoint in a curriculum run directory."""
    ckpt_path = find_final_checkpoint(run_dir)
    if ckpt_path is None:
        print(f"  No checkpoint found in {run_dir}")
        return None

    # Extract metadata from curriculum summary if available
    cur_meta = load_curriculum_metadata(run_dir)
    curriculum = cur_meta.get("curriculum", "")
    flop_budget = cur_meta.get("budget", 0.0)
    size = cur_meta.get("size")

    return evaluate_checkpoint(
        ckpt_path,
        size=size,
        curriculum=curriculum,
        flop_budget=flop_budget,
        skip_bpb=skip_bpb,
        num_samples=num_samples,
        batch_size=batch_size,
        device=device,
    )


def evaluate_sweep_dir(
    sweep_dir: str,
    *,
    skip_bpb: bool = False,
    num_samples: int = 0,
    batch_size: int = 32,
    device: str = "cuda",
) -> List[EvalResult]:
    """Evaluate all run directories under an isoflop sweep directory."""
    results = []

    if not os.path.isdir(sweep_dir):
        print(f"Sweep directory not found: {sweep_dir}")
        return results

    subdirs = sorted(d for d in os.listdir(sweep_dir)
                     if os.path.isdir(os.path.join(sweep_dir, d)))

    print(f"\nScanning {sweep_dir}: found {len(subdirs)} subdirectories")

    for subdir in subdirs:
        run_path = os.path.join(sweep_dir, subdir)

        # Skip if no checkpoint exists
        ckpt = find_final_checkpoint(run_path)
        if ckpt is None:
            continue

        result = evaluate_run_dir(
            run_path,
            skip_bpb=skip_bpb,
            num_samples=num_samples,
            batch_size=batch_size,
            device=device,
        )
        if result is not None:
            results.append(result)

    return results


# ---------------------------------------------------------------
# Results table formatting and persistence
# ---------------------------------------------------------------

def format_results_table(results: List[EvalResult]) -> str:
    """Format results as a human-readable table."""
    lines = []
    lines.append(
        f"{'model':>6s}  {'size':>5s}  {'curriculum':<25s}  {'optimizer':>8s}  "
        f"{'budget':>10s}  {'iter':>6s}  {'train':>8s}  {'val':>8s}  "
        f"{'BPB':>8s}  lr_config"
    )
    lines.append("-" * 130)

    for r in sorted(results, key=lambda x: (
        x.model_family, x.size, x.curriculum, x.optimizer_family
    )):
        train = f"{r.final_train_loss:.4f}" if r.final_train_loss is not None else "—"
        val = f"{r.final_val_loss:.4f}" if r.final_val_loss is not None else "—"
        bpb = f"{r.bpb:.4f}" if r.bpb is not None else "—"
        budget = f"{r.flop_budget:.2e}" if r.flop_budget else "—"
        it = str(r.checkpoint_iter) if r.checkpoint_iter is not None else "—"

        # Compact LR config string
        if r.optimizer_family == "normuon":
            am = r.lr_config.get("adam_mult", "?")
            mm = r.lr_config.get("matrix_mult", "?")
            lr_str = f"am={am},mm={mm}"
        else:
            lr_str = f"lr={r.lr_config.get('lr', '?')}"

        lines.append(
            f"{r.model_family:>6s}  {r.size:>5s}  {r.curriculum:<25s}  "
            f"{r.optimizer_family:>8s}  {budget:>10s}  {it:>6s}  "
            f"{train:>8s}  {val:>8s}  {bpb:>8s}  {lr_str}"
        )

    return "\n".join(lines)


def save_results(results: List[EvalResult], out_path: str):
    """Save evaluation results as JSON and pickle."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # JSON summary
    json_path = out_path if out_path.endswith(".json") else out_path + ".json"
    summaries = [r.summary_dict() for r in results]
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Pickle (full, including samples)
    pkl_path = out_path.replace(".json", ".pkl") if out_path.endswith(".json") else out_path + ".pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Full results saved to {pkl_path}")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-training evaluation for scaling experiments (§6.6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate.py --checkpoint runs/.../ckpt.pt\n"
            "  python evaluate.py --run-dir runs/curriculum/c1_geometric_50M_1e+18_adamw\n"
            "  python evaluate.py --sweep-dir runs/isoflop\n"
            "  python evaluate.py --checkpoint ckpt.pt --skip-bpb --num-samples 5\n"
        ),
    )

    # Input modes (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint", type=str,
        help="Path to a single checkpoint file",
    )
    group.add_argument(
        "--run-dir", type=str,
        help="Path to a curriculum run directory (finds final checkpoint)",
    )
    group.add_argument(
        "--sweep-dir", type=str,
        help="Path to an isoflop sweep directory (evaluates all runs)",
    )

    # Optional overrides
    parser.add_argument("--size", type=str, default=None, choices=ec.ALL_SIZES,
                        help="Model size label (auto-detected from checkpoint if omitted)")
    parser.add_argument("--curriculum", type=str, default="",
                        help="Curriculum name for labeling (auto-detected if omitted)")
    parser.add_argument("--budget", type=float, default=0.0,
                        help="FLOP budget for labeling (auto-detected if omitted)")

    # Eval options
    parser.add_argument("--skip-bpb", action="store_true",
                        help="Skip BPB evaluation (faster)")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Number of generation samples (default: 0 = skip)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for BPB evaluation (default: 32)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")

    # Output
    parser.add_argument("--out", type=str, default=None,
                        help="Output path for results (default: auto)")

    args = parser.parse_args()

    results = []

    if args.checkpoint:
        result = evaluate_checkpoint(
            args.checkpoint,
            size=args.size,
            curriculum=args.curriculum,
            flop_budget=args.budget,
            skip_bpb=args.skip_bpb,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=args.device,
        )
        results.append(result)

    elif args.run_dir:
        result = evaluate_run_dir(
            args.run_dir,
            skip_bpb=args.skip_bpb,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=args.device,
        )
        if result is not None:
            results.append(result)

    elif args.sweep_dir:
        results = evaluate_sweep_dir(
            args.sweep_dir,
            skip_bpb=args.skip_bpb,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=args.device,
        )

    if not results:
        print("\nNo results to report.")
        return

    # Print table
    print(f"\n{'='*70}")
    print("Evaluation Results")
    print(f"{'='*70}")
    print(format_results_table(results))

    # Save
    if args.out is None:
        if args.sweep_dir:
            args.out = os.path.join(args.sweep_dir, "eval_results.json")
        elif args.run_dir:
            args.out = os.path.join(args.run_dir, "eval_results.json")
        else:
            args.out = os.path.join(
                os.path.dirname(args.checkpoint), "eval_results.json"
            )
    save_results(results, args.out)


if __name__ == "__main__":
    main()
