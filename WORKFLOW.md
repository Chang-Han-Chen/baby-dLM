# Experiment Workflow

Running scaling experiments for block-length curriculum training on discrete diffusion LMs.
Three model families (AR, MDLM, BD3-LM), two optimizers (AdamW, NorMuon), ClimbMix-400B data.

---

## Prerequisites

**Hardware:** 1-2 VMs with A100 (80 GB) GPUs. A single A100 can run everything sequentially; two VMs let you parallelize (e.g., AdamW sweeps on VM-1, NorMuon on VM-2).

**Software:**
```bash
pip install -r requirements.txt
```

**Data:** Downloads parquet shards + trains a BPE tokenizer. Everything is stored in `data_cache/` inside the repo.
```bash
python prepare.py --num-shards 10   # ~10 training shards + 1 val shard
```

**Validation:** Run smoke tests before committing to long GPU runs.
```bash
pytest tests/test_cuda_smoke.py -v --tb=short
```
Smoke tests train directly on ClimbMix and skip automatically if `prepare.py` hasn't been run yet.

---

## Step 1: Learning-Rate Calibration

**Goal:** Find the best stable LR for each (optimizer, model, size) combination.

Each sweep currently runs 200-step warmup-stable training (5% linear warmup, then constant LR). The sweep script runs candidates in parallel, auto-detecting how many fit in VRAM, and early-aborts diverged runs.

### AdamW Sweeps — COMPLETED (50M only)

**Decision:** Use LR=1e-3 for all three model families at 50M scale, based on the
AR sweep result (loss=4.507 at 1e-3, cleanly optimal across the 5-point grid).
98M and 170M sweeps are deferred — all Phase 1 experiments run at 50M only.

The shared LR is saved in `calibrated_lrs.json` for AR, MDLM, and BD3-LM at 50M.

```bash
# Only the AR sweep was actually run; MDLM and BD3-LM inherit the same LR.
python run_lr_sweep.py --optimizer adamw --model ar    --size 50M   # DONE → 1e-3
# python run_lr_sweep.py --optimizer adamw --model mdlm  --size 50M  # skipped, using 1e-3
# python run_lr_sweep.py --optimizer adamw --model bd3lm --size 50M  # skipped, using 1e-3
```

### NorMuon Sweeps — COMPLETED (50M only)

**Decision:** Use `adam_mult=0.3, matrix_mult=1.0` for all three model families at 50M,
based on the AR sweep (best loss=5.109 from the 3×3 grid). The pattern shows lower
`adam_mult` is consistently better, and `matrix_mult=1.0` is the sweet spot.

The shared config is saved in `calibrated_normuon.json` for AR, MDLM, and BD3-LM at 50M.

```bash
# Only the AR sweep was actually run; MDLM and BD3-LM inherit the same config.
python run_lr_sweep.py --optimizer normuon --model ar    --size 50M   # DONE → (0.3, 1.0)
# python run_lr_sweep.py --optimizer normuon --model mdlm  --size 50M  # skipped
# python run_lr_sweep.py --optimizer normuon --model bd3lm --size 50M  # skipped
```

### Useful flags

```bash
--jobs auto         # (default) auto-detect parallelism from VRAM
--jobs 2            # force 2 concurrent runs
--sweep-batch-size 64  # reduce micro-batch during sweep to fit more jobs
--no-early-abort    # disable NaN/divergence early abort
--dry-run           # print commands without running
```

### Output

Results land in `results/lr_sweep/<optimizer>/<model>/<size>/lr_<value>/`. Each subfolder contains `loss.pkl` (train/val loss curves) and `ckpt.pt`.

The best LR for each combo is automatically saved to `calibrated_lrs.json` (AdamW) or `calibrated_normuon.json` (NorMuon). These are loaded on import by `experiment_config.py` so subsequent scripts pick them up automatically.

### Selection rule

For both optimizers: discard unstable runs (loss blow-up, grad norm explosion, NaN), then pick the stable run with the best final train loss.

---

## Step 2: Analyze Sweep Results

Pull the results folder back and share with Claude. Claude reads the loss logs, confirms the selected LRs, and generates the exact commands for Step 3.

**What to transfer back:**
```
results/lr_sweep/          # all sweep output
calibrated_lrs.json        # auto-selected AdamW LRs
calibrated_normuon.json    # auto-selected NorMuon configs
```

---

## Step 3: IsoFLOP Scaling Runs

**Goal:** For each (compute budget, curriculum, optimizer), sweep model sizes and record validation loss. This gives one IsoFLOP curve per curriculum.

### Compute budgets

`C in {1e18, 2e18, 4e18, 1e19}` FLOPs.

Step counts are automatically computed from the budget, model family (AR/MDLM use 6N FLOP multiplier, BD3-LM uses 12N due to dual-stream), and effective batch size (256 tokens/step = 128 micro-batch x 2 grad_accum x 2048 seq_len = 524,288 tokens/step).

### Phase 1: Curricula 0-1 at 50M (both optimizers, 1000 steps)

```bash
# Curriculum 0: AR -> BD3-LM(16), sweep p_AR — AdamW
python run_curriculum.py --curriculum c0_plain_p20 --size 50M --steps 1000 --optimizer adamw
python run_curriculum.py --curriculum c0_plain_p30 --size 50M --steps 1000 --optimizer adamw
python run_curriculum.py --curriculum c0_plain_p50 --size 50M --steps 1000 --optimizer adamw
python run_curriculum.py --curriculum c0_plain_p80 --size 50M --steps 1000 --optimizer adamw

# Curriculum 1: Geometric doubling AR -> BD3(2) -> BD3(4) -> BD3(8) -> BD3(16) — AdamW
python run_curriculum.py --curriculum c1_geometric --size 50M --steps 1000 --optimizer adamw

# Same 5 runs with NorMuon
python run_curriculum.py --curriculum c0_plain_p20 --size 50M --steps 1000 --optimizer normuon
python run_curriculum.py --curriculum c0_plain_p30 --size 50M --steps 1000 --optimizer normuon
python run_curriculum.py --curriculum c0_plain_p50 --size 50M --steps 1000 --optimizer normuon
python run_curriculum.py --curriculum c0_plain_p80 --size 50M --steps 1000 --optimizer normuon
python run_curriculum.py --curriculum c1_geometric --size 50M --steps 1000 --optimizer normuon
```

The `--steps 1000` flag splits 1000 total steps across stages by `flop_frac`.
For example, C1 (5 stages × 20% each) gives 200 steps per stage.
C0 with p_AR=0.2 gives 200 AR steps + 800 BD3 steps.

### Model sizes

50M only (768d, 7L) for Phase 1. `run_isoflop.py` now defaults to `["50M"]`.
To re-enable multi-size sweeps later, pass `--sizes 50M 98M 170M`.

### Output

Results land in `results/isoflop/<curriculum>/<optimizer>/<budget>/<size>/`. The script collects `(model_size -> val_loss)` curves and outputs summary tables + pickles.

---

## Step 4: Curriculum Experiments

**Goal:** Compare block-length curriculum schedules against baselines.

### Curriculum definitions

| ID | Schedule | FLOP Split |
|----|----------|------------|
| Baseline AR | AR only | 100% AR |
| Baseline MDLM | MDLM only | 100% MDLM |
| Baseline BD3-LM | BD3(16) only | 100% BD3 |
| C0 (plain) | AR -> BD3(16) | sweep p_AR in {20%, 30%, 50%, 80%} |
| C1 (geometric) | AR -> BD3(2) -> BD3(4) -> BD3(8) -> BD3(16) | 20% each |
| C2 (aggressive) | AR -> BD3(8) -> BD3(64) -> BD3(512) -> BD3(16) | 20% each |
| C3 (AR-heavy) | AR -> BD3(4) -> BD3(16) | sweep p_AR in {20%, 30%, 50%} |

Stage transitions warm-start from the previous checkpoint (weights only; optimizer and LR schedule reset).

### Phase 2: Curricula 2-3 (deferred)

Run only after analyzing Phase 1 results to decide whether NorMuon materially changes the ranking.

```bash
python run_curriculum.py --curriculum c2_aggressive_jump --size 50M --budget 1e18
python run_curriculum.py --curriculum c3_ar_heavy_p20    --size 50M --budget 1e18
```

---

## Two-VM Strategy

Split work across two A100 VMs to halve wall-clock time:

| VM-1 | VM-2 |
|------|------|
| AdamW baselines + C0 + C1 (50M) | NorMuon LR sweeps (3 combos, 50M) |
| — | NorMuon baselines + C0 + C1 (50M) |

AdamW LR calibration is complete (1e-3 for all families at 50M).
NorMuon still needs the 3 sweeps (ar, mdlm, bd3lm × 50M).

Both VMs need `prepare.py` run independently (data is stored locally in `data_cache/`).

After NorMuon sweeps, copy `calibrated_normuon.json` to the AdamW VM so both have the full set of calibrated configs.

---

## Key Config Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Vocab size | 8,192 | BPE tokenizer via `prepare.py` |
| Sequence length | 2,048 | `MAX_SEQ_LEN` in `prepare.py` |
| Micro-batch size | 128 | `CLIMBMIX_BATCH_SIZE` |
| Gradient accumulation | 2 | `CLIMBMIX_GRAD_ACCUM` |
| Effective batch size | 256 | 128 x 2 |
| Tokens per step | 524,288 | 256 x 2,048 |
| LR sweep steps | 200 | `LR_SWEEP_STEPS` |
| LR warmup | 5% | `LR_SWEEP_WARMUP_FRAC` |
| AdamW LR grid (AR) | [1e-4, 3e-4, 1e-3, 3e-3, 1e-2] | `LR_SWEEP_GRIDS` |
| AdamW LR grid (MDLM/BD3) | [3e-4, 1e-3, 3e-3, 1e-2, 3e-2] | `LR_SWEEP_GRIDS` |
| NorMuon grid | adam_mult x matrix_mult, both [0.3, 1.0, 3.0] | `NORMUON_*_GRID` |
| IsoFLOP budgets | {1e18, 2e18, 4e18, 1e19} | `CLIMBMIX_ISOFLOP_BUDGETS` |
| Head dim | 64 (all ClimbMix sizes) | n_embd / n_head |

---

## File Layout

```
baby-dLM/
  PLAN.md                    # Research plan
  WORKFLOW.md                # This file
  train.py                   # Main training loop
  prepare.py                 # Data download + tokenizer training
  experiment_config.py       # Centralized config (sizes, LRs, curricula)
  run_lr_sweep.py            # Parallelized LR sweep
  run_curriculum.py          # Multi-stage curriculum runner
  run_isoflop.py             # IsoFLOP scaling sweep
  evaluate.py                # Evaluation (BPB, generation)
  normuon.py                 # NorMuon optimizer implementation
  backbone.py                # Shared DiffusionBackbone (RoPE, QK-norm, ReGLU)
  model_AR.py                # Autoregressive model
  model_MDLM.py              # Masked diffusion model
  model_bd3lm.py             # Block discrete denoising diffusion model
  block_utils.py             # Block attention utilities
  calibrated_lrs.json        # Auto-generated: AdamW LRs after sweep
  calibrated_normuon.json    # Auto-generated: NorMuon configs after sweep
  data_cache/                # Local data storage (gitignored)
    shards/                  #   Parquet files from ClimbMix
    tokenizer/               #   tokenizer.pkl, token_bytes.pt
  results/                   # Experiment outputs
    lr_sweep/                #   LR calibration runs
    isoflop/                 #   Scaling law runs
    curriculum/              #   Curriculum experiment runs
  tests/
    test_cuda_smoke.py       # CUDA end-to-end smoke tests on ClimbMix (34 tests)
    test_prepare_runtime.py  # Tokenizer + dataloader unit tests (no CUDA/data needed)
```
