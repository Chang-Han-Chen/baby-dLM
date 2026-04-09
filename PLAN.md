# Scaling Block Diffusion on ClimbMix

## Overview

We study how block-length curriculum training affects scaling laws for discrete
diffusion LMs.  All experiments use ClimbMix-400B (BPE, vocab 8192, seq len 2048)
and three model families: AR, MDLM, BD3-LM.

In parallel, we compare the current training setup against one new optimizer
option:
- `adamw`: the existing code path, unchanged. One global learning rate per run.
- `normuon`: a Karpathy-inspired grouped optimizer path with AdamW on
  non-matrix parameter groups and NorMuon on matrix parameter groups.
  This path uses Karpathy's reference learning rates and width scaling formulas
  as the center of its learning-rate sweep.

The initial optimizer comparison covers the no-curriculum baselines and
curricula 0 and 1 only.  Later curricula can be revisited after we understand
whether the grouped optimizer recipe materially changes the ranking.

---

## 1  Model sizes

We scale up from the TinyShakespeare regime to three sizes:

| Label | n_embd | n_layer | n_head | Approx non-embedding params |
|-------|--------|---------|--------|-----------------------------|
| 50M   | 512    | 16      | 8      | 50.3M                       |
| 98M   | 640    | 20      | 10     | 98.3M                       |
| 170M  | 768    | 24      | 12     | 169.9M                      |

We count only non-embedding transformer parameters:
`N = 12 * n_layer * n_embd^2`.
This includes attention and MLP weights and excludes the token embedding and
LM head.

Head dim is always 64 (= n_embd / n_head).

These sizes are still small enough to support short LR sweeps and broad
IsoFLOP experiments on a single H100.

---

## 2  Data

ClimbMix via `prepare.py`.  Download ≥10 shards for training + 1 pinned
validation shard (shard_06542).  BPE tokenizer with vocab_size = 8192.

Token count per run is determined by the chosen FLOP budget and model family,
not by a fixed 1B-token target.  Once tokens_processed is known, step count is
`tokens_processed / (batch_size * seq_len)`.

---

## 3  Curricula

A "curriculum" is a sequence of stages, each defined by its model family and,
for BD3-LM stages, its block_len.
Within each stage the model trains for a fixed fraction of the total FLOP budget.
Stage transitions warm-start from the previous stage's checkpoint (weights only;
optimizer and LR schedule reset).

### Curriculum 0 — plain
```text
stage: AR -> BD3(block_len=16)
FLOP split: p_AR on AR, 1 - p_AR on diffusion
```
Sweep p_AR ∈ {0.2, 0.3, 0.5, 0.8}.

### Curriculum 1 — Geometric doubling (SDAR / NBDiff style)
```text
stage: AR -> BD3(2) -> BD3(4) -> BD3(8) -> BD3(16)
FLOP split: 20% each (5 equal stages)
```
Starts with true autoregressive training and doubles the BD3 block length at
each stage.

### Curriculum 2 — Aggressive early jump (LLaDA 2 style)
```text
stage: AR -> BD3(8) -> BD3(64) -> BD3(512) -> BD3(16)
FLOP split: 20% each (5 equal stages)
```
Jumps quickly to large blocks, then narrows back to block_len=16 for the final
stage.

### Curriculum 3 — AR-heavy warmup, then two diffusion stages
```text
stage: AR -> BD3(4) -> BD3(16)
FLOP split: p_AR on stage 1, 0.8 - p_AR on stage 2, 0.2 on stage 3
```
Sweep p_AR ∈ {0.2, 0.3, 0.5}.

### Curriculum 4 — To be designed
Informed by results of curricula 1–3.  Placeholder for now.

### Baselines (no curriculum)
- **Pure AR**: AR model for the full budget.
- **Pure MDLM**: standard MDLM (no blocks) for the full budget.
- **Pure BD3-LM**: block_len=16 (or other fixed value) for the full budget.

---

## 4  Learning-rate calibration

This is the first experiment.

For each `(optimizer_family, model_family, size)` pair:
- Train for 2000 steps with a warmup-stable schedule (5% linear warmup, then
  constant LR).
- Keep the selection rule the same for both optimizers:
  - track train loss and grad norm throughout the 2000-step run
  - discard unstable runs (loss blow-up / grad norm blow-up / NaNs)
  - choose the stable run with the best train loss
- `adamw` calibration:
  - use the existing single-LR sweep exactly as before
  - one candidate `learning_rate` per run
- `normuon` calibration:
  - use an adapted Karpathy-style parameter grouping:
    - embeddings
    - `lm_head`
    - matrix parameters (all 2D weights except embeddings / `lm_head`)
    - scalar/vector residual parameters (any remaining params with ndim < 2;
      this group may be empty for the current models)
  - use Karpathy-style reference LRs and width scaling as the center of the
    sweep:
    - `embedding_lr = 0.6 * sqrt(768 / d_model)`
    - `unembedding_lr = 0.004 * sqrt(768 / d_model)`
    - `scalar_lr = 0.5` for the scalar/vector group
    - `matrix_lr = 0.04` with no additional width scaling
  - sweep a small 2D multiplier grid around those references:
    - `adam_mult` rescales `embedding_lr`, `unembedding_lr`, and `scalar_lr`
    - `matrix_mult` rescales `matrix_lr`
  - log optimizer family, `adam_mult`, `matrix_mult`, and the realized
    per-group learning rates in addition to loss and grad norm

For BD3-LM, use the fixed-block baseline setting (`block_len=16`) as the
representative calibration run.  After calibration, reuse the chosen LR config
for that `(optimizer_family, model_family, size)` pair across the baselines and
all curricula that use that optimizer family.

The key point is that `adamw` remains untouched.  The only new optimizer path
is `normuon`.

---

## 5  Scaling laws (IsoFLOP)

Compute budgets: `C ∈ {1e18, 2e18, 4e18, 1e19}` FLOPs.

For each (C, curriculum) pair, sweep model sizes and record the best validation
loss.  This gives one IsoFLOP curve per curriculum.

**LR schedule**: warmup-stable (linear warmup for 5% then constant LR, no
cosine decay), using the preselected LR config for the corresponding
`(optimizer_family, model_family, size)` pair.
This lets us read off the loss at any step without retraining, so each
(model_size, curriculum) combination needs only **one run**.

Phase 1 scope:
- Run both optimizer families (`adamw`, `normuon`) on the no-curriculum
  baselines, curriculum 0, and curriculum 1.
- Defer curricula 2–4 until the optimizer comparison is understood.
- Log optimizer family and calibrated LR settings in every result table and
  serialized artifact.

### 5.1  First-pass execution order
1. Implement `--optimizer normuon` while leaving the current `adamw` path
   untouched.
2. Run LR calibration for all `(optimizer_family, model_family, size)` pairs.
3. Freeze the selected LR configs and write them to config/artifact files.
4. Run the no-curriculum baselines for both optimizers.
5. Run curriculum 0 for both optimizers.
6. Run curriculum 1 for both optimizers.
7. Summarize results and decide whether curricula 2–4 are worth running under
   both optimizers.

FLOP accounting (unchanged from `experiment_config.py`):
- Count only non-embedding parameters: `N = 12 * n_layer * n_embd^2`
- AR / MDLM: C = 6 · N · tokens_processed
- BD3-LM (dual-stream training): C = 12 · N · tokens_processed

Equivalent token budgets are therefore determined by the compute budget, model
family, and model size.

---

## 6  Implementation plan

### 6.1  Update train.py for ClimbMix
- Replace the character-level `data.txt` pipeline with `prepare.py`'s
  `make_dataloader` / `Tokenizer`.
- Set `block_size = 2048` (MAX_SEQ_LEN), `vocab_size = 8192`.
- Mask token = a reserved special token (not a regular vocab entry).
- Add `--warmup_stable` flag: when set, hold LR constant after warmup
  (no cosine decay).
- Log grad norm alongside train loss so LR sweeps can identify the best stable
  LR.
- Add optimizer-family support:
  - `adamw` (current path, unchanged)
  - `normuon` (new grouped optimizer path)
- For `normuon`, implement the Karpathy-style parameter grouping and
  width-scaled AdamW subgroup LRs.
- Log the realized per-group learning rates for `normuon`, and keep the current
  scalar-LR logging for `adamw`.
- Keep checkpointing / resume behavior streamlined:
  - `adamw` checkpoints save one optimizer state
  - `normuon` checkpoints save the grouped optimizer state without changing the
    warm-start semantics already used by curricula

### 6.2  Update experiment_config.py
- Add 50M, 98M, 170M to MODEL_SIZES.
- Count FLOPs with non-embedding parameters only.
- Add IsoFLOP budgets: {1e18, 2e18, 4e18, 1e19}.
- Add curriculum definitions as structured config (list of
  `(model_family, block_len, flop_fraction)` stages).
- Add optimizer families and a place to store the selected LR config for each
  `(optimizer_family, model_family, size)` triple.
- For `adamw`, store the selected scalar LR exactly as today.
- For `normuon`, store the selected `adam_mult`, `matrix_mult`, and optionally
  the realized per-group learning rates for logging / reproducibility.

### 6.3  Add LR sweep script
New script `run_lr_sweep.py`:
1. For each `(optimizer_family, model_family, size)`, run 2000-step sweeps.
2. For `adamw`, run the existing 1D LR sweep.
3. For `normuon`, run a small 2D grid over `(adam_mult, matrix_mult)` around
   the Karpathy reference setting.
4. In both cases, record train loss and grad norm curves and pick the best
   stable run using the same rule.
5. Write a summary table + pickle/JSON with the selected optimizer-specific LR
   config.

### 6.4  Add curriculum runner
New script `run_curriculum.py`:
1. Takes a curriculum spec + total FLOP budget + model size.
2. For each stage: compute steps from FLOP fraction and stage objective.
3. Launch `train.py` with the stage's `--model`, `--block_len` when needed,
   the calibrated optimizer config for that `(optimizer_family, model_family,
   size)` pair, and `--resume_from` pointing to the previous stage's checkpoint.
4. For `adamw`, pass the selected scalar LR.
5. For `normuon`, pass the selected multiplier config and let `train.py`
   realize the per-group learning rates from model width.
6. Log per-stage and final validation loss, along with optimizer family and LR
   settings.

### 6.5  Add IsoFLOP sweep script
New script `run_isoflop.py`:
1. For each (budget, curriculum, optimizer_family, model_size): run the
   curriculum.
2. In the first pass, restrict curricula to the baselines plus curriculum 0 and
   curriculum 1.
3. Collect `(model_size -> val_loss)` curves, keyed by optimizer family.
4. Output a table + pickle for plotting.

### 6.6  Evaluation
- Primary metric: validation BPB (bits per byte) via `prepare.py`'s
  `evaluate_bpb`.
- Secondary: generation quality samples for qualitative inspection.
- Keep a per-run results table with:
  - optimizer family
  - curriculum id
  - model size
  - FLOP budget
  - selected LR setting
  - for `normuon`, selected `adam_mult`, selected `matrix_mult`, and realized
    per-group learning rates
  - final train / val metrics

### 6.7  Plotting
Extend `reproduce.ipynb` (or new notebook) to plot:
- LR sweep train-loss and grad-norm diagnostics for both optimizers.
- For `normuon`, LR sweep heatmaps over `(adam_mult, matrix_mult)`.
- IsoFLOP curves (val loss vs model size, one curve per curriculum and
  optimizer family).
- Curriculum learning curves (val loss vs FLOPs, showing stage transitions).
- Optimal model size vs compute for each curriculum.

---

## 7  Open questions

- Should we also sweep block_len within the "no curriculum" BD3-LM baseline
  (e.g., block_len ∈ {4, 8, 16, 32})?
- Is a single BD3-LM LR per size good enough across block_len ∈
  {2, 4, 8, 16, 64, 512}, or should we add block_len-specific LR calibration
  only if instability appears?
- Does the Karpathy width-scaling rule transfer cleanly from AR-heavy training
  to MDLM / BD3-LM, or will diffusion models want a different `adam_mult`
  center?
- If `normuon` helps, is the gain large enough to justify carrying the more
  complicated grouped-LR sweep into the later curricula?
- Curriculum 4 design: consider reverse schedules (large→small block_len),
  or non-monotonic schedules informed by curricula 1–3.
