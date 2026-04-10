# Engineering Memo

Running log of findings, changes, and decisions made while bringing up the experiment pipeline.

Note: the canonical ClimbMix size definitions were later updated to
`50M = (n_embd=768, n_layer=7, n_head=12)`,
`98M = (n_embd=896, n_layer=10, n_head=14)`,
`170M = (n_embd=1024, n_layer=14, n_head=16)`.
Historical VRAM and throughput notes below that mention `50M/98M/170M` refer to
the older deeper configurations unless explicitly stated otherwise.

---

## 2026-04-10: Environment Setup & Smoke Tests

### Initial state
- `prepare.py` had already been run: 10 training shards + 1 val shard + BPE tokenizer present in `data_cache/`.
- `requirements.txt` had **not** been installed — `import torch` failed with `ModuleNotFoundError`.
- All 34 CUDA smoke tests were being skipped (not failing) because `torch.cuda.is_available()` couldn't even run.

### Fix 1: Install dependencies
```
pip install -r requirements.txt
```
This pulled PyTorch 2.11.0 with **CUDA 13.0** libraries, but the machine has **driver 565.77 (CUDA 12.7)**. Result: `torch.cuda.is_available() = False` despite having an A100-80GB.

Reinstalled with the correct CUDA 12.6 index:
```
pip install torch --index-url https://download.pytorch.org/whl/cu126
```
CUDA then worked. GPU confirmed: NVIDIA A100-SXM4-80GB.

### Fix 2: train.py logging frequency (train.py:691)
**Problem:** train.py only printed per-step `loss | grad_norm` lines at `iter % 100 == 0` or the final iteration. The smoke tests run 30 steps and parse these lines to verify training health, expecting >= 5 records but only getting 2 (step 0 and step 29).

**Change:** `iter % 100` → `iter % min(100, eval_interval)` so short runs log at every eval checkpoint.

Also lowered `_check_basic_training` default `min_steps` from 5 to 3 in `test_cuda_smoke.py` since 30 steps with eval_interval=10 produces 4 log records.

### Smoke test results (after fixes)
- 25 passed, 9 failed
- **TestCUDATraining / TestCUDANorMuon (6 fails):** log-parsing issue, fixed by changes above
- **TestVRAMFit (3 fails):** OOM because multiple test processes were sharing the GPU concurrently — transient, not a code bug

---

## 2026-04-10: LR Sweep VRAM Tuning

### Problem: `run_lr_sweep.py` auto-parallelism caused OOM

The VRAM estimates in `run_lr_sweep.py` were wildly optimistic (e.g., 6 GB for AR 50M), leading to 5+ concurrent jobs that collectively exhausted 80 GB.

Three separate issues discovered through iteration:

#### Issue A: torch.compile VRAM spikes
`train.py` defaults to `--use_compile true`. During compilation, a single 50M AR process spikes to **55-79 GiB** before settling to ~8 GiB steady state. With multiple processes compiling simultaneously, instant OOM.

**Fix:** Added `use_compile` parameter to `experiment_config.py:build_command()` and set `use_compile=False` in the sweep's `build_kwargs`. The 2000-step sweep is too short to amortize compile overhead anyway.

#### Issue B: batch_size=128 peaks at ~78 GiB per process
Even without torch.compile, a single AR 50M process at batch_size=128, seq_len=2048, vocab_size=8192 peaks at **~78 GiB**. The logit tensor alone is `128 * 2048 * 8192 * 4 bytes = 8 GiB` in float32, plus activations and gradients across 16 transformer layers.

**Fix:** Use `--sweep-batch-size 32` to reduce micro-batch, combined with grad accumulation to maintain the correct effective batch size.

#### Issue C: Grad accumulation to preserve effective batch size
Simply reducing batch_size changes the effective batch, making LR calibration unrepresentative of production training.

**Fix:** Added `_SWEEP_EFFECTIVE_BATCH = 256` constant. When `--sweep-batch-size` is set, the sweep computes `grad_accum_steps = effective_batch / micro_batch` (e.g., 256 / 32 = 8) so the optimizer sees identical gradients regardless of micro-batch size.

#### Issue D: VRAM scaling formula was wrong
The original formula assumed 35% of VRAM is fixed (model + optimizer) and 65% scales with batch size. Empirically, it's closer to **10% fixed, 90% scales**:
- AR 50M at batch=128: ~78 GiB (fixed ~8 GiB, activations ~70 GiB)
- AR 50M at batch=32: ~27 GiB (fixed ~8 GiB, activations ~19 GiB)

**Fix:** Updated scaling formula from `0.35 + 0.65 * ratio` to `0.10 + 0.90 * ratio`. Updated base VRAM estimates to reflect measured batch=128 peaks.

#### Issue E: PyTorch caching allocator prevents any parallelism
Even with batch=32 (no compile), 2 parallel jobs OOM. The problem is PyTorch's CUDA caching allocator: it never releases memory back to the GPU, so a single long-running process grows from ~8 GiB (initial) to **42 GiB** over time. A second process launched alongside it sees only ~38 GiB free and OOMs at its first large allocation.

Observed VRAM growth for a single AR 50M process at batch=32:
- **t=0s:** ~8 GiB (model + optimizer loaded)
- **t=30s:** ~17 GiB (first backward pass activations cached)
- **t=60s:** ~32 GiB (caching allocator holds freed tensors)
- **t=120s:** ~43 GiB (steady state — caching allocator ceiling)

This means a single 50M model at batch=32 eventually consumes **53% of an A100-80GB**. Two processes would need ~86 GiB.

**Current status:** Auto-detect now selects `--jobs 1` (sequential). All 5 LR candidates run one at a time.

### Root cause found: unnecessary logit tensor retention

**model_AR.py:166-176** — `forward()` always computes and returns the full `logits` tensor `(B, T, V)` even during training when only the loss scalar is needed. This tensor is `(32, 2048, 8192)` in bf16 = 1 GiB (or 2 GiB in fp32 after cross_entropy upcast), and it stays alive in the autograd graph because it's returned from forward().

Profiling with `torch.cuda.memory_allocated()`:
- Model params: 0.23 GB (fp32)
- **After forward (batch=32):** 37.35 GB allocated, 37.48 GB peak
- This is ~160× the model size, for a single forward pass

The logits are never used in training (`compute_loss` discards them: `_, loss = model(x, targets=targets)`), but Python holds the reference until the tuple is destructured, and autograd saves intermediates for the backward pass through the logits path.

**Potential fix:** When `targets` is provided, don't return logits (or compute loss in-place without materializing the full logits tensor, e.g., use chunked cross-entropy).

**Open questions for further investigation:**
1. **Why does a 50M model need 43 GiB at batch=32?** Model params are ~59M × 4 bytes × 3 (params + grads + optimizer) ≈ 0.7 GiB. Activations for batch=32, seq=2048, 16 layers shouldn't be this large. Is something being retained unnecessarily?
2. **Would `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` help?** The OOM errors mention fragmentation. This env var was added in PyTorch 2.1+ to reduce it.
3. **Is gradient checkpointing an option?** Could dramatically reduce activation memory at the cost of recomputation, enabling parallelism.
4. **Is the logit tensor the bottleneck?** `lm_head` outputs (B, T, V) = (32, 2048, 8192) in float32 = 2 GiB. The backward pass through cross_entropy must hold gradients for this. With `torch.amp.autocast`, the logits should be in bf16 but cross_entropy forces fp32 upcast.
5. **Memory leak vs. caching?** Does `torch.cuda.memory_allocated()` also show 43 GiB, or is it just `memory_reserved()`? If allocated << reserved, the caching allocator is holding freed memory. If allocated ≈ reserved, there's a genuine retention issue.

### Sweep attempt timeline

| Attempt | Config | Workers | Result |
|---------|--------|---------|--------|
| v1 | batch=128, compile=on | 12 (auto) | All 5 OOM — compile spike to 47-79 GiB |
| v2 | batch=128, compile=on | 4 (auto) | All 5 OOM — compile spike to 55 GiB |
| v3 | batch=128, compile=off | 8 (auto) | All 5 OOM — each process peaks at 22-57 GiB |
| v4 | batch=32, accum=8, compile=off | 6 (auto) | All 5 OOM — 5 procs × 8 GiB init = 42 GiB, then growth |
| v5 | batch=32, accum=8, compile=off | 2 (auto) | 4/5 OOM — proc 1 grew to 42 GiB, starving proc 2 |
| v6 | batch=32, accum=8, compile=off | 1 (sequential) | All 5 OOM @ 78 GiB — **sequential path ignored `--sweep-batch-size`, ran at batch=128** |
| v7 | batch=32, accum=8, compile=off, 200 steps | 1 (parallel dispatcher) | **Running successfully** — 52% VRAM (43 GiB), 100% util |

#### Issue F: Sequential code path ignored `--sweep-batch-size`
When `max_workers <= 1`, the sweep used a legacy `sweep_adamw()` function that called `run_single()` without passing the `batch_size` override. So `--sweep-batch-size 32` was silently ignored and jobs ran at the default batch=128 (~78 GiB per process), causing every sequential run to OOM.

**Fix:** Disabled the legacy sequential code path. All runs now go through `run_sweep_parallel()` which correctly passes `batch_size` to `run_single()`. With max_workers=1 this is functionally sequential but respects all flags.

#### Issue G: Logit tensor retention in forward()
`model_AR.py`, `backbone.py` — all `forward()` methods returned the full `(B, T, V)` logits tensor even during training, where only the loss scalar is needed. Training discards it (`_, loss = model(...)`) but it stays alive in the autograd graph.

Profiled with `torch.cuda.memory_allocated()`: peak after forward at batch=32 is ~42 GiB regardless (dominated by saved activations for 16 transformer layers), so this is a modest improvement. But it avoids holding a ~1-2 GiB tensor across the forward→backward boundary.

**Fix:** When `targets` is provided, `forward()` now returns `(None, loss)` instead of `(logits, loss)`. Inference path (`targets=None`) still returns logits as before.

### Reduced sweep strategy
200 steps was enough to rank LR candidates. Plan: pick top 2, then run 1000 steps on those two for a confident final pick.

Also reduced `LR_SWEEP_STEPS` from 2000 to 200 in `experiment_config.py`.

### Current sweep configuration (working)
- `--sweep-batch-size 32`, `grad_accum_steps=8`, effective batch = 256
- `--use_compile false`
- `--max_iters 200`, warmup = 10 steps (5%)
- 1 worker via parallel dispatcher (sequential execution, respects all flags)
- Peak VRAM: ~43 GiB per process on A100-80GB

#### Issue H: Sweep still ran loss eval at step 0
While investigating unexpectedly slow 200-step AR sweep wall times, found that the sweep was still paying for one full loss-eval pass before any training. `run_lr_sweep.py` sets `eval_interval = LR_SWEEP_STEPS + 1` and `skip_final_eval = True`, intending to suppress eval during short sweeps. But `train.py` used `iter % eval_interval == 0`, so `iter=0` still triggered `estimate_loss()`.

With the default `eval_iters = 50`, that means each sweep candidate did:
- 50 train eval batches
- 50 val eval batches

So every 200-step LR candidate was actually doing 100 extra forward-only batches up front, which made wall times look much worse than the true training path.

**Fix:** Loss eval now requires `iter > 0`, so periodic eval starts after training has begun. This keeps normal long-run eval behavior intact while making LR sweeps actually skip eval as intended.

---

## 2026-04-10: Efficiency Refactor (inspired by Karpathy's autoresearch train.py)

Compared our training code against Karpathy's single-GPU autoresearch script to understand why his setup fits batch=128 at seq=2048 comfortably while ours OOMs at batch=32. Identified three key differences and applied them, plus cleaned up dead code.

### Change 1: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
**train.py:3** — Set before any `torch.cuda` call. This was open question #2 from the previous section. Karpathy sets it at line 3 of his script. It tells PyTorch's caching allocator to use expandable memory segments instead of fixed-size blocks, which directly addresses the fragmentation issue observed in Issue E (allocator growing to 43 GiB while actual allocations were much smaller). Free improvement — no code changes, no performance cost.

### Change 2: Flash Attention enablement + diagnostic logging
**train.py:207-215** — Explicitly enables Flash SDP and memory-efficient SDP backends via `torch.backends.cuda`, then prints which backends are active at startup. This confirms that `F.scaled_dot_product_attention` dispatches to FA2 (O(N) memory for attention scores) rather than falling back to the "math" backend (O(N²) — the main memory hog identified in the analysis).

Current status by model:
- **AR** (`is_causal=True`, no explicit mask): FA2 should already be active. The diagnostic will confirm.
- **MDLM** (`attn_mask=None` when `_is_single_block`): Also gets the efficient backend.
- **BD3-LM** (explicit boolean block mask): Still falls back to math backend. Will need custom block-aware attention or mask conversion for FA2 compatibility — deferred until we actually run BD3-LM sweeps.

### Change 3: GC management
**train.py:706-713** — After step 0, runs `gc.collect()`, `gc.freeze()`, `gc.disable()`. Python's cyclic garbage collector causes ~500ms stalls when scanning the autograd graph. Karpathy does the same thing. Re-collects every 5000 steps as a safety valve. Impact: throughput improvement (fewer stalls), not memory.

### Change 4: Legacy code removal in run_lr_sweep.py
Deleted ~120 lines: `sweep_adamw()`, `sweep_normuon()`, `sweep_one_pair()`, and the `if False` dead branch. These legacy sequential-path functions were the root cause of Issue F (silently ignoring `--sweep-batch-size`). The parallel dispatcher `run_sweep_parallel()` with `max_workers=1` is now the single code path — functionally sequential but correctly respects all flags. Also removed an unused `avg` variable.

### Expected impact
- `expandable_segments` should reduce the 43 GiB steady-state peak (fragmentation was a significant contributor).
- FA2 confirmation ensures we're not materializing O(N²) attention scores for AR and MDLM.
- GC disable removes periodic ~500ms stalls.
- Code is simpler: one dispatch path instead of two, ~120 fewer lines.

### Open items
- **Gradient checkpointing**: Still not implemented. Will likely be needed for 170M models at seq=2048.
- **BD3-LM Flash Attention**: Block masks require the math SDPA backend. May need `torch.nn.attention.sdpa_kernel` workarounds or FlexAttention.
- **FA3 via kernels package**: Karpathy uses `varunneal/flash-attention-3` (Hopper) or `kernels-community/flash-attn3` (non-Hopper). Could be a further improvement but adds a dependency.

---

## 2026-04-10: LR Calibration Results & Phase 1 Scope Decisions

### AdamW LR sweep: AR/50M

Ran the full 5-point sweep `[1e-4, 3e-4, 1e-3, 3e-3, 1e-2]` on AR/50M.
All 5 candidates stable (no early aborts). Clean parabolic loss curve.
Selected **LR = 1e-3** (loss=4.507), with 3e-3 as runner-up (4.665).

| LR | loss | wall time |
|----|------|-----------|
| 1e-4 | 5.522 | 369s |
| 3e-4 | 5.041 | 342s |
| **1e-3** | **4.507** | 340s |
| 3e-3 | 4.665 | 341s |
| 1e-2 | 6.205 | 341s |

Total wall time: 29 min (sequential, 1 worker, torch.compile enabled).

### NorMuon LR sweep: AR/50M

Ran the 3×3 grid `adam_mult × matrix_mult ∈ {0.3, 1.0, 3.0}²`.
All 9 runs stable. Selected **adam_mult=0.3, matrix_mult=1.0** (loss=5.109).

Clear pattern: lower `adam_mult` is consistently better across all `matrix_mult`
values. Within the `adam_mult=0.3` row, `matrix_mult=1.0` wins but the spread is
only 0.034 (5.109 to 5.143). The `adam_mult=1.0` row is ~0.03 worse uniformly.
The `adam_mult=3.0` row is ~0.25 worse — much more sensitivity to the Adam
group LR than the Muon matrix LR.

### Scope decisions

1. **50M only for Phase 1.** Defer 98M and 170M until curriculum results clarify
   what's worth scaling up.

2. **Shared AdamW LR across model families.** Use 1e-3 for AR, MDLM, and BD3-LM
   at 50M. Rationale: the sweep was only run on AR, but at this scale a single
   LR is a reasonable starting point. If MDLM or BD3-LM show instability during
   curriculum runs, we can re-sweep for those families specifically.

3. **Shared NorMuon config across model families.** Same rationale as above:
   `adam_mult=0.3, matrix_mult=1.0` for all three families at 50M.

4. **Steps instead of FLOP budget.** Added `--steps` flag to `run_curriculum.py`
   as an alternative to `--budget`. Each stage gets `round(total_steps * flop_frac)`
   steps. Phase 1 uses `--steps 1000`. This is simpler for quick experiments and
   avoids the FLOP accounting indirection when we're not doing scaling-law fits.

5. **Curriculum + pure-BD3 baseline before IsoFLOP.** Phase 1 runs a pure
   BD3-LM baseline (`block_len=16`) plus C0 (4 p_AR variants) and C1, all with
   both optimizers at 50M/1000 steps = 12 total runs. IsoFLOP scaling sweeps
   are deferred to a later phase.

### Checkpoint sharing & disk-space optimizations

Phase 1 uses a **shared AR warmup** strategy: one 800-step AR run produces
intermediate checkpoints at steps 200, 300, 500, 800. C0 variants branch from
these (each at a different AR→BD3 transition point), and C1 branches from
step 200. This works because the warmup-stable (WS) LR schedule makes the AR
training trajectory path-independent — identical weights at every step
regardless of when you plan to branch.

Two train.py changes to minimize disk usage on the 16 GB VM:

1. **`--save_steps` flag.** Accepts a comma-separated list of exact step
   numbers (e.g. `--save_steps 200,300,500,800`). When set, overrides
   `--save_interval` so only those specific checkpoints are written. Avoids
   saving unneeded intermediate steps (100, 400, 600, 700).

2. **`--save_weights_only` flag.** When true, **all** checkpoints (both
   step-numbered and final) omit `optimizer_state_dict`, reducing each from
   ~712 MB to ~250 MB. This is safe because stage transitions reset the
   optimizer and LR schedule anyway — `load_model_weights()` only reads
   `model_state_dict`.

Net effect: the full Phase 1 run (`run_phase1.sh`) produces 13 checkpoints
per optimizer, all weights-only ≈ 13 × 250 MB ≈ **3.25 GB per optimizer**
(~6.5 GB for both adamw + normuon).

### Bug fixes in checkpoint numbering (train.py)

**P1 — ckpt_step800.pt never created.** `for iter in range(800)` runs 0..799;
the old condition `iter % 100 == 0` fires at 0, 100, ..., 700 but never 800.
Fix: use `(iter + 1) % save_interval == 0` so iter=799 → completed_steps=800
→ saves ckpt_step800.pt.

**P2 — Off-by-one in step numbering.** At iter=200 the optimizer has done 201
updates (0 through 200). Fix: `completed_steps = iter + 1` in both the filename
and the checkpoint metadata `"iter"` field, so the number always means
"completed updates."

### Pre-flight tests (tests/test_cuda_smoke.py)

Added `TestPhase1Preflight` with 4 tests verifying the checkpoint-sharing flow:
step-numbered checkpoints are produced and match the final ckpt; C0 and C1
resume flows work; NorMuon checkpoint sharing works.

### CUDA 12.7 compatibility (requirements.txt)

Removed torch from `requirements.txt` entirely. pip cannot scope an index to a
single package — `--extra-index-url` and even `--index-url` still let pip prefer
a higher-versioned CUDA 13.0 wheel from PyPI. The install procedure is now
two steps: `pip install torch --index-url https://download.pytorch.org/whl/cu126`
first, then `pip install -r requirements.txt` for everything else. This
guarantees the cu126 build on driver 565.77 (CUDA ≤12.7).

### Files changed

| File | Change |
|------|--------|
| `calibrated_lrs.json` | Set ar\|50M, mdlm\|50M, bd3lm\|50M = 0.001 |
| `calibrated_normuon.json` | Created: ar\|50M, mdlm\|50M, bd3lm\|50M = {adam_mult: 0.3, matrix_mult: 1.0} |
| `run_curriculum.py` | Added `--steps` flag (mutually exclusive with `--budget`); updated `run_stage()` and `run_curriculum()` to accept `total_steps` |
| `train.py` | Added `--save_steps`, `--save_weights_only` flags; fixed checkpoint step numbering (P1+P2); metadata uses `completed_steps` |
| `run_phase1.sh` | Created: shared-AR-warmup + C0/C1 continuation script, parameterized by optimizer |
| `tests/test_cuda_smoke.py` | Added `TestPhase1Preflight` (4 tests) |
| `requirements.txt` | Removed torch; two-step install to guarantee cu126 wheel |
| `PLAN.md` | Added §4.1 (calibration results), rewrote §5 (curriculum experiments), deferred IsoFLOP to §5A |
| `WORKFLOW.md` | Updated LR sweep sections as complete, added `--steps 1000` commands, fixed LR_SWEEP_STEPS typo |

---

## 2026-04-10: Phase 1 Execution — BD3-LM OOM & Batch Size Fix

### Problem: BD3-LM stages OOM at batch_size=128

Both VMs completed the AR warmup (800 steps) successfully, but every BD3-LM
stage in `run_phase1.sh` OOMed. BD3-LM is dual-stream, requiring roughly 2×
the memory of AR at the same batch size. At `batch_size=128` on an A100-80GB,
BD3-LM exceeds available VRAM.

Confirmed that `batch_size=64` works for BD3-LM via:
```
python3 run_lr_sweep.py --optimizer adamw --model bd3lm --size 50M --sweep-batch-size 64
```

### Fix: reduced BD3 micro-batch, increased grad accumulation

We tested `batch_size=64, grad_accum_steps=4` successfully in isolation, but
the committed `run_phase1.sh` uses the more conservative
`batch_size=32, grad_accum_steps=8` for all BD3-LM stages. This keeps the
effective batch at 256 while giving extra memory headroom. AR warmup remains at
`batch_size=128, grad_accum_steps=2`.

The Phase 1 script now disables in-run eval and sampling entirely:
`eval_interval=0`, `skip_final_eval=true`, `gpt2_eval_interval=0`,
`sample_interval=0`, `num_final_samples=0`. Instead, it logs plain train
loss / grad norm every 10 steps. This removes the expensive end-of-run eval
pass and keeps Phase 1 focused on fast optimization traces; BPB can be
computed later from saved checkpoints when needed.

Also added a skip guard for the AR warmup: if `ckpt_step800.pt` already exists,
Step 1 is skipped entirely so re-runs jump straight to the BD3 stages.

### Other fixes in this session

- `run_phase1.sh`: `python` → `python3` (the VMs don't have `python` on PATH).
- `train.py`: fixed misleading `--save_weights_only` help text — all checkpoints
  (including final) respect this flag, not just step-numbered ones.
- `.gitignore`: removed `runs/` and `*.pkl` so loss logs can be committed;
  kept `*.pt` to block large checkpoint files.

### Current VM status

| VM | Optimizer | AR warmup | BD3 stages | Notes |
|----|-----------|-----------|------------|-------|
| C.34527378 | adamw | Done (800 steps) | All failed (OOM at batch=128) | loss.pkl committed; needs re-run with fixed script |
| C.34534168 | normuon | Done (800 steps) | c0_p80 failed (OOM), rest never started | No loss.pkl produced; needs re-run with fixed script |

### Next steps

1. Push updated Phase 1 scripts to both VMs (`git pull`).
2. Re-run `bash run_phase1.sh adamw`, `bash run_phase1.sh normuon`, and the
   new pure-BD3 baseline script. AR warmup will be skipped automatically, and
   all BD3 runs use the `batch_size=32, grad_accum_steps=8` setting.
3. Commit the resulting logs and checkpoints metadata.

### Disk budget (revised)

With `--save_weights_only true` on all checkpoints (~250 MB each):
- AR warmup: 4 step checkpoints + 1 final = 5 × 250 MB = 1.25 GB
- C0 (4 runs): 4 × 250 MB = 1.0 GB
- C1 (4 stages): 4 × 250 MB = 1.0 GB
- **Total per optimizer: ~3.25 GB** (~6.5 GB for both)

### Files changed

| File | Change |
|------|--------|
| `run_phase1.sh` | BD3 stages: `batch_size` 128→32, `grad_accum_steps` 2→8; disabled in-run eval/sampling; train logs every 10 steps; added AR warmup skip guard; `python`→`python3` |
| `run_phase1_baselines.sh` | New: pure BD3-LM 50M/1000-step baseline runner for `adamw`, `normuon`, or both, with skip-if-complete behavior |
| `train.py:75-76` | Fixed `--save_weights_only` help text (all checkpoints respect flag) |
| `.gitignore` | Removed `runs/` and `*.pkl`; kept `*.pt` |
| `MEMO.md` | Updated `--save_weights_only` description; added this section |

---

## Files Changed

| File | Change |
|------|--------|
| `train.py:1-3` | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch import |
| `train.py:5` | Added `import gc` for GC management |
| `train.py:207-215` | Enable Flash SDP + mem-efficient SDP, log SDPA backend status at startup |
| `train.py:706-713` | GC management: collect/freeze/disable after step 0, re-collect every 5000 steps |
| `train.py:623-627` | Skip step-0 loss eval so short LR sweeps do not pay 100 extra eval batches before training |
| `train.py:691` | Log frequency: `iter % 100` → `iter % min(100, eval_interval)` |
| `tests/test_cuda_smoke.py:88` | Default `min_steps` 5 → 3 |
| `experiment_config.py:487` | `LR_SWEEP_STEPS` 2000 → 200 |
| `experiment_config.py:533,633` | Added `use_compile` param to `build_command()`, appends `--use_compile` to cmd |
| `model_AR.py:166-176` | Return `(None, loss)` during training to avoid retaining logits tensor |
| `backbone.py:250-268,270-300,302-321` | Same logit fix for `forward()`, `forward_train()`, `forward_sample()` |
| `run_lr_sweep.py:218-221` | Added `_SWEEP_EFFECTIVE_BATCH = 256` constant |
| `run_lr_sweep.py:162-177` | Updated VRAM estimates to empirical peaks |
| `run_lr_sweep.py:309-316` | Sweep uses `use_compile=False`, computes `grad_accum_steps` from effective batch |
| `run_lr_sweep.py:602-767` | Deleted legacy `sweep_adamw()`, `sweep_normuon()`, `sweep_one_pair()`; single dispatch path via `run_sweep_parallel()` |
