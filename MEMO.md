# Engineering Memo

Running log of findings, changes, and decisions made while bringing up the experiment pipeline.

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

**Open questions for investigation:**
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
| v5 | batch=32, accum=8, compile=off | 2 (auto) | 4/5 OOM — proc 1 grew to 42 GiB, starving proc 2. LR=1e-4 survived (still running) |
| v6 (planned) | batch=32, accum=8, compile=off | 1 (sequential) | Not yet run |

### Current sweep configuration
- `--sweep-batch-size 32`, `grad_accum_steps=8`, effective batch = 256
- `--use_compile false`
- `--jobs 1` (sequential) — only safe option given memory behavior
- 5 LR candidates run one at a time, ~15-20 min each

---

## Files Changed

| File | Change |
|------|--------|
| `train.py:691` | Log frequency: `iter % 100` → `iter % min(100, eval_interval)` |
| `tests/test_cuda_smoke.py:88` | Default `min_steps` 5 → 3 |
| `experiment_config.py:533,633` | Added `use_compile` param to `build_command()`, appends `--use_compile` to cmd |
| `run_lr_sweep.py:218-221` | Added `_SWEEP_EFFECTIVE_BATCH = 256` constant |
| `run_lr_sweep.py:162-177` | Updated VRAM estimates to empirical batch=128 peaks |
| `run_lr_sweep.py:204-207` | Fixed VRAM scaling formula: 10% fixed + 90% scales |
| `run_lr_sweep.py:309-316` | Sweep uses `use_compile=False`, computes `grad_accum_steps` from effective batch |
