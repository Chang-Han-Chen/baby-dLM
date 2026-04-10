#!/usr/bin/env bash
# Phase 1 — Curriculum 2 (LLaDA 2 style aggressive early jump)
#
# Stages:  AR -> BD3(8) -> BD3(64) -> BD3(512) -> BD3(16)
# Split:   20% each  =  200 steps per stage  (1000 total)
# Optimizer: NorMuon only
#
# Reuses the shared AR warmup checkpoint (step 200) produced by run_phase1.sh normuon.
# If that checkpoint doesn't exist yet, this script runs the AR warmup first.
#
# Usage: bash run_phase1_c2.sh
set -euo pipefail

OPT="normuon"
BASE="runs/curriculum/shared_${OPT}"
C2="${BASE}/c2"
EVAL_INTERVAL=0
TRAIN_LOG_INTERVAL=10
SKIP_FINAL_EVAL=true
NUM_FINAL_SAMPLES=0

# NorMuon LR config (from calibration: adam_mult=0.3, matrix_mult=1.0)
LR_ARGS="--learning_rate 1.0 --min_lr 0.1 --adam_mult 0.3 --matrix_mult 1.0 --normuon_weight_decay 0.2"

maybe_run_stage() {
  local ckpt_path="$1"
  local label="$2"
  if [ -f "${ckpt_path}" ]; then
    echo ""
    echo "=== ${label} SKIPPED (${ckpt_path} exists) ==="
    return 1
  fi
  echo ""
  echo "=== ${label} ==="
  mkdir -p "$(dirname "${ckpt_path}")"
  return 0
}

echo "Running Curriculum 2 (LLaDA 2 style) with optimizer=${OPT}"
echo "Stages: AR(200) -> BD3(8,200) -> BD3(64,200) -> BD3(512,200) -> BD3(16,200)"
echo "Output: ${C2}/"

# ============================================================
# Step 0: Shared AR warmup (200 steps) — reuse from Phase 1
# ============================================================
AR_CKPT="${BASE}/ar_warmup/ckpt_step200.pt"
if [ -f "${AR_CKPT}" ]; then
  echo "=== AR warmup SKIPPED (${AR_CKPT} exists) ==="
else
  echo ""
  echo "=== AR warmup (800 steps, saving step 200 checkpoint) ==="
  mkdir -p "${BASE}/ar_warmup"

  python3 -u train.py \
    --data climbmix --model ar \
    --n_embd 768 --n_layer 7 --n_head 12 \
    --dropout 0.2 --batch_size 128 --block_size 2048 \
    --grad_accum_steps 2 --max_iters 800 \
    --warmup_stable true --warmup_iters 40 \
    --eval_interval "${EVAL_INTERVAL}" --eval_iters 50 \
    --train_log_interval "${TRAIN_LOG_INTERVAL}" --save_steps 200,300,500,800 --save_weights_only true \
    --skip_final_eval "${SKIP_FINAL_EVAL}" --skip_final_checkpoint false \
    --use_compile true --num_final_samples "${NUM_FINAL_SAMPLES}" \
    --gpt2_eval_interval 0 --gpt2_eval_samples 0 --sample_interval 0 \
    --loss_log_path "${BASE}/ar_warmup/loss.pkl" \
    --checkpoint_path "${BASE}/ar_warmup/ckpt.pt" \
    --optimizer "${OPT}" ${LR_ARGS}
fi

# ============================================================
# Stage 1: BD3(block_len=8) for 200 steps from AR step 200
# ============================================================
if maybe_run_stage "${C2}/bd3_bl8/ckpt.pt" "C2 stage 1: BD3(bl=8) for 200 steps from AR step 200"; then
  python3 -u train.py \
    --data climbmix --model bd3lm \
    --n_embd 768 --n_layer 7 --n_head 12 \
    --dropout 0.1 --batch_size 32 --block_size 2048 \
    --grad_accum_steps 8 --max_iters 200 \
    --warmup_stable true --warmup_iters 10 \
    --eval_interval "${EVAL_INTERVAL}" --eval_iters 50 \
    --train_log_interval "${TRAIN_LOG_INTERVAL}" --save_interval 1000 --save_weights_only true \
    --skip_final_eval "${SKIP_FINAL_EVAL}" --skip_final_checkpoint false \
    --use_compile false --num_final_samples "${NUM_FINAL_SAMPLES}" \
    --gpt2_eval_interval 0 --gpt2_eval_samples 0 --sample_interval 0 \
    --block_len 8 \
    --resume_from "${AR_CKPT}" \
    --loss_log_path "${C2}/bd3_bl8/loss.pkl" \
    --checkpoint_path "${C2}/bd3_bl8/ckpt.pt" \
    --optimizer "${OPT}" ${LR_ARGS}
fi

# ============================================================
# Stage 2: BD3(block_len=64) for 200 steps from bl=8 checkpoint
# ============================================================
if maybe_run_stage "${C2}/bd3_bl64/ckpt.pt" "C2 stage 2: BD3(bl=64) for 200 steps from bl=8 checkpoint"; then
  python3 -u train.py \
    --data climbmix --model bd3lm \
    --n_embd 768 --n_layer 7 --n_head 12 \
    --dropout 0.1 --batch_size 32 --block_size 2048 \
    --grad_accum_steps 8 --max_iters 200 \
    --warmup_stable true --warmup_iters 10 \
    --eval_interval "${EVAL_INTERVAL}" --eval_iters 50 \
    --train_log_interval "${TRAIN_LOG_INTERVAL}" --save_interval 1000 --save_weights_only true \
    --skip_final_eval "${SKIP_FINAL_EVAL}" --skip_final_checkpoint false \
    --use_compile false --num_final_samples "${NUM_FINAL_SAMPLES}" \
    --gpt2_eval_interval 0 --gpt2_eval_samples 0 --sample_interval 0 \
    --block_len 64 \
    --resume_from "${C2}/bd3_bl8/ckpt.pt" \
    --loss_log_path "${C2}/bd3_bl64/loss.pkl" \
    --checkpoint_path "${C2}/bd3_bl64/ckpt.pt" \
    --optimizer "${OPT}" ${LR_ARGS}
fi

# ============================================================
# Stage 3: BD3(block_len=512) for 200 steps from bl=64 checkpoint
# ============================================================
if maybe_run_stage "${C2}/bd3_bl512/ckpt.pt" "C2 stage 3: BD3(bl=512) for 200 steps from bl=64 checkpoint"; then
  python3 -u train.py \
    --data climbmix --model bd3lm \
    --n_embd 768 --n_layer 7 --n_head 12 \
    --dropout 0.1 --batch_size 32 --block_size 2048 \
    --grad_accum_steps 8 --max_iters 200 \
    --warmup_stable true --warmup_iters 10 \
    --eval_interval "${EVAL_INTERVAL}" --eval_iters 50 \
    --train_log_interval "${TRAIN_LOG_INTERVAL}" --save_interval 1000 --save_weights_only true \
    --skip_final_eval "${SKIP_FINAL_EVAL}" --skip_final_checkpoint false \
    --use_compile false --num_final_samples "${NUM_FINAL_SAMPLES}" \
    --gpt2_eval_interval 0 --gpt2_eval_samples 0 --sample_interval 0 \
    --block_len 512 \
    --resume_from "${C2}/bd3_bl64/ckpt.pt" \
    --loss_log_path "${C2}/bd3_bl512/loss.pkl" \
    --checkpoint_path "${C2}/bd3_bl512/ckpt.pt" \
    --optimizer "${OPT}" ${LR_ARGS}
fi

# ============================================================
# Stage 4: BD3(block_len=16) for 200 steps from bl=512 checkpoint
# ============================================================
if maybe_run_stage "${C2}/bd3_bl16/ckpt.pt" "C2 stage 4: BD3(bl=16) for 200 steps from bl=512 checkpoint"; then
  python3 -u train.py \
    --data climbmix --model bd3lm \
    --n_embd 768 --n_layer 7 --n_head 12 \
    --dropout 0.1 --batch_size 32 --block_size 2048 \
    --grad_accum_steps 8 --max_iters 200 \
    --warmup_stable true --warmup_iters 10 \
    --eval_interval "${EVAL_INTERVAL}" --eval_iters 50 \
    --train_log_interval "${TRAIN_LOG_INTERVAL}" --save_interval 1000 --save_weights_only true \
    --skip_final_eval "${SKIP_FINAL_EVAL}" --skip_final_checkpoint false \
    --use_compile false --num_final_samples "${NUM_FINAL_SAMPLES}" \
    --gpt2_eval_interval 0 --gpt2_eval_samples 0 --sample_interval 0 \
    --block_len 16 \
    --resume_from "${C2}/bd3_bl512/ckpt.pt" \
    --loss_log_path "${C2}/bd3_bl16/loss.pkl" \
    --checkpoint_path "${C2}/bd3_bl16/ckpt.pt" \
    --optimizer "${OPT}" ${LR_ARGS}
fi

echo ""
echo "============================================================"
echo "Curriculum 2 complete (optimizer=${OPT})"
echo "Final checkpoint: ${C2}/bd3_bl16/ckpt.pt"
echo "Results in ${C2}/"
echo "============================================================"
