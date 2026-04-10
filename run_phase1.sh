#!/usr/bin/env bash
# Phase 1 curriculum runs with checkpoint sharing
# Usage: bash run_phase1.sh adamw    (or: bash run_phase1.sh normuon)
set -euo pipefail

OPT=${1:?"Usage: bash run_phase1.sh <adamw|normuon>"}
BASE="runs/curriculum/shared_${OPT}"

echo "Running Phase 1 curricula with optimizer=${OPT}"
echo "Output: ${BASE}/"

# ============================================================
# Step 1: Shared AR warmup (800 steps, checkpoints every 100)
# ============================================================
echo ""
echo "=== Step 1: AR warmup (800 steps) ==="
mkdir -p "${BASE}/ar_warmup"

python -u train.py \
  --data climbmix --model ar \
  --n_embd 768 --n_layer 7 --n_head 12 \
  --dropout 0.2 --batch_size 128 --block_size 2048 \
  --grad_accum_steps 2 --max_iters 800 \
  --warmup_stable true --warmup_iters 40 \
  --eval_interval 300 --eval_iters 50 \
  --train_log_interval 100 --save_steps 200,300,500,800 --save_weights_only true \
  --skip_final_eval false --skip_final_checkpoint false \
  --use_compile true --num_final_samples 5 \
  --gpt2_eval_interval 0 --gpt2_eval_samples 0 --sample_interval 0 \
  --loss_log_path "${BASE}/ar_warmup/loss.pkl" \
  --checkpoint_path "${BASE}/ar_warmup/ckpt.pt" \
  --optimizer "${OPT}" \
  $(if [ "${OPT}" = "adamw" ]; then echo "--learning_rate 0.001 --min_lr 0.0001"; else echo "--learning_rate 1.0 --min_lr 0.1 --adam_mult 0.3 --matrix_mult 1.0 --normuon_weight_decay 0.2"; fi)

# ============================================================
# Step 2: C0 — BD3(16) continuations from AR checkpoints
# ============================================================

echo ""
echo "=== C0 p80: BD3(16) for 200 steps from AR step 800 ==="
mkdir -p "${BASE}/c0_p80_bd3"

python -u train.py \
  --data climbmix --model bd3lm \
  --n_embd 768 --n_layer 7 --n_head 12 \
  --dropout 0.1 --batch_size 128 --block_size 2048 \
  --grad_accum_steps 2 --max_iters 200 \
  --warmup_stable true --warmup_iters 10 \
  --eval_interval 300 --eval_iters 50 \
  --train_log_interval 100 --save_interval 1000 \
  --skip_final_eval false --skip_final_checkpoint false \
  --use_compile true --num_final_samples 5 \
  --gpt2_eval_interval 0 --gpt2_eval_samples 0 --sample_interval 0 \
  --block_len 16 \
  --resume_from "${BASE}/ar_warmup/ckpt_step800.pt" \
  --loss_log_path "${BASE}/c0_p80_bd3/loss.pkl" \
  --checkpoint_path "${BASE}/c0_p80_bd3/ckpt.pt" \
  --optimizer "${OPT}" \
  $(if [ "${OPT}" = "adamw" ]; then echo "--learning_rate 0.001 --min_lr 0.0001"; else echo "--learning_rate 1.0 --min_lr 0.1 --adam_mult 0.3 --matrix_mult 1.0 --normuon_weight_decay 0.2"; fi)

echo ""
echo "=== C0 p50: BD3(16) for 500 steps from AR step 500 ==="
mkdir -p "${BASE}/c0_p50_bd3"

python -u train.py \
  --data climbmix --model bd3lm \
  --n_embd 768 --n_layer 7 --n_head 12 \
  --dropout 0.1 --batch_size 128 --block_size 2048 \
  --grad_accum_steps 2 --max_iters 500 \
  --warmup_stable true --warmup_iters 25 \
  --eval_interval 300 --eval_iters 50 \
  --train_log_interval 100 --save_interval 1000 \
  --skip_final_eval false --skip_final_checkpoint false \
  --use_compile true --num_final_samples 5 \
  --gpt2_eval_interval 0 --gpt2_eval_samples 0 --sample_interval 0 \
  --block_len 16 \
  --resume_from "${BASE}/ar_warmup/ckpt_step500.pt" \
  --loss_log_path "${BASE}/c0_p50_bd3/loss.pkl" \
  --checkpoint_path "${BASE}/c0_p50_bd3/ckpt.pt" \
  --optimizer "${OPT}" \
  $(if [ "${OPT}" = "adamw" ]; then echo "--learning_rate 0.001 --min_lr 0.0001"; else echo "--learning_rate 1.0 --min_lr 0.1 --adam_mult 0.3 --matrix_mult 1.0 --normuon_weight_decay 0.2"; fi)

echo ""
echo "=== C0 p30: BD3(16) for 700 steps from AR step 300 ==="
mkdir -p "${BASE}/c0_p30_bd3"

python -u train.py \
  --data climbmix --model bd3lm \
  --n_embd 768 --n_layer 7 --n_head 12 \
  --dropout 0.1 --batch_size 128 --block_size 2048 \
  --grad_accum_steps 2 --max_iters 700 \
  --warmup_stable true --warmup_iters 35 \
  --eval_interval 300 --eval_iters 50 \
  --train_log_interval 100 --save_interval 1000 \
  --skip_final_eval false --skip_final_checkpoint false \
  --use_compile true --num_final_samples 5 \
  --gpt2_eval_interval 0 --gpt2_eval_samples 0 --sample_interval 0 \
  --block_len 16 \
  --resume_from "${BASE}/ar_warmup/ckpt_step300.pt" \
  --loss_log_path "${BASE}/c0_p30_bd3/loss.pkl" \
  --checkpoint_path "${BASE}/c0_p30_bd3/ckpt.pt" \
  --optimizer "${OPT}" \
  $(if [ "${OPT}" = "adamw" ]; then echo "--learning_rate 0.001 --min_lr 0.0001"; else echo "--learning_rate 1.0 --min_lr 0.1 --adam_mult 0.3 --matrix_mult 1.0 --normuon_weight_decay 0.2"; fi)

echo ""
echo "=== C0 p20: BD3(16) for 800 steps from AR step 200 ==="
mkdir -p "${BASE}/c0_p20_bd3"

python -u train.py \
  --data climbmix --model bd3lm \
  --n_embd 768 --n_layer 7 --n_head 12 \
  --dropout 0.1 --batch_size 128 --block_size 2048 \
  --grad_accum_steps 2 --max_iters 800 \
  --warmup_stable true --warmup_iters 40 \
  --eval_interval 300 --eval_iters 50 \
  --train_log_interval 100 --save_interval 1000 \
  --skip_final_eval false --skip_final_checkpoint false \
  --use_compile true --num_final_samples 5 \
  --gpt2_eval_interval 0 --gpt2_eval_samples 0 --sample_interval 0 \
  --block_len 16 \
  --resume_from "${BASE}/ar_warmup/ckpt_step200.pt" \
  --loss_log_path "${BASE}/c0_p20_bd3/loss.pkl" \
  --checkpoint_path "${BASE}/c0_p20_bd3/ckpt.pt" \
  --optimizer "${OPT}" \
  $(if [ "${OPT}" = "adamw" ]; then echo "--learning_rate 0.001 --min_lr 0.0001"; else echo "--learning_rate 1.0 --min_lr 0.1 --adam_mult 0.3 --matrix_mult 1.0 --normuon_weight_decay 0.2"; fi)

# ============================================================
# Step 3: C1 — Geometric doubling from shared AR step 200
# ============================================================

echo ""
echo "=== C1: BD3(bl=2) for 200 steps from shared AR step 200 ==="
mkdir -p "${BASE}/c1_bd3_bl2"

python -u train.py \
  --data climbmix --model bd3lm \
  --n_embd 768 --n_layer 7 --n_head 12 \
  --dropout 0.1 --batch_size 128 --block_size 2048 \
  --grad_accum_steps 2 --max_iters 200 \
  --warmup_stable true --warmup_iters 10 \
  --eval_interval 300 --eval_iters 50 \
  --train_log_interval 100 --save_interval 1000 \
  --skip_final_eval false --skip_final_checkpoint false \
  --use_compile true --num_final_samples 5 \
  --gpt2_eval_interval 0 --gpt2_eval_samples 0 --sample_interval 0 \
  --block_len 2 \
  --resume_from "${BASE}/ar_warmup/ckpt_step200.pt" \
  --loss_log_path "${BASE}/c1_bd3_bl2/loss.pkl" \
  --checkpoint_path "${BASE}/c1_bd3_bl2/ckpt.pt" \
  --optimizer "${OPT}" \
  $(if [ "${OPT}" = "adamw" ]; then echo "--learning_rate 0.001 --min_lr 0.0001"; else echo "--learning_rate 1.0 --min_lr 0.1 --adam_mult 0.3 --matrix_mult 1.0 --normuon_weight_decay 0.2"; fi)

echo ""
echo "=== C1: BD3(bl=4) for 200 steps from bl=2 checkpoint ==="
mkdir -p "${BASE}/c1_bd3_bl4"

python -u train.py \
  --data climbmix --model bd3lm \
  --n_embd 768 --n_layer 7 --n_head 12 \
  --dropout 0.1 --batch_size 128 --block_size 2048 \
  --grad_accum_steps 2 --max_iters 200 \
  --warmup_stable true --warmup_iters 10 \
  --eval_interval 300 --eval_iters 50 \
  --train_log_interval 100 --save_interval 1000 \
  --skip_final_eval false --skip_final_checkpoint false \
  --use_compile true --num_final_samples 5 \
  --gpt2_eval_interval 0 --gpt2_eval_samples 0 --sample_interval 0 \
  --block_len 4 \
  --resume_from "${BASE}/c1_bd3_bl2/ckpt.pt" \
  --loss_log_path "${BASE}/c1_bd3_bl4/loss.pkl" \
  --checkpoint_path "${BASE}/c1_bd3_bl4/ckpt.pt" \
  --optimizer "${OPT}" \
  $(if [ "${OPT}" = "adamw" ]; then echo "--learning_rate 0.001 --min_lr 0.0001"; else echo "--learning_rate 1.0 --min_lr 0.1 --adam_mult 0.3 --matrix_mult 1.0 --normuon_weight_decay 0.2"; fi)

echo ""
echo "=== C1: BD3(bl=8) for 200 steps from bl=4 checkpoint ==="
mkdir -p "${BASE}/c1_bd3_bl8"

python -u train.py \
  --data climbmix --model bd3lm \
  --n_embd 768 --n_layer 7 --n_head 12 \
  --dropout 0.1 --batch_size 128 --block_size 2048 \
  --grad_accum_steps 2 --max_iters 200 \
  --warmup_stable true --warmup_iters 10 \
  --eval_interval 300 --eval_iters 50 \
  --train_log_interval 100 --save_interval 1000 \
  --skip_final_eval false --skip_final_checkpoint false \
  --use_compile true --num_final_samples 5 \
  --gpt2_eval_interval 0 --gpt2_eval_samples 0 --sample_interval 0 \
  --block_len 8 \
  --resume_from "${BASE}/c1_bd3_bl4/ckpt.pt" \
  --loss_log_path "${BASE}/c1_bd3_bl8/loss.pkl" \
  --checkpoint_path "${BASE}/c1_bd3_bl8/ckpt.pt" \
  --optimizer "${OPT}" \
  $(if [ "${OPT}" = "adamw" ]; then echo "--learning_rate 0.001 --min_lr 0.0001"; else echo "--learning_rate 1.0 --min_lr 0.1 --adam_mult 0.3 --matrix_mult 1.0 --normuon_weight_decay 0.2"; fi)

echo ""
echo "=== C1: BD3(bl=16) for 200 steps from bl=8 checkpoint ==="
mkdir -p "${BASE}/c1_bd3_bl16"

python -u train.py \
  --data climbmix --model bd3lm \
  --n_embd 768 --n_layer 7 --n_head 12 \
  --dropout 0.1 --batch_size 128 --block_size 2048 \
  --grad_accum_steps 2 --max_iters 200 \
  --warmup_stable true --warmup_iters 10 \
  --eval_interval 300 --eval_iters 50 \
  --train_log_interval 100 --save_interval 1000 \
  --skip_final_eval false --skip_final_checkpoint false \
  --use_compile true --num_final_samples 5 \
  --gpt2_eval_interval 0 --gpt2_eval_samples 0 --sample_interval 0 \
  --block_len 16 \
  --resume_from "${BASE}/c1_bd3_bl8/ckpt.pt" \
  --loss_log_path "${BASE}/c1_bd3_bl16/loss.pkl" \
  --checkpoint_path "${BASE}/c1_bd3_bl16/ckpt.pt" \
  --optimizer "${OPT}" \
  $(if [ "${OPT}" = "adamw" ]; then echo "--learning_rate 0.001 --min_lr 0.0001"; else echo "--learning_rate 1.0 --min_lr 0.1 --adam_mult 0.3 --matrix_mult 1.0 --normuon_weight_decay 0.2"; fi)

echo ""
echo "============================================================"
echo "Phase 1 complete for optimizer=${OPT}"
echo "Results in ${BASE}/"
echo "============================================================"
