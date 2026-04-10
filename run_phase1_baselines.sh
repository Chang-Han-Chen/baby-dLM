#!/usr/bin/env bash
# Phase 1 pure-BD3 baseline runs
# Usage:
#   bash run_phase1_baselines.sh          # run both optimizers
#   bash run_phase1_baselines.sh adamw    # run one optimizer
#   bash run_phase1_baselines.sh normuon
set -euo pipefail

TARGET=${1:-all}
EVAL_INTERVAL=0
TRAIN_LOG_INTERVAL=10
SKIP_FINAL_EVAL=true
NUM_FINAL_SAMPLES=0

case "${TARGET}" in
  adamw|normuon|all)
    ;;
  *)
    echo "Usage: bash run_phase1_baselines.sh [adamw|normuon|all]" >&2
    exit 1
    ;;
esac

run_one() {
  local opt="$1"
  local base="runs/curriculum/baseline_bd3lm_16_50M_1000steps_${opt}"

  echo ""
  echo "Running Phase 1 pure-BD3 baseline with optimizer=${opt}"
  echo "Output: ${base}/"

  if [ -f "${base}/ckpt.pt" ]; then
    echo "=== Pure BD3 baseline SKIPPED (${base}/ckpt.pt exists) ==="
    return 0
  fi

  mkdir -p "${base}"

  python3 -u train.py \
    --data climbmix --model bd3lm \
    --n_embd 768 --n_layer 7 --n_head 12 \
    --dropout 0.1 --batch_size 32 --block_size 2048 \
    --grad_accum_steps 8 --max_iters 1000 \
    --warmup_stable true --warmup_iters 50 \
    --eval_interval "${EVAL_INTERVAL}" --eval_iters 50 \
    --train_log_interval "${TRAIN_LOG_INTERVAL}" --save_interval 1000 --save_weights_only true \
    --skip_final_eval "${SKIP_FINAL_EVAL}" --skip_final_checkpoint false \
    --use_compile false --num_final_samples "${NUM_FINAL_SAMPLES}" \
    --gpt2_eval_interval 0 --gpt2_eval_samples 0 --sample_interval 0 \
    --block_len 16 \
    --loss_log_path "${base}/loss.pkl" \
    --checkpoint_path "${base}/ckpt.pt" \
    --optimizer "${opt}" \
    $(if [ "${opt}" = "adamw" ]; then echo "--learning_rate 0.001 --min_lr 0.0001"; else echo "--learning_rate 1.0 --min_lr 0.1 --adam_mult 0.3 --matrix_mult 1.0 --normuon_weight_decay 0.2"; fi)
}

if [ "${TARGET}" = "all" ]; then
  run_one adamw
  run_one normuon
else
  run_one "${TARGET}"
fi

echo ""
echo "============================================================"
echo "Phase 1 pure-BD3 baseline complete for target=${TARGET}"
echo "============================================================"
