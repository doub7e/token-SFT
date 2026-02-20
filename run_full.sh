#!/usr/bin/env bash
# run_full.sh - Full ASFT training with original config (3 epochs).
set -euo pipefail

WORK_DIR="/scratch/cvlab/home/shuli/agentic-research/ASFT"
MODEL_DIR="${WORK_DIR}/models/Qwen2.5-1.5B"
DATA_DIR="${WORK_DIR}/data"
OUTPUT_DIR="${WORK_DIR}/output/asft-qwen2.5-1.5b-full"

cd "${WORK_DIR}"

echo "==> Starting ASFT training (Qwen2.5-1.5B, LoRA, 3 epochs)..."
python train_v2.py \
    --model_name_or_path "${MODEL_DIR}" \
    --data_path "${DATA_DIR}/numina_cot_10k.jsonl" \
    --mode asft \
    --kl_weight 0.03 \
    --use_lora True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --learning_rate 5e-4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --global_batch_size 64 \
    --output_dir "${OUTPUT_DIR}" \
    --precision bf16 \
    --seed 42

echo "==> Done. Checkpoint saved to ${OUTPUT_DIR}"
