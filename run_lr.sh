#!/usr/bin/env bash
# run_lr.sh - Scheme C: Likelihood Ratio Weighting
#   Same base config as run_paper.sh (Qwen2.5-7B, LoRA r=32, 100K, 1ep)
#   Mode: sft-lr with lr_weight_clamp_max=5.0
#   Upweight tokens where model got worse than reference (p_ref/p_theta)
set -euo pipefail

WORK_DIR="/scratch/cvlab/home/shuli/agentic-research/ASFT"
MODEL_DIR="${WORK_DIR}/models/Qwen2.5-7B"
DATA_DIR="${WORK_DIR}/data"
OUTPUT_DIR="${WORK_DIR}/output/sft-lr-qwen2.5-7b"

cd "${WORK_DIR}"

# ── 1. Download model ──────────────────────────────────────────────────────
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "==> Downloading Qwen2.5-7B..."
    huggingface-cli download Qwen/Qwen2.5-7B \
        --local-dir "${MODEL_DIR}" \
        --local-dir-use-symlinks False
else
    echo "==> Model already exists, skipping download."
fi

# ── 2. Download 100K training data ────────────────────────────────────────
if [ ! -f "${DATA_DIR}/numina_cot_100k.jsonl" ]; then
    echo "==> Downloading ASFT dataset (100K)..."
    python download_data.py --output_dir "${DATA_DIR}"
else
    echo "==> Data already exists, skipping download."
fi

# ── 3. Train ───────────────────────────────────────────────────────────────
echo "==> Starting sft-lr training (Qwen2.5-7B, LoRA r=32, clamp_max=5.0)..."
python train_v2.py \
    --model_name_or_path "${MODEL_DIR}" \
    --data_path "${DATA_DIR}/numina_cot_100k.jsonl" \
    --mode sft-lr \
    --lr_weight_clamp_max 5.0 \
    --use_lora True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --global_batch_size 256 \
    --model_max_length 2048 \
    --output_dir "${OUTPUT_DIR}" \
    --precision bf16 \
    --seed 42

echo "==> Done. Checkpoint saved to ${OUTPUT_DIR}"
