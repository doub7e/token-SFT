#!/usr/bin/env bash
# run_paper.sh - Paper-aligned ASFT reproduction:
#   Qwen2.5-7B, LoRA r=32 (full FT OOMs even on 4×H100 due to reference model),
#   numina_cot_100K, model_max_length=2048, 1 epoch
#   Matches paper: lr=5e-5, global_batch=256, kl=0.05
set -euo pipefail

WORK_DIR="/scratch/cvlab/home/shuli/agentic-research/ASFT"
MODEL_DIR="${WORK_DIR}/models/Qwen2.5-7B"
DATA_DIR="${WORK_DIR}/data"
OUTPUT_DIR="${WORK_DIR}/output/asft-qwen2.5-7b-paper"

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
# Paper: Qwen2.5-7B, 100K, lr=5e-5, global_batch=256, max_len=2048, kl=0.05, 1ep
# Full FT OOMs on 4×H100 (reference model not ZeRO-sharded = +14GB/GPU).
# Using LoRA r=32 instead: reference via disable_adapter (zero extra memory).
echo "==> Starting ASFT training (Qwen2.5-7B, LoRA r=32, paper hyperparams)..."
python train_v2.py \
    --model_name_or_path "${MODEL_DIR}" \
    --data_path "${DATA_DIR}/numina_cot_100k.jsonl" \
    --mode asft \
    --kl_weight 0.05 \
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
