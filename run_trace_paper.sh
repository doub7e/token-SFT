#!/usr/bin/env bash
# Paper-aligned ASFT training with per-token gradient tracing.
# Qwen2.5-7B, LoRA r=32, numina_cot_100K, lr=5e-5, kl=0.05, max_len=2048
set -euo pipefail

WORK_DIR="/scratch/cvlab/home/shuli/agentic-research/ASFT"
MODEL_DIR="${WORK_DIR}/models/Qwen2.5-7B"
DATA_DIR="${WORK_DIR}/data"
OUTPUT_DIR="${WORK_DIR}/output/asft-qwen2.5-7b-paper-trace"

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

# ── 3. Train with gradient tracing ────────────────────────────────────────
echo "==> Starting ASFT training (Qwen2.5-7B, LoRA r=32, paper config + trace)..."
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
    --seed 42 \
    --trace_every_n_steps 20 \
    --trace_proj_dim_factor 256 \
    --trace_module_filter "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

echo "==> Done. Checkpoint: ${OUTPUT_DIR}"
echo "==> Trace files: ${OUTPUT_DIR}/token_grad_trace/"
