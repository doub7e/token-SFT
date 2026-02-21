#!/usr/bin/env bash
# run_eval_lr.sh - Merge sft-lr LoRA checkpoint then run MATH-500 evaluation.
set -euo pipefail

WORK_DIR="/scratch/cvlab/home/shuli/agentic-research/ASFT"
BASE_MODEL="${WORK_DIR}/models/Qwen2.5-7B"
LORA_CKPT="${WORK_DIR}/output/sft-lr-qwen2.5-7b"
MERGED_MODEL="${WORK_DIR}/output/sft-lr-qwen2.5-7b-merged"
EVAL_OUTPUT="${WORK_DIR}/output/sft-lr-qwen2.5-7b-matheval"

cd "${WORK_DIR}"

export BASE_MODEL LORA_CKPT MERGED_MODEL

# ── 1. Merge LoRA into base model ─────────────────────────────────────────
if [ ! -f "${MERGED_MODEL}/config.json" ]; then
    echo "==> Merging LoRA adapter into 7B base model..."
    python - <<'PYEOF'
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_path  = os.environ.get("BASE_MODEL")
lora_path  = os.environ.get("LORA_CKPT")
save_path  = os.environ.get("MERGED_MODEL")

print(f"Base:  {base_path}")
print(f"LoRA:  {lora_path}")
print(f"Save:  {save_path}")

tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, lora_path, torch_dtype=torch.bfloat16)
merged = model.merge_and_unload()
os.makedirs(save_path, exist_ok=True)
merged.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Merge complete!")
PYEOF
else
    echo "==> Merged model already exists, skipping merge."
fi

# ── 2. Run math evaluation (MATH-500) ─────────────────────────────────────
echo "==> Running MATH-500 evaluation..."
cd "${WORK_DIR}/eval/math_evaluation"

export CUDA_VISIBLE_DEVICES=0

TOKENIZERS_PARALLELISM=false python3 -u math_eval.py \
    --model_name_or_path "${MERGED_MODEL}" \
    --data_names "math_oai" \
    --output_dir "${EVAL_OUTPUT}" \
    --split "test" \
    --prompt_type "cot" \
    --num_test_sample -1 \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm

echo "==> Evaluation done. Results at ${EVAL_OUTPUT}"
