#!/usr/bin/env bash
# run_eval_full.sh - Merge full-run LoRA checkpoint then run math evaluation.
set -euo pipefail

WORK_DIR="/scratch/cvlab/home/shuli/agentic-research/ASFT"
BASE_MODEL="${WORK_DIR}/models/Qwen2.5-1.5B"
LORA_CKPT="${WORK_DIR}/output/asft-qwen2.5-1.5b-full"
MERGED_MODEL="${WORK_DIR}/output/asft-qwen2.5-1.5b-full-merged"
EVAL_OUTPUT="${WORK_DIR}/output/asft-qwen2.5-1.5b-full-matheval"

cd "${WORK_DIR}"

# ── 1. Merge LoRA ──────────────────────────────────────────────────────────
if [ ! -f "${MERGED_MODEL}/config.json" ]; then
    echo "==> Merging LoRA adapter (full run)..."
    BASE_MODEL="${BASE_MODEL}" LORA_CKPT="${LORA_CKPT}" MERGED_MODEL="${MERGED_MODEL}" \
    python - <<'PYEOF'
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_path  = os.environ["BASE_MODEL"]
lora_path  = os.environ["LORA_CKPT"]
save_path  = os.environ["MERGED_MODEL"]

print(f"Base:  {base_path}")
print(f"LoRA:  {lora_path}")
print(f"Save:  {save_path}")

tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, lora_path, torch_dtype=torch.float16)
merged = model.merge_and_unload()
os.makedirs(save_path, exist_ok=True)
merged.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Merge complete!")
PYEOF
else
    echo "==> Merged model already exists, skipping merge."
fi

# ── 2. Math eval ───────────────────────────────────────────────────────────
echo "==> Running math evaluation on MATH-500..."
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
