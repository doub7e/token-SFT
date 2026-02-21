#!/usr/bin/env bash
# Launch the per-token gradient visualizer (Gradio).
# For RunAI interactive jobs: kubectl port-forward <pod> 7863:7863
set -euo pipefail

WORK_DIR="/scratch/cvlab/home/shuli/agentic-research/ASFT"
TRACE_DIR="${WORK_DIR}/output/asft-qwen2.5-1.5b-trace/token_grad_trace"
MODEL_PATH="${WORK_DIR}/models/Qwen2.5-1.5B"

cd "${WORK_DIR}"

export EXTRA_LIBS="/scratch/cvlab/home/shuli/agentic-research/ASFT/.pip_libs"
pip install --target="${EXTRA_LIBS}" plotly scikit-learn 2>/dev/null || true
export PYTHONPATH="${EXTRA_LIBS}:${PYTHONPATH:-}"

echo "==> Launching per-token gradient visualizer on port 7863..."
echo "    Trace dir: ${TRACE_DIR}"
echo "    Model path: ${MODEL_PATH}"
python visualize_token_gradients.py \
    --trace-dir "${TRACE_DIR}" \
    --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 7863
