#!/usr/bin/env bash
# Run analysis on 7B paper-aligned trace data
set -eo pipefail
cd /scratch/cvlab/home/shuli/agentic-research/ASFT
export PYTHONPATH=/tmp/pylibs:${PYTHONPATH:-}
pip install --target=/tmp/pylibs plotly matplotlib pandas 2>/dev/null | tail -1

python analyze_token_gradients.py \
    --trace-dir output/asft-qwen2.5-7b-paper-trace/token_grad_trace \
    --model-path models/Qwen2.5-7B \
    --output-dir output/asft-qwen2.5-7b-paper-trace/token_grad_trace/analysis

echo "==> 7B analysis complete"
