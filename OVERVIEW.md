# ASFT Project Overview

## What
Reproduction and extension of [ASFT (Anchored Supervised Fine-Tuning)](https://arxiv.org/abs/2509.23753) — an ICLR 2026 method that adds KL divergence regularization to SFT for LLM post-training.

**Upstream**: [zhuchichi56/ASFT](https://github.com/zhuchichi56/ASFT)
**Our fork**: [doub7e/token-SFT](https://github.com/doub7e/token-SFT)

## Core Idea
ASFT = DFT loss + KL(policy ‖ reference). The `--mode` flag in `train_v2.py` controls the training mode:
- `sft`: standard supervised fine-tuning.
- `dft`: distribution-level fine-tuning (token-level KL to ground truth).
- `sft+kl`: SFT + KL regularization to reference model.
- `asft`: DFT + KL regularization to reference model (the full method).

## Directory Layout
- `train_v2.py`: main training script (all modes: sft, dft, sft+kl, asft).
- `train_v2.sh`: original training shell script from upstream.
- `scripts/`: DeepSpeed config files (`ds_zero2_*.json`, `ds_zero3_*.json`).
- `eval/`: evaluation code.
  - `eval/math_evaluation/`: MATH-500 and other math benchmarks (vLLM-based).
  - `eval/alpacaeval/`: AlpacaEval benchmark.
  - `eval/medeval/`: medical evaluation.
- `dev/`: development/debugging scripts.
- `merge_lora.py` / `merge.sh`: LoRA adapter merging utilities.
- `download_data.py`: dataset downloader from HuggingFace.
- `data/`: training data (numina_cot_100k, open_math_instruct_2, etc.).
- `models/`: local model snapshots (Qwen2.5-1.5B, Qwen2.5-7B).
- `output/`: training checkpoints and eval results.
- `diary/`: per-commit diary entries and experiment progress.
- `run_*.sh`: our experiment scripts (see below).

## Our Run Scripts
| Script | Description |
| --- | --- |
| `run_quick.sh` | Quick 200-step sanity check (Qwen2.5-1.5B, LoRA r=8, 10K data) |
| `run_full.sh` | Full 3-epoch training (Qwen2.5-1.5B, LoRA r=8, 10K data) |
| `run_paper.sh` | Paper-aligned config (Qwen2.5-7B, LoRA r=32, 100K, 1 epoch) |
| `run_eval.sh` | Merge LoRA + MATH-500 eval for quick-run checkpoint |
| `run_eval_full.sh` | Merge LoRA + MATH-500 eval for full-run checkpoint |
| `run_eval_paper.sh` | Merge LoRA + MATH-500 eval for paper-aligned checkpoint |

## Key Execution Chain
1. `run_*.sh` calls `python train_v2.py` with config flags.
2. `train_v2.py` loads model + data, builds `CustomSFTTrainer` (extends trl's `SFTTrainer`).
3. In ASFT mode: computes standard loss + `kl_weight * KL(policy ‖ reference)`.
4. Reference model: in LoRA mode uses `disable_adapter()` (zero extra memory); in full FT loads separate `original_model`.
5. Eval scripts merge LoRA → run `eval/math_evaluation/math_eval.py` with vLLM.

## Key Hyperparameters (paper config)
| Param | Value |
| --- | --- |
| Model | Qwen2.5-7B |
| Data | numina_cot_100k (100K samples) |
| Mode | `asft` |
| KL weight | 0.05 |
| Learning rate | 5e-5 |
| Epochs | 1 |
| Global batch size | 256 |
| Max sequence length | 2048 |
| Precision | bf16 |

## Environment
- **Container**: `registry.rcp.epfl.ch/shuli/archer:asft`
- **Key deps**: transformers, trl (0.8.6–0.9.6), peft, vllm (0.7.3), deepspeed, torch 2.5.1

## Recommended Read Order
1. This file (`OVERVIEW.md`)
2. `diary/index.md` and `diary/progress.md` (experiment results)
3. `train_v2.py` (core training logic, especially `CustomSFTTrainer`)
4. `run_paper.sh` → `run_eval_paper.sh` (paper reproduction pipeline)
5. `eval/math_evaluation/math_eval.py` (evaluation harness)
