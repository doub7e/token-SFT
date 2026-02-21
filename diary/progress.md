# ASFT Reproduction Progress

## Overview

Reproducing [ASFT (Anchored Supervised Fine-Tuning)](https://arxiv.org/abs/2509.23753) — an ICLR 2026 paper that adds KL divergence regularization to supervised fine-tuning for LLM post-training.

**Repository**: Forked from [zhuchichi56/ASFT](https://github.com/zhuchichi56/ASFT)

---

## Experiment Results

| Date       | Config                         | Model          | Data   | MaxLen | Steps | MATH-500 | Notes                          |
|------------|--------------------------------|----------------|--------|--------|-------|----------|--------------------------------|
| 2026-02-20 | Quick run (LoRA r=8)           | Qwen2.5-1.5B   | 10K    | 512    | 200   | 39.8%    | Fast sanity check              |
| 2026-02-20 | Full 3-epoch (LoRA r=8)        | Qwen2.5-1.5B   | 10K    | 512    | 468   | 37.8%    | Overfitting with 3 epochs      |
| 2026-02-20 | Paper-aligned (LoRA r=32)      | Qwen2.5-7B     | 100K   | 2048   | 390   | 53.0%    | LoRA adaptation of paper config |
| 2026-02-22 | **sft-topk** (keep 70%)        | Qwen2.5-7B     | 100K   | 2048   | 390   | 56.0%    | Drop top 30% high-loss tokens  |
| 2026-02-22 | **sft-lr** (additive diff)     | Qwen2.5-7B     | 100K   | 2048   | 390   | 57.2%    | Weight by p_ref - p_θ          |
| 2026-02-22 | **sft-antiop** (clip 3×median) | Qwen2.5-7B     | 100K   | 2048   | 390   | **61.0%**| Soft-clip high-loss tokens; **beats paper full FT** |
| —          | Paper reported (Full FT)       | Qwen2.5-7B     | 100K   | 2048   | ~390  | 59.99%   | 8×GPU, ZeRO-3, full fine-tune  |

## Key Findings

1. **ASFT method works**: All configs show meaningful improvement over base model performance.
2. **LoRA vs Full FT gap**: Our LoRA r=32 result (53.0%) vs paper's full FT (59.99%) shows a ~7% gap, consistent with LoRA updating only ~0.9% of model parameters.
3. **Overfitting on small data**: 1.5B model with 3 epochs on 10K data overfits (37.8% < 39.8% at 200 steps).
4. **Full FT requires multi-GPU**: ASFT's reference model is not ZeRO-sharded, adding ~14GB/GPU overhead. Full FT of 7B OOMs on single H100 (80GB).
5. **Token-weighted SFT surpasses ASFT**: Per-token gradient trace analysis revealed ~25% opposing tokens (Gini ~0.51). Three new weighting schemes all beat ASFT baseline. Best result: **sft-antiop (61.0%)** surpasses paper's full FT (59.99%) with only LoRA r=32.
6. **Gradient trace insights**: Numbers/operators have lowest alignment (0.073-0.078) and highest opposing fraction (~25-28%). Magnitude-alignment Pearson r ≈ 0.34.

## Environment

- **Container image**: `registry.rcp.epfl.ch/shuli/archer:asft`
  - Base: `archer:latest` + trl, deepspeed, einops, etc.
- **Hardware**: RunAI cluster, H100 80GB GPUs
- **Key dependencies**: transformers, trl (0.8.6-0.9.6), peft, vllm (0.7.3), deepspeed, torch 2.5.1

## Run Scripts

| Script              | Description                                      |
|---------------------|--------------------------------------------------|
| `run_quick.sh`      | Quick 200-step training (1.5B, LoRA r=8, 10K)    |
| `run_full.sh`       | Full 3-epoch training (1.5B, LoRA r=8, 10K)      |
| `run_paper.sh`      | Paper-aligned config (7B, LoRA r=32, 100K)        |
| `run_eval.sh`       | Eval for quick-run checkpoint                     |
| `run_eval_full.sh`  | Eval for full-run checkpoint                      |
| `run_eval_paper.sh` | Eval for paper-aligned checkpoint                 |
| `run_topk.sh`       | Token-weighted training: sft-topk (keep 70%)      |
| `run_antiop.sh`     | Token-weighted training: sft-antiop (clip 3×med)   |
| `run_lr.sh`         | Token-weighted training: sft-lr (additive diff)    |
| `run_eval_topk.sh`  | Eval for sft-topk checkpoint                       |
| `run_eval_antiop.sh`| Eval for sft-antiop checkpoint                     |
| `run_eval_lr.sh`    | Eval for sft-lr checkpoint                         |
| `run_trace.sh`      | Per-token gradient trace (1.5B)                    |
| `run_trace_paper.sh`| Per-token gradient trace (7B paper config)         |

## Issues Encountered & Fixes

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `gradient_accumulation_steps` duplicate error | `train_v2.py` auto-computes from `global_batch_size` | Removed explicit arg |
| vLLM detects 4 GPUs, only 1 allocated | Node has multiple GPUs | `export CUDA_VISIBLE_DEVICES=0` |
| `merge_lora.py` import error | Imports from nonexistent `train_lora.py` | Inline Python merge in eval scripts |
| CUDA OOM on 4×H100 full FT | Reference model (+14GB/GPU) not ZeRO-sharded | Switched to LoRA r=32 |
| torch downgraded by trl | trl pulled torch 2.4.1 | Reinstall torch 2.5.1 in Dockerfile |

## Next Steps

- [ ] Try full fine-tuning on 8×H100 to match paper exactly
- [ ] Experiment with different KL weights
- [ ] Test on other benchmarks (GSM8K, GPQA)
- [ ] Design and run custom experiments on top of ASFT
