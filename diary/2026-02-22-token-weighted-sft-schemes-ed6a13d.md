# Token-Weighted SFT: topk, antiop, lr schemes

| Field   | Value |
|---------|-------|
| Date    | 2026-02-22 |
| Commit  | `ed6a13d` |
| Author  | Claude + shuli |
| Scope   | `train_v2.py`, run scripts, gradient trace tooling, visualizer |

## Summary

Designed and implemented three new token-level loss weighting schemes based on per-token gradient trace analysis insights. All three schemes beat the ASFT LoRA baseline on MATH-500, with the best (sft-antiop) surpassing the paper's full fine-tuning result.

## Motivation (from gradient trace analysis)

Per-token gradient alignment analysis on 1.5B and 7B ASFT training revealed:

| Insight | 1.5B | 7B |
|---------|------|-----|
| Gini coefficient | 0.515 | 0.503 |
| Tokens for 80% gradient weight | 42.5% | 43.9% |
| Fraction of opposing tokens | ~22% | ~25% |
| Number token alignment | 0.056 | 0.078 |
| Operator token alignment | 0.057 | 0.073 |
| Word token alignment | 0.070 | 0.152 |
| Magnitude-alignment Pearson r | 0.309 | 0.342 |

Key insight: Numbers and operators (which are harder to predict, thus higher CE loss) have the lowest gradient alignment and highest opposing fraction. High-loss tokens tend to produce gradients that fight the aggregate learning direction.

## Three Schemes Implemented

### Scheme A: Top-K Selective Loss (`sft-topk`)
- **Idea**: Drop highest-loss tokens entirely; keep only lowest K% by CE loss
- **Implementation**: Compute per-token CE, find K-th percentile threshold, mask out tokens above it
- **Hyperparameter**: `topk_keep_ratio=0.7` (keep 70%, drop worst 30%)
- **No KL, no reference model**

### Scheme D: Anti-Opposing Soft Clipping (`sft-antiop`)
- **Idea**: Instead of hard masking, soft-clip contribution of high-loss tokens
- **Implementation**: `w_t = 1.0` if loss <= c * median, else `w_t = cap / loss` (decays smoothly)
- **Hyperparameter**: `antiop_clip_mult=3.0` (clip at 3x median loss)
- **No KL, no reference model**

### Scheme C: Likelihood Ratio Weighting (`sft-lr`)
- **Idea**: Upweight tokens where model got worse vs reference (p_ref > p_theta)
- **Implementation**: `w_t = 1 + alpha * (p_ref - p_t)`, clamped to [0.1, +inf]
- **Hyperparameters**: `lr_weight_alpha=1.0`
- **Uses reference model (via LoRA disable_adapter) but NO KL term**

## Results

| Method | MATH-500 Acc | Delta vs ASFT baseline |
|--------|-------------|------------------------|
| ASFT baseline (LoRA r=32) | 53.0% | — |
| sft-topk (keep 70%) | 56.0% | +3.0 |
| sft-lr (additive diff) | 57.2% | +4.2 |
| **sft-antiop** (clip 3×median) | **61.0%** | **+8.0** |
| Paper full FT (8×GPU) | 59.99% | +6.99 |

All methods trained on Qwen2.5-7B, LoRA r=32, 100K data, 1 epoch, single H100.

## Files Changed

- `train_v2.py`: Added 3 new mode branches in `compute_loss()`, new CLI args, trace callback integration
- `run_topk.sh`, `run_antiop.sh`, `run_lr.sh`: Training scripts
- `run_eval_topk.sh`, `run_eval_antiop.sh`, `run_eval_lr.sh`: Eval scripts
- `token_grad_tracer.py`: Per-token gradient trace callback
- `analyze_token_gradients.py`: Offline trace analysis (14 metrics)
- `compare_traces.py`: Cross-model comparison tool
- `visualize_token_gradients.py`: Gradio visualizer with dot-product metric
- `OVERVIEW.md`: Project overview and entry points

## Debugging Notes

- sft-lr initially showed loss ~60 (vs expected ~3). Root cause: HuggingFace Trainer loss normalization with `gradient_accumulation_steps=128` scales logged loss. Per-micro-batch loss was correct (~0.5). Not a bug.
- Multiplicative ratio `p_ref/p_t` caused positive correlation with CE (high-loss tokens also get high weight), defeating the purpose. Switched to additive difference.
