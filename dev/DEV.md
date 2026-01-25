# ASFT Dev Notes


## 🧭 Tested Schemes & Insights

We tested baseline DDP, DeepSpeed Zero-2, and Zero-3 with bf16/fp16, plus optimizer alignment and KL-weight sweeps. Insights:


- **bf16 is always more stable than fp16**; fp16 requires careful loss scale settings and still lags behind in accuracy.
- **DeepSpeed Zero (Zero-2/Zero-3)** is less stable in practice; **native (non-DeepSpeed) runs** are the most stable overall.
- **KL weight ≈ 0.03** consistently yields the highest medical accuracy with bf16; larger KL weights hurt stability.
- **Optimizer alignment (AdamW)** improves results for both Zero-2 and Zero-3 compared to default optimizer settings.

---

## 📊 Experiment Tables

### ASFT Med (DeepSpeed / Zero)

| Exp ID | Model | Key Params | medqa_acc | mmlu_acc | medmcqa_acc | Status | Notes |
|--------|-------|------------|-----------|----------|-------------|--------|-------|
| b1-bf16 | Llama-2-7b-hf | asft,bf16,3ep,8gpu | 0.3833 | 0.4560 | 0.3992 | ✅ | baseline | 
| z2-bf16-fail | Llama-2-7b-hf | asft,zero2,bf16,3ep,8gpu | - | - | - | ❌ | original_model 设备不一致 | 
| z2-bf16 | Llama-2-7b-hf | asft,zero2,bf16,3ep,8gpu | 0.3284 | 0.3786 | 0.3545 | ✅ | zero2 bf16 | 
| z2-fp16-fail | Llama-2-7b-hf | asft,zero2,fp16,3ep,8gpu | - | - | - | ❌ | loss scale underflow | 
| z2-fp16 | Llama-2-7b-hf | asft,zero2,fp16,3ep,8gpu,loss_scale=1 | 0.2922 | 0.2793 | 0.3383 | ✅ | zero2 fp16 (fixed loss scale) | 
| z3-bf16 | Llama-2-7b-hf | asft,zero3,bf16,3ep,8gpu | 0.3252 | 0.3768 | 0.3627 | ✅ | zero3 bf16 | 
| z3-fp16 | Llama-2-7b-hf | asft,zero3,fp16,3ep,8gpu,loss_scale=1 | 0.2922 | 0.2793 | 0.3383 | ✅ | zero3 fp16 | 
| z2-bf16-opt | Llama-2-7b-hf | asft,zero2,bf16,3ep,8gpu,opt=adamw | 0.3535 | 0.4093 | 0.3768 | ✅ | zero2 bf16 optimizer aligned | 
| z3-bf16-opt | Llama-2-7b-hf | asft,zero3,bf16,3ep,8gpu,opt=adamw | 0.3339 | 0.4122 | 0.3698 | ✅ | zero3 bf16 optimizer aligned | 
| z2-bf16-opt-kl007 | Llama-2-7b-hf | asft,zero2,bf16,3ep,8gpu,kl=0.07 | 0.3417 | 0.4074 | 0.3787 | ✅ | zero2 bf16 kl=0.07 | 
| z2-bf16-opt-kl003 | Llama-2-7b-hf | asft,zero2,bf16,3ep,8gpu,kl=0.03 | 0.3920 | 0.4575 | 0.3952 | ✅ | zero2 bf16 kl=0.03 (best) | 
| z3-bf16-opt-kl003 | Llama-2-7b-hf | asft,zero3,bf16,3ep,8gpu,kl=0.03 | 0.3668 | 0.4472 | 0.3854 | ✅ | zero3 bf16 kl=0.03 | 
| z3-bf16-opt-kl0025 | Llama-2-7b-hf | asft,zero3,bf16,3ep,8gpu,kl=0.025 | 0.3401 | 0.3640 | 0.3648 | ✅ | zero3 bf16 kl=0.025 | 

### Math Results Summary

| Mode | Engine | LR | BS | Seq | math_oai | minerva_math | olympiadbench | aime24 | amc23 |
|---|---|---|---|---|---|---|---|---|---|
| sft | baseline | 5e-5 | 256 | 2048 | 62.0 | 18.0 | 27.4 | 10.0 | 40.0 |
| sft | baseline | 2e-5 | 256 | 2048 | 70.0 | 22.1 | 29.0 | 10.0 | 45.0 |
| sft | zero2 | 2e-5 | 256 | 2048 | 62.6 | 19.9 | 27.9 | 10.0 | 32.5 |
| sft | zero2 | 5e-5 | 256 | 2048 | 58.0 | 21.7 | 24.7 | 0.0 | 30.0 |
| asft | baseline | 2e-5 | 256 | 2048 | 69.0 | 23.2 | 30.4 | 10.0 | 50.0 |
| asft | baseline | 5e-5 | 256 | 2048 | 63.4 | 18.4 | 28.6 | 0.0 | 30.0 |
| asft | zero2 | 2e-5 | 256 | 2048 | 64.0 | 22.8 | 31.0 | 10.0 | 40.0 |
| asft | zero2 | 5e-5 | 256 | 2048 | 60.0 | 20.6 | 25.9 | 6.7 | 32.5 |
| dft | baseline | 2e-5 | 256 | 2048 | 70.4 | 24.3 | 30.4 | 3.3 | 45.0 |
| dft | baseline | 5e-5 | 256 | 2048 | 61.4 | 26.1 | 26.4 | 3.3 | 40.0 |
| dft | zero2 | 2e-5 | 256 | 2048 | 64.0 | 23.9 | 26.2 | 10.0 | 25.0 |
| dft | zero2 | 5e-5 | 256 | 2048 | 49.6 | 18.0 | 20.1 | 6.7 | 17.5 |

### Math Results Summary (8-GPU eval, n_sampling=16, temperature=1, top_p=1)

| Mode | Engine | LR | BS | Seq | math_oai | minerva_math | olympiadbench | aime24 | amc23 |
|---|---|---|---|---|---|---|---|---|---|
| asft | baseline | 2e-5 | 1 | 2048 | 63.8 | 20.6 | 28.9 | 10.0 | 47.5 |
| asft | baseline | 5e-5 | 1 | 2048 | 59.2 | 18.0 | 24.6 | 6.7 | 25.0 |
| asft | zero2 | 2e-5 | 1 | 2048 | 61.6 | 19.1 | 26.1 | 6.7 | 30.0 |
| asft | zero2 | 5e-5 | 1 | 2048 | 54.4 | 17.6 | 21.0 | 3.3 | 25.0 |
| dft | baseline | 2e-5 | 1 | 2048 | 67.2 | 20.2 | 29.9 | 3.3 | 37.5 |
| dft | baseline | 5e-5 | 1 | 2048 | 60.0 | 22.8 | 26.2 | 6.7 | 25.0 |
| dft | zero2 | 2e-5 | 1 | 2048 | 63.4 | 24.3 | 27.0 | 3.3 | 32.5 |
| dft | zero2 | 5e-5 | 1 | 2048 | 50.2 | 18.0 | 18.7 | 3.3 | 15.0 |
| sft | baseline | 2e-5 | 2 | 2048 | 46.8 | 7.4 | 15.1 | 3.3 | 25.0 |
| sft | baseline | 5e-5 | 2 | 2048 | 49.4 | 10.3 | 18.4 | 6.7 | 30.0 |
| sft | zero2 | 2e-5 | 2 | 2048 | 56.0 | 10.3 | 20.1 | 3.3 | 17.5 |
| sft | zero2 | 5e-5 | 2 | 2048 | 55.2 | 13.2 | 19.4 | 3.3 | 17.5 |
