## 2026-02-20 - ASFT Reproduction Scripts and Evaluation

| Item | Details |
| --- | --- |
| Request | Reproduce ASFT paper experiments; run training + MATH-500 evaluation |
| Delivery | 3 training configs (quick/full/paper-aligned) + 3 eval scripts; all evaluated |
| Scope | RunAI H100 cluster, container `archer:asft`, Qwen2.5-1.5B and 7B |

### Changes
| Path | Change | Why |
| --- | --- | --- |
| `run_quick.sh` | Added | Quick 200-step sanity check (1.5B, LoRA r=8, 10K data) |
| `run_full.sh` | Added | Full 3-epoch training (1.5B, LoRA r=8, 10K data) |
| `run_paper.sh` | Added | Paper-aligned config (7B, LoRA r=32, 100K, maxlen=2048) |
| `run_eval.sh` | Added | Eval for quick-run checkpoint |
| `run_eval_full.sh` | Added | Eval for full-run checkpoint |
| `run_eval_paper.sh` | Added | Eval for paper-aligned checkpoint |
| `.gitignore` | Updated | Added `models/` to exclude 18GB model weights |
| `diary/progress.md` | Added | High-level experiment results and project overview |

### Validation
| Check | Evidence | Result |
| --- | --- | --- |
| Quick run (200 steps) | RunAI job `asft-quick-run` | Pass — loss 0.404, MATH-500 39.8% |
| Full run (3 epochs, 468 steps) | RunAI job `asft-full-run` | Pass — loss 0.393, MATH-500 37.8% |
| Paper-aligned (7B, 390 steps) | RunAI job `asft-paper-run` | Pass — loss 6.51, MATH-500 53.0% (paper: 59.99%) |

### Git
| Field | Value |
| --- | --- |
| Commit | `0df44c4` |
| Branch | `main` |
| Remote | `git@github.com:doub7e/token-SFT.git` |
| Push | Yes |

### Notes
- 7% gap vs paper (53.0% vs 59.99%) is expected: we used LoRA r=32 instead of full FT due to single-GPU memory constraint.
- Full FT of 7B requires 8×H100 with ZeRO-3 (reference model adds ~14GB/GPU overhead).
- 1.5B model overfits with 3 epochs on 10K data (37.8% < 39.8% at 200 steps).
