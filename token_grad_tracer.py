"""
Per-token gradient tracing with LoGRA two-sided random projection.

For a Linear layer, the per-token gradient is rank-1: nabla_W_t = dy_t outer x_t.
With LoGRA projection we store two small vectors per (layer, token):
  Xp_t in R^{k_i}  and  Yp_t in R^{k_o}
instead of the full outer product.

Two visualization approaches are supported downstream:
  1. Alignment: cosine similarity of each token's gradient with the full-sequence gradient
  2. Clustering: t-SNE/PCA of normalized per-token gradient directions

Adapted from code-for-reference/compute_per_sample_gradients_logra.py.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback


# ---------------------------------------------------------------------------
# Utility: deterministic hashing & Rademacher projection
# (from code-for-reference/compute_per_sample_gradients_logra.py)
# ---------------------------------------------------------------------------

def _stable_hash32(s: str) -> int:
    """Deterministic 32-bit hash (Python's hash is process-random)."""
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h & 0x7FFFFFFF)


def _make_rademacher_proj(
    in_features: int,
    out_dim: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Rademacher projection P in R^{out_dim x in_features} scaled by 1/out_dim."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    signs = torch.randint(
        0, 2, (out_dim, in_features), dtype=torch.int8, device=device, generator=g
    )
    P = signs.to(dtype).mul_(2).add_(-1).mul_(1.0 / float(max(out_dim, 1)))
    return P


# ---------------------------------------------------------------------------
# Hook manager: captures per-token projected activations & grad_outputs
# ---------------------------------------------------------------------------

DEFAULT_MODULE_FILTER = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class _PerTokenLoGRAHookManager:
    """Hook manager that stores per-token Xp and Yp separately (no bmm/sum over T).

    For each hooked Linear module:
      - Forward hook stores  Xp = x @ P_in.T  as [B, T, k_i]
      - Backward hook stores Yp = dy @ P_out.T as [B, T, k_o]
    """

    def __init__(self, proj_dim_factor: int):
        self.proj_dim_factor = int(proj_dim_factor)
        self.handles: List[torch.utils.hooks.RemovableHook] = []
        self.linear_modules: List[Tuple[str, nn.Linear]] = []
        self.module_dims: Dict[str, Tuple[int, int]] = {}  # name -> (k_i, k_o)
        # Per-module per-token results (populated each forward+backward)
        self.xp_results: Dict[str, torch.Tensor] = {}  # name -> [B, T, k_i]
        self.yp_results: Dict[str, torch.Tensor] = {}  # name -> [B, T, k_o]

    # -- projection init --------------------------------------------------

    def _ensure_projs(self, name: str, lin: nn.Linear):
        device = lin.weight.device
        dtype = torch.float32
        k_i = max(1, lin.in_features // self.proj_dim_factor)
        k_o = max(1, lin.out_features // self.proj_dim_factor)
        self.module_dims[name] = (k_i, k_o)
        if not hasattr(lin, "_lg_P_in"):
            seed_in = (_stable_hash32(name + ":P_in") ^ 0xA5A5A5A5) & 0x7FFFFFFF
            P_in = _make_rademacher_proj(lin.in_features, k_i, seed_in, device, dtype)
            object.__setattr__(lin, "_lg_k_i", k_i)
            lin.register_buffer("_lg_P_in", P_in, persistent=False)
        if not hasattr(lin, "_lg_P_out"):
            seed_out = (_stable_hash32(name + ":P_out") ^ 0x5A5A5A5A) & 0x7FFFFFFF
            P_out = _make_rademacher_proj(lin.out_features, k_o, seed_out, device, dtype)
            object.__setattr__(lin, "_lg_k_o", k_o)
            lin.register_buffer("_lg_P_out", P_out, persistent=False)

    # -- hooks -------------------------------------------------------------

    def _fwd_hook(self, name: str):
        def hook(module: nn.Module, inputs, output):
            with torch.no_grad():
                x = inputs[0]
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                B, T, I = x.shape
                k_i = cast(int, getattr(module, "_lg_k_i"))
                P_in = cast(torch.Tensor, getattr(module, "_lg_P_in"))
                Xp = torch.matmul(
                    x.reshape(B * T, I).to(torch.float32), P_in.t()
                ).view(B, T, k_i)
                module._lg_Xp = Xp
        return hook

    def _bwd_hook(self, name: str):
        """MODIFIED from reference: stores Yp per token, no bmm/sum."""
        def hook(module: nn.Module, grad_input, grad_output):
            with torch.no_grad():
                if not hasattr(module, "_lg_Xp"):
                    return
                dy = grad_output[0]
                Xp = cast(torch.Tensor, getattr(module, "_lg_Xp"))
                if dy.dim() == 2:
                    dy = dy.unsqueeze(1)
                B, T, O = dy.shape
                k_o = cast(int, getattr(module, "_lg_k_o"))
                P_out = cast(torch.Tensor, getattr(module, "_lg_P_out"))
                Yp = torch.matmul(
                    dy.reshape(B * T, O).to(torch.float32), P_out.t()
                ).view(B, T, k_o)

                # Store per-token Xp and Yp (move to CPU, cast to bfloat16)
                self.xp_results[name] = Xp.detach().cpu().to(torch.bfloat16)
                self.yp_results[name] = Yp.detach().cpu().to(torch.bfloat16)

                delattr(module, "_lg_Xp")
        return hook

    # -- registration ------------------------------------------------------

    def register(
        self,
        model: nn.Module,
        module_name_filter: Optional[List[str]] = None,
        lora_only: bool = False,
    ):
        if module_name_filter is None:
            module_name_filter = DEFAULT_MODULE_FILTER
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if not any(substr in name for substr in module_name_filter):
                continue
            # When lora_only=True, skip base layers — only keep lora_A / lora_B
            if lora_only:
                if "lora_" not in name:
                    continue
            self._ensure_projs(name, mod)
            self.linear_modules.append((name, mod))
        # Attach hooks
        for name, mod in self.linear_modules:
            self.handles.append(mod.register_forward_hook(self._fwd_hook(name)))
            self.handles.append(mod.register_full_backward_hook(self._bwd_hook(name)))
        lora_tag = " (LoRA layers only)" if lora_only else ""
        print(
            f"[TokenGradTrace] Registered hooks for {len(self.linear_modules)} modules "
            f"(proj_dim_factor={self.proj_dim_factor}){lora_tag}"
        )
        for name, mod in self.linear_modules:
            k_i, k_o = self.module_dims[name]
            print(f"  {name}: in={mod.in_features}->k_i={k_i}, out={mod.out_features}->k_o={k_o}")

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()

    def clear(self):
        self.xp_results.clear()
        self.yp_results.clear()

    def layer_names(self) -> List[str]:
        return [name for name, _ in self.linear_modules]


# ---------------------------------------------------------------------------
# Safe layer-name encoding for NPZ keys
# ---------------------------------------------------------------------------

def _safe_layer_name(name: str) -> str:
    return name.replace(".", "__")


def _unsafe_layer_name(safe: str) -> str:
    return safe.replace("__", ".")


# ---------------------------------------------------------------------------
# Trainer callback
# ---------------------------------------------------------------------------

class TokenGradTraceCallback(TrainerCallback):
    """HuggingFace TrainerCallback that captures per-token projected gradients.

    Every `trace_every_n_steps` steps, it:
      1. Picks one random sequence from the current batch
      2. Does a separate forward+backward with hooks
      3. Saves per-token Xp/Yp to an NPZ file
    """

    def __init__(
        self,
        trainer,
        trace_every_n_steps: int = 10,
        trace_proj_dim_factor: int = 256,
        trace_output_dir: Optional[str] = None,
        trace_module_filter: str = "q_proj,v_proj",
    ):
        self.trainer = trainer
        self.trace_every_n_steps = trace_every_n_steps
        self.proj_dim_factor = trace_proj_dim_factor
        self.output_dir = trace_output_dir or os.path.join(
            trainer.args.output_dir, "token_grad_trace"
        )
        if isinstance(trace_module_filter, (list, tuple)):
            self.module_filter = [s.strip() for s in trace_module_filter]
        else:
            self.module_filter = [s.strip() for s in trace_module_filter.split(",")]
        self.hook_mgr: Optional[_PerTokenLoGRAHookManager] = None
        self._steps_dir = os.path.join(self.output_dir, "steps")
        self._manifest_path = os.path.join(self.output_dir, "manifest.jsonl")
        os.makedirs(self._steps_dir, exist_ok=True)

    # -- callback methods --------------------------------------------------

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step == 0 or step % self.trace_every_n_steps != 0:
            return
        batch = getattr(self.trainer, "_last_batch", None)
        if batch is None:
            return
        model = kwargs.get("model", self.trainer.model)
        try:
            self._capture_and_save(model, batch, step)
        except Exception as e:
            print(f"[TokenGradTrace] Error at step {step}: {e}")
            import traceback
            traceback.print_exc()

    # -- core capture logic ------------------------------------------------

    @torch.no_grad()
    def _init_hook_mgr(self, model: nn.Module):
        """Lazy-init hook manager on first capture."""
        if self.hook_mgr is not None:
            return
        # Unwrap PEFT / DDP
        base = model
        if hasattr(base, "module"):
            base = base.module
        # Auto-detect LoRA: if model is PeftModel, only hook LoRA adapter layers
        is_lora = hasattr(base, "peft_config") or hasattr(base, "active_adapter")
        if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
            base = base.base_model.model
        self.hook_mgr = _PerTokenLoGRAHookManager(self.proj_dim_factor)
        self.hook_mgr.register(base, self.module_filter, lora_only=is_lora)
        self._base_model = base
        if is_lora:
            print("[TokenGradTrace] LoRA detected: hooking only lora_A/lora_B layers")

    def _capture_and_save(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        step: int,
    ):
        # Lazy init
        self._init_hook_mgr(model)
        hook_mgr = self.hook_mgr

        # Pick 1 random sequence
        B = batch["input_ids"].shape[0]
        idx = torch.randint(0, B, (1,)).item()
        input_ids = batch["input_ids"][idx : idx + 1]  # [1, T]
        labels = batch["labels"][idx : idx + 1]         # [1, T]
        attn_mask = batch.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask[idx : idx + 1]

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        # Clear previous results
        hook_mgr.clear()

        # Switch to eval, do separate forward+backward
        was_training = model.training
        model.eval()

        # Enable grad for this scope
        with torch.enable_grad():
            # Embed and make embeddings require grad to drive backprop
            base = self._base_model
            embed_layer = base.get_input_embeddings()
            inputs_embeds = embed_layer(input_ids)  # [1, T, D]
            inputs_embeds = inputs_embeds.detach().requires_grad_(True)

            fwd_kwargs = {"inputs_embeds": inputs_embeds, "use_cache": False}
            if attn_mask is not None:
                fwd_kwargs["attention_mask"] = attn_mask

            outputs = base(**fwd_kwargs)
            logits = outputs.logits  # [1, T, V]

            # Shifted CE loss with valid_mask (labels != -100)
            shift_logits = logits[:, :-1, :].contiguous()  # [1, T-1, V]
            shift_labels = labels[:, 1:].contiguous()       # [1, T-1]
            valid_mask = shift_labels != -100               # [1, T-1]

            if valid_mask.sum() == 0:
                print(f"[TokenGradTrace] Step {step}: no valid tokens, skipping.")
                if was_training:
                    model.train()
                return

            T_minus_1, V = shift_logits.shape[1], shift_logits.shape[2]
            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, V),
                shift_labels.clamp(min=0).view(-1),
                reduction="none",
            ).view(1, T_minus_1)

            loss = (per_token_loss * valid_mask.float()).sum()
            loss.backward()

        # Collect Xp/Yp per layer
        layer_names = hook_mgr.layer_names()
        save_dict: Dict[str, np.ndarray] = {}
        save_dict["step"] = np.int64(step)
        save_dict["input_ids"] = input_ids[0].cpu().numpy().astype(np.int32)
        save_dict["labels"] = labels[0].cpu().numpy().astype(np.int32)
        if attn_mask is not None:
            save_dict["attention_mask"] = attn_mask[0].cpu().numpy().astype(np.int8)
        else:
            save_dict["attention_mask"] = np.ones(input_ids.shape[1], dtype=np.int8)
        save_dict["valid_mask"] = valid_mask[0].cpu().numpy().astype(np.int8)
        save_dict["layer_names"] = np.array(layer_names, dtype=object)

        for name in layer_names:
            safe = _safe_layer_name(name)
            xp = hook_mgr.xp_results.get(name)
            yp = hook_mgr.yp_results.get(name)
            if xp is not None:
                # xp shape: [1, T, k_i] -> [T, k_i]; cast to float16 for numpy compat
                save_dict[f"Xp__{safe}"] = xp[0].float().numpy().astype(np.float16)
            if yp is not None:
                save_dict[f"Yp__{safe}"] = yp[0].float().numpy().astype(np.float16)

        # Save NPZ atomically
        # np.savez auto-appends .npz, so use a tmp basename without .npz
        filename = f"step_{step:06d}.npz"
        filepath = os.path.join(self._steps_dir, filename)
        tmp_base = os.path.join(self._steps_dir, f".tmp_{step:06d}_{uuid.uuid4().hex[:8]}")
        np.savez(tmp_base, **save_dict)
        # np.savez creates tmp_base + ".npz"
        os.replace(tmp_base + ".npz", filepath)

        # Append to manifest
        record = {
            "step": step,
            "file": f"steps/{filename}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_tokens": int(input_ids.shape[1]),
            "num_valid_tokens": int(valid_mask.sum().item()),
            "num_layers": len(layer_names),
        }
        with open(self._manifest_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"[TokenGradTrace] Step {step}: saved {len(layer_names)} layers, "
            f"T={input_ids.shape[1]}, valid={record['num_valid_tokens']} -> {filepath}"
        )

        # Cleanup
        model.zero_grad(set_to_none=True)
        hook_mgr.clear()
        if was_training:
            model.train()
