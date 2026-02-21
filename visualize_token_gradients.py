#!/usr/bin/env python3
"""Interactive Gradio visualizer for per-token gradient traces.

Two tabs:
  1. Alignment View  — cosine similarity of each token's gradient with the full-sequence gradient
  2. Clustering View  — t-SNE / PCA of normalized per-token gradient directions

Efficient: no outer products are ever materialized.
"""

from __future__ import annotations

import argparse
import html
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Token classification
# ---------------------------------------------------------------------------

def _classify_token(text: str) -> str:
    """Classify a token into a semantic category for visualization."""
    text = text.strip()
    if not text:
        return "whitespace"
    if text in (".", ",", ";", ":", "!", "?", "'", '"', "(", ")", "[", "]", "{", "}"):
        return "punctuation"
    if text in ("\\", "\\\\", "$$", "$"):
        return "latex/math"
    if text.isdigit() or (len(text) > 0 and all(c.isdigit() or c == "." for c in text)):
        return "number"
    if text in ("+", "-", "*", "/", "=", "<", ">", "^", "_", "|", "&"):
        return "operator"
    if text in ("\n", "\t", "\r"):
        return "newline/tab"
    if text.isalpha():
        return "word"
    if any(c.isalpha() for c in text):
        return "subword"
    return "other"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-token gradient alignment & clustering visualizer.")
    p.add_argument("--trace-dir", type=str, default="output/token_grad_trace",
                   help="Directory with manifest.jsonl and steps/*.npz")
    p.add_argument("--model-path", type=str, default="./models/Qwen2.5-1.5B",
                   help="Tokenizer/model path for token decoding")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7863)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_trace_manifest(trace_dir: str) -> List[Dict[str, Any]]:
    """Read manifest.jsonl or fall back to scanning steps/ directory."""
    trace_dir = Path(trace_dir).expanduser().resolve()
    manifest = trace_dir / "manifest.jsonl"
    records: List[Dict[str, Any]] = []
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        steps_dir = trace_dir / "steps"
        if steps_dir.exists():
            for npz in sorted(steps_dir.glob("step_*.npz")):
                step = int(npz.stem.split("_")[-1])
                records.append({"step": step, "file": f"steps/{npz.name}"})
    records.sort(key=lambda x: int(x["step"]))
    return records


@lru_cache(maxsize=128)
def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def load_step_data(trace_dir: str, record: Dict) -> Dict[str, Any]:
    """Load and structure NPZ data for a single step."""
    path = str(Path(trace_dir).expanduser().resolve() / record["file"])
    raw = _load_npz(path)

    layer_names = raw["layer_names"].tolist() if "layer_names" in raw else []
    result = {
        "step": int(raw["step"]),
        "input_ids": raw["input_ids"].astype(np.int32),
        "labels": raw["labels"].astype(np.int32),
        "attention_mask": raw["attention_mask"].astype(np.int8),
        "valid_mask": raw["valid_mask"].astype(np.int8),
        "layer_names": layer_names,
        "Xp": {},  # layer_name -> [T, k_i]
        "Yp": {},  # layer_name -> [T, k_o]
    }
    for name in layer_names:
        safe = name.replace(".", "__")
        xp_key = f"Xp__{safe}"
        yp_key = f"Yp__{safe}"
        if xp_key in raw:
            result["Xp"][name] = raw[xp_key].astype(np.float32)
        if yp_key in raw:
            result["Yp"][name] = raw[yp_key].astype(np.float32)
    return result


@lru_cache(maxsize=2)
def _load_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _decode_token(tokenizer, tid: int) -> str:
    if tokenizer is None:
        return str(tid)
    text = tokenizer.decode([int(tid)])
    return text if text else str(tid)


# ---------------------------------------------------------------------------
# Alignment computation (no outer products)
# ---------------------------------------------------------------------------

def compute_alignment(
    data: Dict[str, Any],
    layer_name: Optional[str] = None,
    aggregation: str = "holistic",
    metric: str = "cosine",
) -> np.ndarray:
    """Compute per-token alignment with the full-sequence gradient.

    Args:
        metric: "cosine" for cosine similarity (direction only, in [-1, 1]),
                "dot" for dot product G . g_t (magnitude-weighted contribution).

    Returns alignment array of shape [T-1] (shifted, matching valid_mask).

    For a single layer:
      G = sum_s (Yp_s outer Xp_s)   (full-sequence gradient)
      g_t = Yp_t outer Xp_t         (token gradient)

      G . g_t = sum_s (Yp_s . Yp_t)(Xp_s . Xp_t)
      ||g_t||^2 = ||Yp_t||^2 * ||Xp_t||^2
      ||G||^2 = sum_{s,s'} (Yp_s . Yp_{s'})(Xp_s . Xp_{s'})

    For holistic multi-layer: sum numerators/denominators across layers.
    For mean: average per-layer values.
    """
    valid_mask = data["valid_mask"].astype(bool)  # [T-1]
    T_minus_1 = len(valid_mask)

    layers = [layer_name] if layer_name else data["layer_names"]
    layers = [l for l in layers if l in data["Xp"] and l in data["Yp"]]

    if not layers:
        return np.zeros(T_minus_1, dtype=np.float32)

    if aggregation == "holistic":
        # Sum numerators and denominators across layers
        total_dot_with_G = np.zeros(T_minus_1, dtype=np.float64)
        total_g_norm_sq = np.zeros(T_minus_1, dtype=np.float64)
        total_G_norm_sq = 0.0

        for name in layers:
            Xp = data["Xp"][name]  # [T, k_i]
            Yp = data["Yp"][name]  # [T, k_o]

            # Use shifted tokens: positions 0..T-2 predict labels 1..T-1
            Xp_s = Xp[:T_minus_1].astype(np.float64)  # [T-1, k_i]
            Yp_s = Yp[:T_minus_1].astype(np.float64)  # [T-1, k_o]

            # Mask: only sum over valid positions
            mask_f = valid_mask.astype(np.float64)  # [T-1]

            # X_dots[s,t] = Xp_s[s] . Xp_s[t]
            X_dots = Xp_s @ Xp_s.T  # [T-1, T-1]
            Y_dots = Yp_s @ Yp_s.T  # [T-1, T-1]

            # Cross-product matrix
            cross = X_dots * Y_dots  # [T-1, T-1]

            # Mask rows for valid source tokens
            cross_masked = cross * mask_f[:, None]  # zero out invalid source rows

            # dot_with_G[t] = sum_s (valid) cross[s, t]
            dot_with_G = cross_masked.sum(axis=0)  # [T-1]
            total_dot_with_G += dot_with_G

            # ||g_t||^2 = ||Yp_t||^2 * ||Xp_t||^2
            g_norm_sq = (np.linalg.norm(Yp_s, axis=1) ** 2) * (np.linalg.norm(Xp_s, axis=1) ** 2)
            total_g_norm_sq += g_norm_sq

            # ||G||^2 = sum_{s,s'} cross_masked[s,s'] (both valid)
            cross_both_valid = cross * mask_f[:, None] * mask_f[None, :]
            total_G_norm_sq += cross_both_valid.sum()

        if metric == "dot":
            # Raw dot product G . g_t — measures magnitude-weighted contribution
            return total_dot_with_G.astype(np.float32)
        else:
            # Cosine similarity — direction only
            G_norm = np.sqrt(max(total_G_norm_sq, 1e-30))
            g_norm = np.sqrt(np.maximum(total_g_norm_sq, 1e-30))
            alignment = total_dot_with_G / (G_norm * g_norm + 1e-30)
            alignment = np.clip(alignment, -1.0, 1.0).astype(np.float32)
            return alignment

    else:  # mean per-layer
        per_layer = []
        for name in layers:
            val = compute_alignment(data, layer_name=name, aggregation="holistic",
                                    metric=metric)
            per_layer.append(val)
        return np.mean(per_layer, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Clustering (t-SNE / PCA)
# ---------------------------------------------------------------------------

def compute_token_features(
    data: Dict[str, Any],
    layer_name: Optional[str] = None,
) -> np.ndarray:
    """Build per-token feature vector for clustering.

    Feature = concat(Yp_t / ||Yp_t||, Xp_t / ||Xp_t||) per layer, concat across layers.
    Returns [T-1, D] feature matrix.
    """
    valid_mask = data["valid_mask"].astype(bool)
    T_minus_1 = len(valid_mask)

    layers = [layer_name] if layer_name else data["layer_names"]
    layers = [l for l in layers if l in data["Xp"] and l in data["Yp"]]

    feats = []
    for name in layers:
        Xp = data["Xp"][name][:T_minus_1].astype(np.float32)
        Yp = data["Yp"][name][:T_minus_1].astype(np.float32)

        xp_norm = np.linalg.norm(Xp, axis=1, keepdims=True)
        yp_norm = np.linalg.norm(Yp, axis=1, keepdims=True)
        Xp_n = Xp / np.maximum(xp_norm, 1e-8)
        Yp_n = Yp / np.maximum(yp_norm, 1e-8)
        feats.append(np.concatenate([Yp_n, Xp_n], axis=1))

    if not feats:
        return np.zeros((T_minus_1, 2), dtype=np.float32)
    return np.concatenate(feats, axis=1)


def run_dim_reduction(
    features: np.ndarray,
    valid_mask: np.ndarray,
    method: str = "pca",
) -> np.ndarray:
    """Reduce features to 2D via PCA or t-SNE. Returns [N, 2] for valid tokens."""
    from sklearn.decomposition import PCA

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 2:
        return np.zeros((len(valid_idx), 2), dtype=np.float32)

    X = features[valid_idx]

    if method == "tsne":
        from sklearn.manifold import TSNE
        n_components_pca = min(50, X.shape[1], X.shape[0])
        if X.shape[1] > n_components_pca:
            X = PCA(n_components=n_components_pca).fit_transform(X)
        perplexity = min(30, max(5, len(valid_idx) // 4))
        coords = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(X)
    else:
        coords = PCA(n_components=2).fit_transform(X)

    return coords.astype(np.float32)


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _alignment_to_rgb(val: float, vmin: float = -1.0, vmax: float = 1.0) -> str:
    """Diverging: blue (vmin) -> white (0) -> red (vmax)."""
    if not np.isfinite(val):
        return "rgb(140,140,140)"
    # Normalize to [-1, 1] using the symmetric scale max(|vmin|, |vmax|)
    scale = max(abs(vmin), abs(vmax), 1e-30)
    normed = float(np.clip(val / scale, -1, 1))
    t = abs(normed)
    if normed >= 0:
        r, g, b = 255, int(255 * (1 - t)), int(255 * (1 - t))
    else:
        r, g, b = int(255 * (1 - t)), int(255 * (1 - t)), 255
    return f"rgb({r},{g},{b})"


def _render_alignment_heatmap(
    tokens: List[str],
    alignment: np.ndarray,
    valid_mask: np.ndarray,
    labels: np.ndarray,
    input_ids: np.ndarray,
    metric: str = "cosine",
) -> str:
    """Render tokens as colored spans. Gray for instruction tokens (label==-100)."""
    # Determine color scale: cosine is always [-1,1], dot uses data range
    if metric == "cosine":
        vmin, vmax = -1.0, 1.0
    else:
        valid_vals = alignment[valid_mask.astype(bool)]
        if valid_vals.size > 0:
            abs_max = max(abs(valid_vals.min()), abs(valid_vals.max()), 1e-30)
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = -1.0, 1.0

    color_threshold = 0.4 if metric == "cosine" else 0.4 * max(abs(vmin), abs(vmax))

    chunks: List[str] = []
    T_minus_1 = len(valid_mask)
    for t in range(T_minus_1):
        tok = html.escape(tokens[t] if t < len(tokens) else "?")
        val = float(alignment[t])
        is_valid = bool(valid_mask[t])
        is_instruction = bool(labels[t + 1] == -100) if t + 1 < len(labels) else True

        if is_instruction:
            bg = "rgb(200,200,200)"
            fg = "rgb(100,100,100)"
        elif is_valid:
            bg = _alignment_to_rgb(val, vmin, vmax)
            fg = "#333" if abs(val) < color_threshold else "white"
        else:
            bg = "rgb(180,180,180)"
            fg = "rgb(100,100,100)"

        tooltip = html.escape(
            f"pos={t} id={int(input_ids[t])} alignment={val:.5f} valid={is_valid} label={int(labels[t+1]) if t+1 < len(labels) else -100}"
        )
        chunks.append(
            f"<span title='{tooltip}' style='display:inline-block;margin:1px;padding:3px 5px;"
            f"border-radius:4px;background:{bg};color:{fg};font-family:monospace;font-size:12px'>"
            f"{tok}</span>"
        )
    return "<div style='line-height:1.8'>" + "".join(chunks) + "</div>"


# ---------------------------------------------------------------------------
# Gradio App
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- shared helpers ----

    def _load(trace_dir_str: str, model_path: str):
        trace_dir = Path(trace_dir_str).expanduser().resolve()
        if not trace_dir.exists():
            return f"Trace dir not found: `{trace_dir}`", [], gr.Dropdown(choices=[], value=None), pd.DataFrame()
        records = load_trace_manifest(str(trace_dir))
        if not records:
            return f"No trace records in `{trace_dir}`.", [], gr.Dropdown(choices=[], value=None), pd.DataFrame()
        try:
            _ = _load_tokenizer(model_path)
            tok_info = f"Tokenizer loaded from `{model_path}`."
        except Exception as e:
            tok_info = f"Tokenizer failed: {e}"
        steps = [int(r["step"]) for r in records]
        df = pd.DataFrame(records)
        return f"Loaded {len(records)} steps from `{trace_dir}`. {tok_info}", records, gr.Dropdown(choices=steps, value=steps[0]), df

    # ==== Tab 1: Alignment ====

    def alignment_on_step(records, trace_dir_str, model_path, step, layer_choice, aggregation, metric):
        if not records or step is None:
            return "<div>Load data first.</div>", "", pd.DataFrame()

        rec = next((r for r in records if int(r["step"]) == int(step)), None)
        if rec is None:
            return f"<div>Step {step} not found.</div>", "", pd.DataFrame()

        data = load_step_data(trace_dir_str, rec)

        layer = None if layer_choice == "(all layers)" else layer_choice

        alignment = compute_alignment(data, layer_name=layer, aggregation=aggregation,
                                      metric=metric)
        valid_mask = data["valid_mask"].astype(bool)

        tokenizer = None
        try:
            tokenizer = _load_tokenizer(model_path)
        except Exception:
            pass

        tokens = [_decode_token(tokenizer, int(tid)) for tid in data["input_ids"]]

        heatmap_html = _render_alignment_heatmap(
            tokens, alignment, valid_mask, data["labels"], data["input_ids"],
            metric=metric,
        )

        # Summary stats
        valid_align = alignment[valid_mask]
        if valid_align.size > 0:
            metric_label = "Cosine Similarity" if metric == "cosine" else "Dot Product (G · gₜ)"
            stats = (
                f"**Step {step}** | Layer: {layer_choice} | Aggregation: {aggregation} | Metric: {metric_label}  \n"
                f"Valid tokens: {valid_mask.sum()} / {len(valid_mask)}  \n"
                f"Alignment: mean={valid_align.mean():.4f}, std={valid_align.std():.4f}, "
                f"min={valid_align.min():.4f}, max={valid_align.max():.4f}  \n"
                f"Positive (aligned): {(valid_align > 0).sum()} | Negative (opposing): {(valid_align < 0).sum()}"
            )
        else:
            stats = f"**Step {step}** — no valid tokens."

        # Token detail table
        rows = []
        T_minus_1 = len(valid_mask)
        for t in range(T_minus_1):
            is_valid = bool(valid_mask[t])
            label_val = int(data["labels"][t + 1]) if t + 1 < len(data["labels"]) else -100
            rows.append({
                "position": t,
                "token": tokens[t] if t < len(tokens) else "?",
                "token_id": int(data["input_ids"][t]),
                "alignment": float(alignment[t]),
                "valid": is_valid,
                "label": label_val,
                "is_response": label_val != -100,
            })
        df = pd.DataFrame(rows)

        return heatmap_html, stats, df

    def alignment_get_layers(records, trace_dir_str, step):
        """Return available layer names for the selected step."""
        if not records or step is None:
            return gr.Dropdown(choices=["(all layers)"], value="(all layers)")
        rec = next((r for r in records if int(r["step"]) == int(step)), None)
        if rec is None:
            return gr.Dropdown(choices=["(all layers)"], value="(all layers)")
        data = load_step_data(trace_dir_str, rec)
        choices = ["(all layers)"] + data["layer_names"]
        return gr.Dropdown(choices=choices, value="(all layers)")

    # ==== Tab 2: Clustering ====

    def clustering_view(records, trace_dir_str, model_path, steps_selected, layer_choice, method, color_by):
        if not records or not steps_selected:
            return go.Figure(), ""

        tokenizer = None
        try:
            tokenizer = _load_tokenizer(model_path)
        except Exception:
            pass

        all_coords = []
        all_meta = []

        for step_val in steps_selected:
            rec = next((r for r in records if int(r["step"]) == int(step_val)), None)
            if rec is None:
                continue
            data = load_step_data(trace_dir_str, rec)
            layer = None if layer_choice == "(all layers)" else layer_choice
            features = compute_token_features(data, layer_name=layer)
            valid_mask = data["valid_mask"].astype(bool)

            coords = run_dim_reduction(features, valid_mask, method=method)

            valid_idx = np.where(valid_mask)[0]
            for i, t in enumerate(valid_idx):
                tid = int(data["input_ids"][t])
                tok = _decode_token(tokenizer, tid)
                label_val = int(data["labels"][t + 1]) if t + 1 < len(data["labels"]) else -100
                is_response = label_val != -100

                Xp_norms = []
                Yp_norms = []
                layers_used = [layer] if layer else data["layer_names"]
                layers_used = [l for l in layers_used if l in data["Xp"] and l in data["Yp"]]
                for ln in layers_used:
                    if t < data["Xp"][ln].shape[0]:
                        Xp_norms.append(float(np.linalg.norm(data["Xp"][ln][t])))
                        Yp_norms.append(float(np.linalg.norm(data["Yp"][ln][t])))
                grad_mag = sum(x * y for x, y in zip(Xp_norms, Yp_norms))

                all_coords.append(coords[i])
                all_meta.append({
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                    "step": int(step_val),
                    "position": int(t),
                    "token": tok,
                    "token_id": tid,
                    "role": "response" if is_response else "instruction",
                    "type": _classify_token(tok),
                    "grad_magnitude": grad_mag,
                })

        if not all_meta:
            return go.Figure(), "No valid tokens to cluster."

        df = pd.DataFrame(all_meta)

        color_col = color_by
        if color_by == "position":
            color_col = "position"
        elif color_by == "type":
            color_col = "type"
        elif color_by == "grad_magnitude":
            color_col = "grad_magnitude"
        elif color_by == "step":
            df["step_str"] = df["step"].astype(str)
            color_col = "step_str"

        axis_label = "t-SNE" if method == "tsne" else "PCA"

        fig = px.scatter(
            df, x="x", y="y",
            color=color_col,
            hover_data=["token", "token_id", "position", "step", "role", "type", "grad_magnitude"],
            title=f"Per-Token Gradient Clustering ({axis_label})",
            labels={"x": f"{axis_label}-1", "y": f"{axis_label}-2"},
            color_continuous_scale="Viridis" if color_col in ("position", "grad_magnitude") else None,
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(height=600)

        stats = f"Showing {len(df)} tokens from {df['step'].nunique()} step(s). Method: {method}."
        return fig, stats

    # ==== Build Gradio UI ====

    with gr.Blocks(title="Per-Token Gradient Visualizer") as demo:
        gr.Markdown("# Per-Token Gradient Alignment & Clustering Visualizer")

        with gr.Row():
            trace_dir_tb = gr.Textbox(label="Trace Directory", value=args.trace_dir, scale=3)
            model_path_tb = gr.Textbox(label="Tokenizer Path", value=args.model_path, scale=3)
            load_btn = gr.Button("Load", variant="primary", scale=1)

        load_status = gr.Markdown()
        records_state = gr.State([])
        manifest_df = gr.Dataframe(label="Trace Manifest", wrap=True, interactive=False)

        with gr.Tabs():
            # ---- Tab 1: Alignment ----
            with gr.Tab("Alignment View"):
                with gr.Row():
                    align_step_dd = gr.Dropdown(label="Step", choices=[], value=None, scale=2)
                    align_layer_dd = gr.Dropdown(label="Layer", choices=["(all layers)"], value="(all layers)", scale=3)
                    align_agg_radio = gr.Radio(
                        label="Multi-Layer Aggregation",
                        choices=["holistic", "mean"],
                        value="holistic",
                        scale=2,
                    )
                    align_metric_radio = gr.Radio(
                        label="Metric",
                        choices=["cosine", "dot"],
                        value="cosine",
                        scale=2,
                    )

                align_stats_md = gr.Markdown()
                align_heatmap_html = gr.HTML(label="Token Alignment Heatmap")
                align_detail_df = gr.Dataframe(label="Token Details (sortable)", wrap=True, interactive=False)

            # ---- Tab 2: Clustering ----
            with gr.Tab("Clustering View"):
                with gr.Row():
                    cluster_steps_cb = gr.CheckboxGroup(label="Steps", choices=[], value=[])
                    cluster_layer_dd = gr.Dropdown(label="Layer", choices=["(all layers)"], value="(all layers)", scale=2)

                with gr.Row():
                    cluster_method_radio = gr.Radio(label="Method", choices=["pca", "tsne"], value="pca")
                    cluster_color_radio = gr.Radio(
                        label="Color By",
                        choices=["position", "type", "grad_magnitude", "step"],
                        value="position",
                    )

                cluster_plot = gr.Plot(label="Scatter")
                cluster_stats_md = gr.Markdown()

        # ---- Callbacks ----

        def on_load(trace_dir_str, model_path):
            status, records, step_dd, mdf = _load(trace_dir_str, model_path)
            steps = [int(r["step"]) for r in records]
            return (
                status, records, step_dd, mdf,
                gr.Dropdown(choices=steps, value=steps[0] if steps else None),  # align_step_dd
                gr.CheckboxGroup(choices=[str(s) for s in steps], value=[str(steps[0])] if steps else []),  # cluster_steps_cb
            )

        load_btn.click(
            fn=on_load,
            inputs=[trace_dir_tb, model_path_tb],
            outputs=[load_status, records_state, align_step_dd, manifest_df, align_step_dd, cluster_steps_cb],
        )

        # Alignment: update layers when step changes
        align_step_dd.change(
            fn=alignment_get_layers,
            inputs=[records_state, trace_dir_tb, align_step_dd],
            outputs=[align_layer_dd],
        ).then(
            fn=alignment_on_step,
            inputs=[records_state, trace_dir_tb, model_path_tb, align_step_dd, align_layer_dd, align_agg_radio, align_metric_radio],
            outputs=[align_heatmap_html, align_stats_md, align_detail_df],
        )

        # Alignment: re-render on layer/agg/metric change
        for component in [align_layer_dd, align_agg_radio, align_metric_radio]:
            component.change(
                fn=alignment_on_step,
                inputs=[records_state, trace_dir_tb, model_path_tb, align_step_dd, align_layer_dd, align_agg_radio, align_metric_radio],
                outputs=[align_heatmap_html, align_stats_md, align_detail_df],
            )

        # Clustering: update on any control change
        cluster_inputs = [records_state, trace_dir_tb, model_path_tb, cluster_steps_cb, cluster_layer_dd, cluster_method_radio, cluster_color_radio]
        cluster_outputs = [cluster_plot, cluster_stats_md]

        for component in [cluster_steps_cb, cluster_layer_dd, cluster_method_radio, cluster_color_radio]:
            component.change(
                fn=clustering_view,
                inputs=cluster_inputs,
                outputs=cluster_outputs,
            )

    demo.launch(server_name=args.host, server_port=args.port, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
