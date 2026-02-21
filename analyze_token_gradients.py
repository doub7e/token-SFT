#!/usr/bin/env python3
"""
Offline analysis of per-token gradient traces.

Produces:
  1. Alignment evolution over training steps (how alignment distribution shifts)
  2. Token-type analysis: instruction vs response token alignment statistics
  3. Per-layer contribution analysis: which layers contribute most to alignment
  4. Gradient magnitude evolution: how token gradient norms change over training
  5. Alignment vs position: do later tokens align more/less?
  6. "Surprising" tokens: highest-opposing tokens across steps
  7. Layer-wise gradient norm ratio (lora_A vs lora_B)
  8. Response-internal patterns: alignment of reasoning vs answer tokens

Saves all figures to an output directory as PDFs + a summary JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Import from our own modules
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_token_gradients import (
    load_trace_manifest,
    load_step_data,
    compute_alignment,
    compute_token_features,
    _classify_token,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", type=str, required=True)
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Where to save figures. Defaults to <trace-dir>/analysis")
    return p.parse_args()


@lru_cache(maxsize=2)
def _get_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _decode_token(tokenizer, tid: int) -> str:
    if tokenizer is None:
        return str(tid)
    t = tokenizer.decode([int(tid)])
    return t if t else str(tid)


# ---------------------------------------------------------------------------
# Analysis 1: Alignment evolution over training steps
# ---------------------------------------------------------------------------
def analysis_alignment_evolution(records, trace_dir, out_dir):
    """Track how alignment distribution changes over training."""
    steps = []
    means = []
    stds = []
    q10s = []
    q50s = []
    q90s = []
    frac_negative = []

    for rec in records:
        data = load_step_data(trace_dir, rec)
        alignment = compute_alignment(data, layer_name=None, aggregation="holistic")
        valid = data["valid_mask"].astype(bool)
        va = alignment[valid]
        if va.size == 0:
            continue
        steps.append(data["step"])
        means.append(float(va.mean()))
        stds.append(float(va.std()))
        q10, q50, q90 = np.quantile(va, [0.1, 0.5, 0.9])
        q10s.append(float(q10))
        q50s.append(float(q50))
        q90s.append(float(q90))
        frac_negative.append(float((va < 0).sum() / va.size))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.fill_between(steps, q10s, q90s, alpha=0.2, color="steelblue", label="10th-90th pctl")
    ax.plot(steps, q50s, "-o", color="steelblue", markersize=3, label="Median")
    ax.plot(steps, means, "--", color="darkorange", linewidth=1.5, label="Mean")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Token-Gradient Alignment (cosine sim)")
    ax.set_title("Alignment Distribution Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, frac_negative, "-s", color="crimson", markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Fraction of Opposing Tokens")
    ax.set_title("Fraction of Tokens with Negative Alignment")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alignment_evolution.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Saved alignment_evolution.pdf")

    return {
        "steps": steps,
        "alignment_mean": means,
        "alignment_std": stds,
        "frac_negative": frac_negative,
    }


# ---------------------------------------------------------------------------
# Analysis 2: Instruction vs Response alignment
# ---------------------------------------------------------------------------
def analysis_instruction_vs_response(records, trace_dir, out_dir):
    """Compare alignment for instruction tokens vs response tokens."""
    steps = []
    instr_means = []
    resp_means = []
    instr_stds = []
    resp_stds = []

    for rec in records:
        data = load_step_data(trace_dir, rec)
        alignment = compute_alignment(data, layer_name=None, aggregation="holistic")
        valid = data["valid_mask"].astype(bool)
        labels = data["labels"]
        T_minus_1 = len(valid)

        # Instruction tokens: labels[t+1] == -100
        # Response tokens: labels[t+1] != -100
        is_response = np.array([labels[t + 1] != -100 if t + 1 < len(labels) else False
                                for t in range(T_minus_1)])
        is_instruction = ~is_response

        resp_align = alignment[is_response & valid]
        instr_align = alignment[is_instruction & valid]

        steps.append(data["step"])
        resp_means.append(float(resp_align.mean()) if resp_align.size > 0 else 0)
        instr_means.append(float(instr_align.mean()) if instr_align.size > 0 else 0)
        resp_stds.append(float(resp_align.std()) if resp_align.size > 0 else 0)
        instr_stds.append(float(instr_align.std()) if instr_align.size > 0 else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, resp_means, "-o", color="forestgreen", markersize=4, label="Response tokens")
    ax.plot(steps, instr_means, "-s", color="purple", markersize=4, label="Instruction tokens")
    ax.fill_between(steps,
                     [m - s for m, s in zip(resp_means, resp_stds)],
                     [m + s for m, s in zip(resp_means, resp_stds)],
                     alpha=0.15, color="forestgreen")
    ax.fill_between(steps,
                     [m - s for m, s in zip(instr_means, instr_stds)],
                     [m + s for m, s in zip(instr_means, instr_stds)],
                     alpha=0.15, color="purple")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Alignment")
    ax.set_title("Instruction vs Response Token Alignment")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "instruction_vs_response.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Saved instruction_vs_response.pdf")

    return {"steps": steps, "response_mean": resp_means, "instruction_mean": instr_means}


# ---------------------------------------------------------------------------
# Analysis 3: Per-layer contribution
# ---------------------------------------------------------------------------
def analysis_per_layer_contribution(records, trace_dir, out_dir):
    """Compute per-layer alignment spread for a mid-training and late step."""
    if len(records) < 2:
        print("  Skipping per-layer analysis (need >= 2 steps)")
        return {}

    # Pick early, mid, late steps
    indices = [0, len(records) // 2, -1]
    fig, axes = plt.subplots(1, len(indices), figsize=(6 * len(indices), 5), sharey=True)

    result = {}
    for i, idx in enumerate(indices):
        rec = records[idx]
        data = load_step_data(trace_dir, rec)
        valid = data["valid_mask"].astype(bool)

        layer_alignments = {}
        for name in data["layer_names"]:
            if name not in data["Xp"] or name not in data["Yp"]:
                continue
            a = compute_alignment(data, layer_name=name, aggregation="holistic")
            va = a[valid]
            if va.size > 0:
                layer_alignments[name] = float(va.mean())

        if not layer_alignments:
            continue

        # Sort by mean alignment
        sorted_layers = sorted(layer_alignments.items(), key=lambda x: x[1], reverse=True)
        names = [n.split(".")[-1] if "lora" in n else n.split(".")[-2] for n, _ in sorted_layers]
        vals = [v for _, v in sorted_layers]

        # Simplify layer names for display: extract layer_idx and proj type
        short_names = []
        for full_name, _ in sorted_layers:
            parts = full_name.split(".")
            layer_idx = None
            proj_type = None
            for j, p in enumerate(parts):
                if p == "layers" and j + 1 < len(parts):
                    layer_idx = parts[j + 1]
                if p in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                    proj_type = p
                if "lora_" in p:
                    proj_type = (proj_type or "") + "." + p
            short = f"L{layer_idx}.{proj_type}" if layer_idx and proj_type else full_name[-30:]
            short_names.append(short)

        ax = axes[i]
        colors = ["steelblue" if v >= 0 else "salmon" for v in vals]
        # Show top-15 and bottom-15 for readability
        if len(vals) > 30:
            show_idx = list(range(15)) + list(range(len(vals) - 15, len(vals)))
            ax.barh([short_names[j] for j in show_idx],
                    [vals[j] for j in show_idx],
                    color=[colors[j] for j in show_idx])
        else:
            ax.barh(short_names, vals, color=colors)
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Mean Alignment")
        ax.set_title(f"Step {data['step']}")
        ax.tick_params(axis="y", labelsize=7)

        result[f"step_{data['step']}"] = {
            "top5": sorted_layers[:5],
            "bottom5": sorted_layers[-5:],
        }

    fig.suptitle("Per-Layer Alignment Contribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_layer_contribution.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Saved per_layer_contribution.pdf")
    return result


# ---------------------------------------------------------------------------
# Analysis 4: Gradient magnitude evolution
# ---------------------------------------------------------------------------
def analysis_gradient_magnitude(records, trace_dir, out_dir):
    """Track how gradient magnitudes (||Yp_t|| * ||Xp_t||) evolve."""
    steps = []
    resp_mag_mean = []
    resp_mag_std = []
    instr_mag_mean = []

    for rec in records:
        data = load_step_data(trace_dir, rec)
        valid = data["valid_mask"].astype(bool)
        labels = data["labels"]
        T_minus_1 = len(valid)
        is_response = np.array([labels[t + 1] != -100 if t + 1 < len(labels) else False
                                for t in range(T_minus_1)])

        # Sum gradient magnitude across all layers
        total_mag = np.zeros(T_minus_1, dtype=np.float64)
        for name in data["layer_names"]:
            if name not in data["Xp"] or name not in data["Yp"]:
                continue
            Xp = data["Xp"][name][:T_minus_1].astype(np.float32)
            Yp = data["Yp"][name][:T_minus_1].astype(np.float32)
            mag = np.linalg.norm(Xp, axis=1) * np.linalg.norm(Yp, axis=1)
            total_mag += mag

        resp_m = total_mag[is_response & valid]
        instr_m = total_mag[~is_response & valid]

        steps.append(data["step"])
        resp_mag_mean.append(float(resp_m.mean()) if resp_m.size > 0 else 0)
        resp_mag_std.append(float(resp_m.std()) if resp_m.size > 0 else 0)
        instr_mag_mean.append(float(instr_m.mean()) if instr_m.size > 0 else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, resp_mag_mean, "-o", color="forestgreen", markersize=4, label="Response tokens")
    ax.plot(steps, instr_mag_mean, "-s", color="purple", markersize=4, label="Instruction tokens")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Gradient Magnitude (sum over layers)")
    ax.set_title("Per-Token Gradient Magnitude Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gradient_magnitude.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Saved gradient_magnitude.pdf")
    return {"steps": steps, "resp_mag_mean": resp_mag_mean, "instr_mag_mean": instr_mag_mean}


# ---------------------------------------------------------------------------
# Analysis 5: Alignment vs position
# ---------------------------------------------------------------------------
def analysis_alignment_vs_position(records, trace_dir, out_dir):
    """Is there a positional trend in alignment?"""
    # Collect from all steps
    all_positions = []
    all_alignments = []
    all_steps = []

    for rec in records:
        data = load_step_data(trace_dir, rec)
        alignment = compute_alignment(data, layer_name=None, aggregation="holistic")
        valid = data["valid_mask"].astype(bool)
        labels = data["labels"]
        T_minus_1 = len(valid)
        is_response = np.array([labels[t + 1] != -100 if t + 1 < len(labels) else False
                                for t in range(T_minus_1)])
        mask = is_response & valid
        positions = np.arange(T_minus_1)[mask]
        aligns = alignment[mask]
        all_positions.extend(positions.tolist())
        all_alignments.extend(aligns.tolist())
        all_steps.extend([data["step"]] * len(positions))

    if not all_positions:
        print("  Skipping alignment_vs_position (no data)")
        return {}

    positions = np.array(all_positions)
    aligns = np.array(all_alignments)

    # Bin positions
    max_pos = int(positions.max()) + 1
    n_bins = min(30, max_pos)
    bin_edges = np.linspace(0, max_pos, n_bins + 1)
    bin_means = []
    bin_centers = []
    bin_stds = []
    for i in range(n_bins):
        mask = (positions >= bin_edges[i]) & (positions < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_means.append(float(aligns[mask].mean()))
            bin_stds.append(float(aligns[mask].std()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bin_centers, bin_means, width=(bin_edges[1] - bin_edges[0]) * 0.8,
           color="steelblue", alpha=0.7)
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt="none", ecolor="gray", alpha=0.5)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Token Position in Sequence (response tokens only)")
    ax.set_ylabel("Mean Alignment")
    ax.set_title("Alignment vs Position (aggregated over all steps)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alignment_vs_position.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Saved alignment_vs_position.pdf")
    return {"bin_centers": bin_centers, "bin_means": bin_means}


# ---------------------------------------------------------------------------
# Analysis 6: Surprising (opposing) tokens
# ---------------------------------------------------------------------------
def analysis_surprising_tokens(records, trace_dir, model_path, out_dir, top_k=30):
    """Find the most-opposing response tokens across all steps."""
    tokenizer = None
    try:
        tokenizer = _get_tokenizer(model_path)
    except Exception:
        pass

    rows = []
    for rec in records:
        data = load_step_data(trace_dir, rec)
        alignment = compute_alignment(data, layer_name=None, aggregation="holistic")
        valid = data["valid_mask"].astype(bool)
        labels = data["labels"]
        T_minus_1 = len(valid)
        is_response = np.array([labels[t + 1] != -100 if t + 1 < len(labels) else False
                                for t in range(T_minus_1)])

        for t in range(T_minus_1):
            if not (is_response[t] and valid[t]):
                continue
            rows.append({
                "step": data["step"],
                "position": t,
                "token_id": int(data["input_ids"][t]),
                "token": _decode_token(tokenizer, int(data["input_ids"][t])),
                "alignment": float(alignment[t]),
            })

    if not rows:
        print("  Skipping surprising_tokens (no response tokens)")
        return {}

    import pandas as pd
    df = pd.DataFrame(rows)

    # Most opposing
    opposing = df.nsmallest(top_k, "alignment")
    # Most aligned
    aligned = df.nlargest(top_k, "alignment")

    # Save as CSV
    opposing.to_csv(os.path.join(out_dir, "most_opposing_tokens.csv"), index=False)
    aligned.to_csv(os.path.join(out_dir, "most_aligned_tokens.csv"), index=False)

    # Summary figure: histogram of alignments
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["alignment"].values, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero alignment")
    ax.set_xlabel("Alignment (cosine similarity)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Token Alignments (N={len(df)}, all steps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alignment_histogram.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Saved alignment_histogram.pdf, most_opposing_tokens.csv, most_aligned_tokens.csv")

    return {
        "top_opposing": opposing[["step", "token", "alignment"]].to_dict("records")[:10],
        "top_aligned": aligned[["step", "token", "alignment"]].to_dict("records")[:10],
    }


# ---------------------------------------------------------------------------
# Analysis 7: Alignment heatmap (step x position)
# ---------------------------------------------------------------------------
def analysis_alignment_heatmap(records, trace_dir, out_dir):
    """2D heatmap: training step (y) vs token position (x), color = alignment."""
    max_T = 0
    step_data = []
    for rec in records:
        data = load_step_data(trace_dir, rec)
        alignment = compute_alignment(data, layer_name=None, aggregation="holistic")
        valid = data["valid_mask"].astype(bool)
        labels = data["labels"]
        T_minus_1 = len(valid)
        is_response = np.array([labels[t + 1] != -100 if t + 1 < len(labels) else False
                                for t in range(T_minus_1)])
        # Mark instruction tokens as NaN
        display = np.full(T_minus_1, np.nan)
        display[is_response & valid] = alignment[is_response & valid]
        step_data.append((data["step"], display))
        max_T = max(max_T, T_minus_1)

    if not step_data:
        return {}

    # Build matrix (steps x max_T)
    matrix = np.full((len(step_data), max_T), np.nan)
    step_labels = []
    for i, (step, display) in enumerate(step_data):
        matrix[i, :len(display)] = display
        step_labels.append(str(step))

    fig, ax = plt.subplots(figsize=(14, max(4, len(step_data) * 0.3)))
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("lightgray")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-1, vmax=1,
                   interpolation="nearest")
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Training Step")
    ax.set_yticks(range(len(step_labels)))
    ax.set_yticklabels(step_labels, fontsize=8)
    ax.set_title("Per-Token Alignment Heatmap Over Training")
    plt.colorbar(im, ax=ax, label="Alignment (cosine sim)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alignment_heatmap.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Saved alignment_heatmap.pdf")
    return {}


# ---------------------------------------------------------------------------
# Analysis 8: LoRA A vs B gradient norms
# ---------------------------------------------------------------------------
def analysis_lora_a_vs_b(records, trace_dir, out_dir):
    """Compare gradient magnitudes between LoRA A and LoRA B layers."""
    steps = []
    a_means = []
    b_means = []

    for rec in records:
        data = load_step_data(trace_dir, rec)
        valid = data["valid_mask"].astype(bool)
        T_minus_1 = len(valid)

        a_mags = []
        b_mags = []
        for name in data["layer_names"]:
            if name not in data["Xp"] or name not in data["Yp"]:
                continue
            Xp = data["Xp"][name][:T_minus_1].astype(np.float32)
            Yp = data["Yp"][name][:T_minus_1].astype(np.float32)
            mag = np.linalg.norm(Xp, axis=1) * np.linalg.norm(Yp, axis=1)
            valid_mag = mag[valid]
            if valid_mag.size == 0:
                continue
            if "lora_A" in name:
                a_mags.append(float(valid_mag.mean()))
            elif "lora_B" in name:
                b_mags.append(float(valid_mag.mean()))

        if a_mags and b_mags:
            steps.append(data["step"])
            a_means.append(float(np.mean(a_mags)))
            b_means.append(float(np.mean(b_mags)))

    if not steps:
        print("  Skipping lora_a_vs_b (no LoRA layers found)")
        return {}

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, a_means, "-o", color="steelblue", markersize=4, label="LoRA A (down-proj)")
    ax.plot(steps, b_means, "-s", color="darkorange", markersize=4, label="LoRA B (up-proj)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Gradient Magnitude")
    ax.set_title("LoRA A vs LoRA B Gradient Magnitudes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lora_a_vs_b.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Saved lora_a_vs_b.pdf")
    return {"steps": steps, "lora_a_mean": a_means, "lora_b_mean": b_means}


# ---------------------------------------------------------------------------
# Analysis 9: Token Category Analysis
# ---------------------------------------------------------------------------
def analysis_token_category(records, trace_dir, model_path, out_dir):
    """Break down alignment by token category (numbers, operators, text, punctuation, etc.)."""
    tokenizer = None
    try:
        tokenizer = _get_tokenizer(model_path)
    except Exception:
        pass

    cat_alignments: Dict[str, list] = {}

    for rec in records:
        data = load_step_data(trace_dir, rec)
        alignment = compute_alignment(data, layer_name=None, aggregation="holistic")
        valid = data["valid_mask"].astype(bool)
        labels = data["labels"]
        T_minus_1 = len(valid)
        is_response = np.array([labels[t + 1] != -100 if t + 1 < len(labels) else False
                                for t in range(T_minus_1)])

        for t in range(T_minus_1):
            if not (is_response[t] and valid[t]):
                continue
            token_text = _decode_token(tokenizer, int(data["input_ids"][t]))
            cat = _classify_token(token_text)
            cat_alignments.setdefault(cat, []).append(float(alignment[t]))

    if not cat_alignments:
        print("  Skipping token_category (no data)")
        return {}

    # Sort categories by mean alignment
    cat_stats = {}
    for cat, vals in sorted(cat_alignments.items(), key=lambda x: np.mean(x[1])):
        arr = np.array(vals)
        cat_stats[cat] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "frac_negative": float((arr < 0).mean()),
            "count": len(vals),
        }

    categories = list(cat_stats.keys())
    means = [cat_stats[c]["mean"] for c in categories]
    stds = [cat_stats[c]["std"] for c in categories]
    counts = [cat_stats[c]["count"] for c in categories]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar chart of mean alignment by category
    ax = axes[0]
    colors = ["steelblue" if m >= 0 else "salmon" for m in means]
    bars = ax.barh(categories, means, xerr=stds, color=colors, alpha=0.8, capsize=3)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Mean Alignment")
    ax.set_title("Alignment by Token Category")
    ax.grid(True, alpha=0.3, axis="x")
    # Annotate counts
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"n={count}", va="center", fontsize=7, color="gray")

    # Right: fraction negative by category
    ax = axes[1]
    frac_negs = [cat_stats[c]["frac_negative"] for c in categories]
    ax.barh(categories, frac_negs, color="crimson", alpha=0.6)
    ax.set_xlabel("Fraction Opposing (alignment < 0)")
    ax.set_title("Opposing Fraction by Token Category")
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "token_category.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Saved token_category.pdf")
    return cat_stats


# ---------------------------------------------------------------------------
# Analysis 10: Gradient Magnitude vs Alignment Scatter
# ---------------------------------------------------------------------------
def analysis_magnitude_vs_alignment(records, trace_dir, out_dir):
    """Scatter plot: is alignment correlated with gradient magnitude?"""
    all_mag = []
    all_align = []

    for rec in records:
        data = load_step_data(trace_dir, rec)
        alignment = compute_alignment(data, layer_name=None, aggregation="holistic")
        valid = data["valid_mask"].astype(bool)
        labels = data["labels"]
        T_minus_1 = len(valid)
        is_response = np.array([labels[t + 1] != -100 if t + 1 < len(labels) else False
                                for t in range(T_minus_1)])
        mask = is_response & valid

        # Sum gradient magnitude across all layers
        total_mag = np.zeros(T_minus_1, dtype=np.float64)
        for name in data["layer_names"]:
            if name not in data["Xp"] or name not in data["Yp"]:
                continue
            Xp = data["Xp"][name][:T_minus_1].astype(np.float32)
            Yp = data["Yp"][name][:T_minus_1].astype(np.float32)
            mag = np.linalg.norm(Xp, axis=1) * np.linalg.norm(Yp, axis=1)
            total_mag += mag

        all_mag.extend(total_mag[mask].tolist())
        all_align.extend(alignment[mask].tolist())

    if not all_mag:
        print("  Skipping magnitude_vs_alignment (no data)")
        return {}

    mag = np.array(all_mag)
    align = np.array(all_align)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: 2D density / scatter
    ax = axes[0]
    ax.scatter(align, mag, s=1, alpha=0.15, color="steelblue", rasterized=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Alignment (cosine similarity)")
    ax.set_ylabel("Gradient Magnitude (sum over layers)")
    ax.set_title("Gradient Magnitude vs Alignment")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Right: bin alignment and show mean magnitude per bin
    ax = axes[1]
    n_bins = 30
    bin_edges = np.linspace(-1, 1, n_bins + 1)
    bin_centers = []
    bin_mag_means = []
    bin_counts = []
    for i in range(n_bins):
        mask_bin = (align >= bin_edges[i]) & (align < bin_edges[i + 1])
        if mask_bin.sum() > 3:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_mag_means.append(float(mag[mask_bin].mean()))
            bin_counts.append(int(mask_bin.sum()))

    ax.bar(bin_centers, bin_mag_means,
           width=(bin_edges[1] - bin_edges[0]) * 0.8, color="steelblue", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Alignment Bin")
    ax.set_ylabel("Mean Gradient Magnitude")
    ax.set_title("Mean Gradient Magnitude by Alignment Bin")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "magnitude_vs_alignment.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Saved magnitude_vs_alignment.pdf")

    # Compute correlation
    corr = float(np.corrcoef(align, np.log1p(mag))[0, 1])
    return {"pearson_corr_align_vs_log_mag": corr, "n_tokens": len(mag)}


# ---------------------------------------------------------------------------
# Analysis 11: Module-Type Alignment Comparison
# ---------------------------------------------------------------------------
def analysis_module_type_comparison(records, trace_dir, out_dir):
    """Compare alignment across projection types (q/k/v/o/gate/up/down)."""
    proj_types = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # Separate lora_A and lora_B
    type_alignments: Dict[str, list] = {}

    for rec in records:
        data = load_step_data(trace_dir, rec)
        valid = data["valid_mask"].astype(bool)
        labels = data["labels"]
        T_minus_1 = len(valid)
        is_response = np.array([labels[t + 1] != -100 if t + 1 < len(labels) else False
                                for t in range(T_minus_1)])
        mask = is_response & valid

        for name in data["layer_names"]:
            if name not in data["Xp"] or name not in data["Yp"]:
                continue
            # Identify projection type
            proj = None
            for pt in proj_types:
                if pt in name:
                    proj = pt
                    break
            if proj is None:
                continue

            # Identify LoRA component
            lora_tag = ""
            if "lora_A" in name:
                lora_tag = ".A"
            elif "lora_B" in name:
                lora_tag = ".B"

            key = proj + lora_tag

            a = compute_alignment(data, layer_name=name, aggregation="holistic")
            va = a[mask]
            if va.size > 0:
                type_alignments.setdefault(key, []).append(float(va.mean()))

    if not type_alignments:
        print("  Skipping module_type_comparison (no data)")
        return {}

    # Aggregate
    type_stats = {}
    for key in sorted(type_alignments.keys()):
        vals = np.array(type_alignments[key])
        type_stats[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "count": len(vals),
        }

    keys = list(type_stats.keys())
    means = [type_stats[k]["mean"] for k in keys]
    stds = [type_stats[k]["std"] for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(keys))
    colors = ["steelblue" if m >= 0 else "salmon" for m in means]
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=9)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Module Type")
    ax.set_ylabel("Mean Alignment")
    ax.set_title("Alignment by Module Type (averaged over layers and steps)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "module_type_comparison.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Saved module_type_comparison.pdf")
    return type_stats


# ---------------------------------------------------------------------------
# Analysis 12: Alignment Concentration (Pareto/Gini)
# ---------------------------------------------------------------------------
def analysis_alignment_concentration(records, trace_dir, out_dir):
    """What fraction of tokens contribute most of the gradient signal?

    Computes:
    - Lorenz curve for gradient magnitude (Pareto analysis)
    - Gini coefficient
    - Fraction of tokens contributing 80% of total gradient
    """
    all_mag = []

    for rec in records:
        data = load_step_data(trace_dir, rec)
        valid = data["valid_mask"].astype(bool)
        labels = data["labels"]
        T_minus_1 = len(valid)
        is_response = np.array([labels[t + 1] != -100 if t + 1 < len(labels) else False
                                for t in range(T_minus_1)])
        mask = is_response & valid

        total_mag = np.zeros(T_minus_1, dtype=np.float64)
        for name in data["layer_names"]:
            if name not in data["Xp"] or name not in data["Yp"]:
                continue
            Xp = data["Xp"][name][:T_minus_1].astype(np.float32)
            Yp = data["Yp"][name][:T_minus_1].astype(np.float32)
            mag = np.linalg.norm(Xp, axis=1) * np.linalg.norm(Yp, axis=1)
            total_mag += mag

        all_mag.extend(total_mag[mask].tolist())

    if not all_mag:
        print("  Skipping alignment_concentration (no data)")
        return {}

    mag = np.array(all_mag)
    mag_sorted = np.sort(mag)
    n = len(mag_sorted)
    cumsum = np.cumsum(mag_sorted) / mag_sorted.sum()
    x_frac = np.arange(1, n + 1) / n

    # Gini coefficient
    gini = 1 - 2 * np.trapezoid(cumsum, x_frac)

    # Find fraction contributing 80% of total
    idx_80 = np.searchsorted(cumsum, 0.2)  # bottom 20% of cumulative = top 80%
    frac_for_80 = 1 - idx_80 / n

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Lorenz curve
    ax = axes[0]
    ax.plot(x_frac, cumsum, color="steelblue", linewidth=2, label="Actual")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect equality")
    ax.fill_between(x_frac, cumsum, x_frac, alpha=0.15, color="steelblue")
    ax.set_xlabel("Fraction of Tokens (sorted by gradient magnitude)")
    ax.set_ylabel("Cumulative Fraction of Total Gradient")
    ax.set_title(f"Lorenz Curve (Gini = {gini:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: gradient magnitude distribution
    ax = axes[1]
    ax.hist(mag, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(np.median(mag), color="red", linestyle="--", linewidth=1.5,
               label=f"Median = {np.median(mag):.1f}")
    ax.axvline(np.mean(mag), color="orange", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(mag):.1f}")
    ax.set_xlabel("Gradient Magnitude (sum over layers)")
    ax.set_ylabel("Count")
    ax.set_title(f"Gradient Magnitude Distribution\n"
                 f"Top {frac_for_80*100:.0f}% tokens carry 80% of gradient signal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alignment_concentration.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Saved alignment_concentration.pdf")
    return {
        "gini_coefficient": float(gini),
        "frac_tokens_for_80pct_gradient": float(frac_for_80),
        "n_tokens": n,
    }


# ---------------------------------------------------------------------------
# Analysis 13: Alignment Autocorrelation
# ---------------------------------------------------------------------------
def analysis_alignment_autocorrelation(records, trace_dir, out_dir, max_lag=30):
    """Do neighboring tokens have correlated alignment? Reveals structure in gradient patterns."""
    # Collect per-step autocorrelation
    step_autocorrs = []

    for rec in records:
        data = load_step_data(trace_dir, rec)
        alignment = compute_alignment(data, layer_name=None, aggregation="holistic")
        valid = data["valid_mask"].astype(bool)
        labels = data["labels"]
        T_minus_1 = len(valid)
        is_response = np.array([labels[t + 1] != -100 if t + 1 < len(labels) else False
                                for t in range(T_minus_1)])
        mask = is_response & valid

        # Get contiguous response alignment (skip non-response gaps)
        resp_align = alignment[mask]
        if len(resp_align) < max_lag + 10:
            continue

        # Compute autocorrelation for each lag
        mean_a = resp_align.mean()
        var_a = resp_align.var()
        if var_a < 1e-12:
            continue
        autocorrs = []
        for lag in range(1, max_lag + 1):
            c = np.mean((resp_align[:-lag] - mean_a) * (resp_align[lag:] - mean_a)) / var_a
            autocorrs.append(float(c))
        step_autocorrs.append(autocorrs)

    if not step_autocorrs:
        print("  Skipping alignment_autocorrelation (insufficient data)")
        return {}

    arr = np.array(step_autocorrs)  # [n_steps, max_lag]
    mean_ac = arr.mean(axis=0)
    std_ac = arr.std(axis=0)
    lags = np.arange(1, max_lag + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(lags, mean_ac - std_ac, mean_ac + std_ac, alpha=0.2, color="steelblue")
    ax.plot(lags, mean_ac, "-o", color="steelblue", markersize=3, linewidth=2)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    # Significance bounds (approx 1/sqrt(N))
    avg_n = np.mean([len(a) for a in step_autocorrs]) if step_autocorrs else 100
    sig_bound = 1.96 / np.sqrt(avg_n)
    ax.axhline(sig_bound, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label=f"95% CI ({sig_bound:.3f})")
    ax.axhline(-sig_bound, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Lag (token distance)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Alignment Autocorrelation (do neighboring tokens co-align?)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alignment_autocorrelation.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Saved alignment_autocorrelation.pdf")
    return {"lags": lags.tolist(), "mean_autocorr": mean_ac.tolist()}


# ---------------------------------------------------------------------------
# Analysis 14: Layer-Position Interaction Heatmap
# ---------------------------------------------------------------------------
def analysis_layer_position_heatmap(records, trace_dir, out_dir, n_pos_bins=20):
    """2D heatmap: layer (y) vs position bin (x), color = mean alignment.

    Reveals whether certain layers specialize in early vs late positions.
    """
    # Collect layer names from first record
    data0 = load_step_data(trace_dir, records[0])

    # Filter to a manageable set of layers (LoRA layers or all)
    all_layer_names = [n for n in data0["layer_names"]
                       if n in data0["Xp"] and n in data0["Yp"]]

    # Group by layer index + proj type for readability
    def _short_name(full_name):
        parts = full_name.split(".")
        layer_idx = proj_type = lora_tag = None
        for j, p in enumerate(parts):
            if p == "layers" and j + 1 < len(parts):
                layer_idx = parts[j + 1]
            if p in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                proj_type = p
            if "lora_A" in p:
                lora_tag = "A"
            elif "lora_B" in p:
                lora_tag = "B"
        s = f"L{layer_idx}.{proj_type}" if layer_idx and proj_type else full_name[-25:]
        if lora_tag:
            s += f".{lora_tag}"
        return s

    # Filter to v_proj layers only (they dominate alignment)
    v_proj_layers = [n for n in all_layer_names if "v_proj" in n]
    if not v_proj_layers:
        v_proj_layers = all_layer_names[:20]  # fallback

    # Collect alignment per (layer, position_bin) across all steps
    layer_pos_data: Dict[str, list] = {n: [] for n in v_proj_layers}

    for rec in records:
        data = load_step_data(trace_dir, rec)
        valid = data["valid_mask"].astype(bool)
        labels = data["labels"]
        T_minus_1 = len(valid)
        is_response = np.array([labels[t + 1] != -100 if t + 1 < len(labels) else False
                                for t in range(T_minus_1)])
        mask = is_response & valid
        positions = np.arange(T_minus_1)

        for name in v_proj_layers:
            if name not in data["Xp"] or name not in data["Yp"]:
                continue
            a = compute_alignment(data, layer_name=name, aggregation="holistic")
            for t in range(T_minus_1):
                if mask[t]:
                    layer_pos_data[name].append((positions[t], float(a[t])))

    if not any(layer_pos_data.values()):
        print("  Skipping layer_position_heatmap (no data)")
        return {}

    # Find max position
    max_pos = max(p for name in layer_pos_data for p, _ in layer_pos_data[name]) + 1
    bin_edges = np.linspace(0, max_pos, n_pos_bins + 1)

    # Build matrix
    layer_names_sorted = sorted(v_proj_layers, key=lambda n: (
        int(n.split("layers.")[1].split(".")[0]) if "layers." in n else 0,
        n
    ))
    matrix = np.full((len(layer_names_sorted), n_pos_bins), np.nan)
    short_names = [_short_name(n) for n in layer_names_sorted]

    for i, name in enumerate(layer_names_sorted):
        entries = layer_pos_data[name]
        if not entries:
            continue
        positions_arr = np.array([p for p, _ in entries])
        aligns_arr = np.array([a for _, a in entries])
        for j in range(n_pos_bins):
            bin_mask = (positions_arr >= bin_edges[j]) & (positions_arr < bin_edges[j + 1])
            if bin_mask.sum() > 0:
                matrix[i, j] = aligns_arr[bin_mask].mean()

    fig, ax = plt.subplots(figsize=(12, max(4, len(layer_names_sorted) * 0.25)))
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("lightgray")
    vmax = max(0.3, np.nanmax(np.abs(matrix)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xlabel("Position Bin")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=6)
    bin_labels = [f"{int(bin_edges[j])}" for j in range(0, n_pos_bins, max(1, n_pos_bins // 10))]
    ax.set_xticks(range(0, n_pos_bins, max(1, n_pos_bins // 10)))
    ax.set_xticklabels(bin_labels, fontsize=8)
    ax.set_title("Layer-Position Alignment Interaction (v_proj layers)")
    plt.colorbar(im, ax=ax, label="Mean Alignment")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "layer_position_heatmap.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Saved layer_position_heatmap.pdf")
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    trace_dir = args.trace_dir
    model_path = args.model_path
    out_dir = args.output_dir or os.path.join(trace_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading traces from: {trace_dir}")
    records = load_trace_manifest(trace_dir)
    print(f"Found {len(records)} trace steps")

    if not records:
        print("No trace records found. Exiting.")
        return

    summary = {"trace_dir": trace_dir, "num_steps": len(records)}

    print("\n=== Analysis 1: Alignment Evolution ===")
    summary["alignment_evolution"] = analysis_alignment_evolution(records, trace_dir, out_dir)

    print("\n=== Analysis 2: Instruction vs Response ===")
    summary["instruction_vs_response"] = analysis_instruction_vs_response(records, trace_dir, out_dir)

    print("\n=== Analysis 3: Per-Layer Contribution ===")
    summary["per_layer"] = analysis_per_layer_contribution(records, trace_dir, out_dir)

    print("\n=== Analysis 4: Gradient Magnitude ===")
    summary["gradient_magnitude"] = analysis_gradient_magnitude(records, trace_dir, out_dir)

    print("\n=== Analysis 5: Alignment vs Position ===")
    summary["alignment_vs_position"] = analysis_alignment_vs_position(records, trace_dir, out_dir)

    print("\n=== Analysis 6: Surprising Tokens ===")
    summary["surprising_tokens"] = analysis_surprising_tokens(
        records, trace_dir, model_path, out_dir
    )

    print("\n=== Analysis 7: Alignment Heatmap ===")
    analysis_alignment_heatmap(records, trace_dir, out_dir)

    print("\n=== Analysis 8: LoRA A vs B ===")
    summary["lora_a_vs_b"] = analysis_lora_a_vs_b(records, trace_dir, out_dir)

    print("\n=== Analysis 9: Token Category ===")
    summary["token_category"] = analysis_token_category(records, trace_dir, model_path, out_dir)

    print("\n=== Analysis 10: Magnitude vs Alignment ===")
    summary["magnitude_vs_alignment"] = analysis_magnitude_vs_alignment(records, trace_dir, out_dir)

    print("\n=== Analysis 11: Module Type Comparison ===")
    summary["module_type_comparison"] = analysis_module_type_comparison(records, trace_dir, out_dir)

    print("\n=== Analysis 12: Alignment Concentration ===")
    summary["alignment_concentration"] = analysis_alignment_concentration(records, trace_dir, out_dir)

    print("\n=== Analysis 13: Alignment Autocorrelation ===")
    summary["alignment_autocorrelation"] = analysis_alignment_autocorrelation(records, trace_dir, out_dir)

    print("\n=== Analysis 14: Layer-Position Heatmap ===")
    analysis_layer_position_heatmap(records, trace_dir, out_dir)

    # Save summary
    summary_path = os.path.join(out_dir, "summary.json")

    def _jsonable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_jsonable)
    print(f"\nSummary saved to {summary_path}")
    print(f"All figures saved to {out_dir}")


if __name__ == "__main__":
    main()
