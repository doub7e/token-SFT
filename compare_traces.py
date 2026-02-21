#!/usr/bin/env python3
"""
Compare per-token gradient trace analyses between two models (e.g., 1.5B vs 7B).

Produces side-by-side comparison figures highlighting:
  1. Alignment evolution comparison
  2. Positional decay comparison
  3. Module type contribution comparison
  4. Gradient concentration (Gini) comparison
  5. Token category comparison

Usage:
  python compare_traces.py \
      --summary-a output/asft-qwen2.5-1.5b-trace/token_grad_trace/analysis_v2/summary.json \
      --summary-b output/asft-qwen2.5-7b-paper-trace/token_grad_trace/analysis/summary.json \
      --label-a "Qwen2.5-1.5B" --label-b "Qwen2.5-7B" \
      --output-dir output/trace_comparison
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-a", type=str, required=True)
    p.add_argument("--summary-b", type=str, required=True)
    p.add_argument("--label-a", type=str, default="Model A")
    p.add_argument("--label-b", type=str, default="Model B")
    p.add_argument("--output-dir", type=str, default="output/trace_comparison")
    return p.parse_args()


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def compare_alignment_evolution(sa, sb, la, lb, out_dir):
    """Side-by-side alignment evolution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ae_a = sa.get("alignment_evolution", {})
    ae_b = sb.get("alignment_evolution", {})

    # Left: mean alignment
    ax = axes[0]
    if ae_a.get("steps"):
        ax.plot(ae_a["steps"], ae_a["alignment_mean"], "-o", color="steelblue",
                markersize=3, label=la)
        ax.fill_between(ae_a["steps"],
                        [m - s for m, s in zip(ae_a["alignment_mean"], ae_a["alignment_std"])],
                        [m + s for m, s in zip(ae_a["alignment_mean"], ae_a["alignment_std"])],
                        alpha=0.15, color="steelblue")
    if ae_b.get("steps"):
        ax.plot(ae_b["steps"], ae_b["alignment_mean"], "-s", color="darkorange",
                markersize=3, label=lb)
        ax.fill_between(ae_b["steps"],
                        [m - s for m, s in zip(ae_b["alignment_mean"], ae_b["alignment_std"])],
                        [m + s for m, s in zip(ae_b["alignment_mean"], ae_b["alignment_std"])],
                        alpha=0.15, color="darkorange")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Alignment")
    ax.set_title("Alignment Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Middle: fraction negative
    ax = axes[1]
    if ae_a.get("steps"):
        ax.plot(ae_a["steps"], ae_a["frac_negative"], "-o", color="steelblue",
                markersize=3, label=la)
    if ae_b.get("steps"):
        ax.plot(ae_b["steps"], ae_b["frac_negative"], "-s", color="darkorange",
                markersize=3, label=lb)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Fraction Opposing")
    ax.set_title("Fraction of Opposing Tokens")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: bar chart of overall mean
    ax = axes[2]
    means = []
    labels = []
    colors = []
    if ae_a.get("alignment_mean"):
        means.append(np.mean(ae_a["alignment_mean"]))
        labels.append(la)
        colors.append("steelblue")
    if ae_b.get("alignment_mean"):
        means.append(np.mean(ae_b["alignment_mean"]))
        labels.append(lb)
        colors.append("darkorange")
    ax.bar(labels, means, color=colors, alpha=0.8)
    ax.set_ylabel("Overall Mean Alignment")
    ax.set_title("Average Alignment Across All Steps")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Alignment Comparison: {la} vs {lb}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_alignment_evolution.pdf"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved compare_alignment_evolution.pdf")


def compare_positional_decay(sa, sb, la, lb, out_dir):
    """Compare how alignment decays with position."""
    fig, ax = plt.subplots(figsize=(10, 5))

    avp_a = sa.get("alignment_vs_position", {})
    avp_b = sb.get("alignment_vs_position", {})

    if avp_a.get("bin_centers"):
        # Normalize positions to [0, 1] for comparison
        centers_a = np.array(avp_a["bin_centers"])
        max_a = centers_a.max() if len(centers_a) > 0 else 1
        ax.plot(centers_a / max_a, avp_a["bin_means"], "-o", color="steelblue",
                markersize=4, label=la)
    if avp_b.get("bin_centers"):
        centers_b = np.array(avp_b["bin_centers"])
        max_b = centers_b.max() if len(centers_b) > 0 else 1
        ax.plot(centers_b / max_b, avp_b["bin_means"], "-s", color="darkorange",
                markersize=4, label=lb)

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Normalized Position (0=start, 1=end of response)")
    ax.set_ylabel("Mean Alignment")
    ax.set_title("Positional Decay of Alignment")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_positional_decay.pdf"), dpi=150)
    plt.close(fig)
    print("  Saved compare_positional_decay.pdf")


def compare_module_types(sa, sb, la, lb, out_dir):
    """Compare alignment by module type."""
    mtc_a = sa.get("module_type_comparison", {})
    mtc_b = sb.get("module_type_comparison", {})

    all_keys = sorted(set(list(mtc_a.keys()) + list(mtc_b.keys())))
    if not all_keys:
        print("  Skipping module type comparison (no data)")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(all_keys))
    width = 0.35

    means_a = [mtc_a.get(k, {}).get("mean", 0) for k in all_keys]
    means_b = [mtc_b.get(k, {}).get("mean", 0) for k in all_keys]
    stds_a = [mtc_a.get(k, {}).get("std", 0) for k in all_keys]
    stds_b = [mtc_b.get(k, {}).get("std", 0) for k in all_keys]

    ax.bar(x - width / 2, means_a, width, yerr=stds_a, label=la,
           color="steelblue", alpha=0.8, capsize=2)
    ax.bar(x + width / 2, means_b, width, yerr=stds_b, label=lb,
           color="darkorange", alpha=0.8, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(all_keys, rotation=45, ha="right", fontsize=9)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Module Type")
    ax.set_ylabel("Mean Alignment")
    ax.set_title("Alignment by Module Type")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_module_types.pdf"), dpi=150)
    plt.close(fig)
    print("  Saved compare_module_types.pdf")


def compare_concentration(sa, sb, la, lb, out_dir):
    """Compare gradient concentration (Gini, 80% threshold)."""
    ac_a = sa.get("alignment_concentration", {})
    ac_b = sb.get("alignment_concentration", {})

    if not ac_a and not ac_b:
        print("  Skipping concentration comparison (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gini comparison
    ax = axes[0]
    ginis = []
    labels = []
    colors = []
    if ac_a.get("gini_coefficient") is not None:
        ginis.append(ac_a["gini_coefficient"])
        labels.append(la)
        colors.append("steelblue")
    if ac_b.get("gini_coefficient") is not None:
        ginis.append(ac_b["gini_coefficient"])
        labels.append(lb)
        colors.append("darkorange")
    ax.bar(labels, ginis, color=colors, alpha=0.8)
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Gradient Concentration (Gini)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    for i, (g, l) in enumerate(zip(ginis, labels)):
        ax.text(i, g + 0.02, f"{g:.3f}", ha="center", fontsize=11)

    # 80% threshold comparison
    ax = axes[1]
    fracs = []
    labels2 = []
    colors2 = []
    if ac_a.get("frac_tokens_for_80pct_gradient") is not None:
        fracs.append(ac_a["frac_tokens_for_80pct_gradient"] * 100)
        labels2.append(la)
        colors2.append("steelblue")
    if ac_b.get("frac_tokens_for_80pct_gradient") is not None:
        fracs.append(ac_b["frac_tokens_for_80pct_gradient"] * 100)
        labels2.append(lb)
        colors2.append("darkorange")
    ax.bar(labels2, fracs, color=colors2, alpha=0.8)
    ax.set_ylabel("% of Tokens")
    ax.set_title("Fraction of Tokens Carrying 80% of Gradient")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")
    for i, (f, l) in enumerate(zip(fracs, labels2)):
        ax.text(i, f + 2, f"{f:.1f}%", ha="center", fontsize=11)

    fig.suptitle(f"Gradient Concentration: {la} vs {lb}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_concentration.pdf"), dpi=150)
    plt.close(fig)
    print("  Saved compare_concentration.pdf")


def compare_token_categories(sa, sb, la, lb, out_dir):
    """Compare alignment breakdown by token category."""
    tc_a = sa.get("token_category", {})
    tc_b = sb.get("token_category", {})

    all_cats = sorted(set(list(tc_a.keys()) + list(tc_b.keys())))
    if not all_cats:
        print("  Skipping token category comparison (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean alignment by category
    ax = axes[0]
    y = np.arange(len(all_cats))
    height = 0.35
    means_a = [tc_a.get(c, {}).get("mean", 0) for c in all_cats]
    means_b = [tc_b.get(c, {}).get("mean", 0) for c in all_cats]
    ax.barh(y - height / 2, means_a, height, label=la, color="steelblue", alpha=0.8)
    ax.barh(y + height / 2, means_b, height, label=lb, color="darkorange", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(all_cats)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Mean Alignment")
    ax.set_title("Alignment by Token Category")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    # Opposing fraction by category
    ax = axes[1]
    fracs_a = [tc_a.get(c, {}).get("frac_negative", 0) for c in all_cats]
    fracs_b = [tc_b.get(c, {}).get("frac_negative", 0) for c in all_cats]
    ax.barh(y - height / 2, fracs_a, height, label=la, color="steelblue", alpha=0.8)
    ax.barh(y + height / 2, fracs_b, height, label=lb, color="darkorange", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(all_cats)
    ax.set_xlabel("Fraction Opposing")
    ax.set_title("Opposing Fraction by Token Category")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(f"Token Category Comparison: {la} vs {lb}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_token_categories.pdf"), dpi=150)
    plt.close(fig)
    print("  Saved compare_token_categories.pdf")


def compare_lora_a_vs_b(sa, sb, la, lb, out_dir):
    """Compare LoRA A vs B gradient magnitudes."""
    lab_a = sa.get("lora_a_vs_b", {})
    lab_b = sb.get("lora_a_vs_b", {})

    if not lab_a.get("steps") and not lab_b.get("steps"):
        print("  Skipping lora A vs B comparison (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Model A
    ax = axes[0]
    if lab_a.get("steps"):
        ax.plot(lab_a["steps"], lab_a["lora_a_mean"], "-o", color="steelblue",
                markersize=3, label="LoRA A")
        ax.plot(lab_a["steps"], lab_a["lora_b_mean"], "-s", color="darkorange",
                markersize=3, label="LoRA B")
        ratios = [b / a if a > 0 else 0 for a, b in
                  zip(lab_a["lora_a_mean"], lab_a["lora_b_mean"])]
        avg_ratio = np.mean(ratios)
        ax.set_title(f"{la}\nB/A ratio: {avg_ratio:.1f}x")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Gradient Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Model B
    ax = axes[1]
    if lab_b.get("steps"):
        ax.plot(lab_b["steps"], lab_b["lora_a_mean"], "-o", color="steelblue",
                markersize=3, label="LoRA A")
        ax.plot(lab_b["steps"], lab_b["lora_b_mean"], "-s", color="darkorange",
                markersize=3, label="LoRA B")
        ratios = [b / a if a > 0 else 0 for a, b in
                  zip(lab_b["lora_a_mean"], lab_b["lora_b_mean"])]
        avg_ratio = np.mean(ratios)
        ax.set_title(f"{lb}\nB/A ratio: {avg_ratio:.1f}x")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Gradient Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.suptitle("LoRA A vs B Gradient Magnitudes", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_lora_a_vs_b.pdf"), dpi=150)
    plt.close(fig)
    print("  Saved compare_lora_a_vs_b.pdf")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading summaries...")
    sa = load_summary(args.summary_a)
    sb = load_summary(args.summary_b)
    la, lb = args.label_a, args.label_b

    print(f"\n=== 1. Alignment Evolution ===")
    compare_alignment_evolution(sa, sb, la, lb, args.output_dir)

    print(f"\n=== 2. Positional Decay ===")
    compare_positional_decay(sa, sb, la, lb, args.output_dir)

    print(f"\n=== 3. Module Type ===")
    compare_module_types(sa, sb, la, lb, args.output_dir)

    print(f"\n=== 4. Concentration ===")
    compare_concentration(sa, sb, la, lb, args.output_dir)

    print(f"\n=== 5. Token Categories ===")
    compare_token_categories(sa, sb, la, lb, args.output_dir)

    print(f"\n=== 6. LoRA A vs B ===")
    compare_lora_a_vs_b(sa, sb, la, lb, args.output_dir)

    print(f"\nAll comparison figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
