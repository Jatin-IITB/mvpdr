"""Aggregate experiment results into summary tables and update figures.

Scans the results/ directory for completed experiments and produces:
  - Markdown summary tables
  - LaTeX-ready tables
  - Updated publication figures with real numbers

Usage:
    python scripts/aggregate_results.py --results_dir results/
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_result_files(results_dir):
    """Recursively find all results.json files."""
    results = []
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f == "results.json":
                path = os.path.join(root, f)
                with open(path) as fh:
                    data = json.load(fh)
                data["_path"] = path
                results.append(data)
            elif f == "summary.json" and "zeroshot" in root:
                with open(os.path.join(root, f)) as fh:
                    zs_data = json.load(fh)
                for item in zs_data:
                    item["_path"] = os.path.join(root, f)
                    item["model"] = "CLIP Zero-Shot"
                    item["training"] = {"shots": 0}
                    item["performance"] = {
                        "accuracy": item["accuracy"],
                        "precision": item["precision"],
                        "recall": item["recall"],
                        "f1": item["f1"],
                    }
                    results.append(item)
    return results


def build_accuracy_table(results):
    """Build the main accuracy comparison table."""
    table = defaultdict(lambda: defaultdict(list))

    for r in results:
        model = r.get("model", "MVPDR")
        dataset = r.get("dataset", "unknown")
        shots = r.get("training", {}).get("shots", r.get("hyperparameters", {}).get("shots", 0))
        acc = r.get("performance", {}).get("accuracy", 0)

        key = (model, shots)
        table[key][dataset].append(acc)

    return table


def format_markdown_tables(results, output_path):
    """Generate markdown summary tables."""
    table = build_accuracy_table(results)

    datasets = sorted({r.get("dataset", "unknown") for r in results})

    lines = ["# MVPDR Experiment Results\n"]

    lines.append("## Classification Accuracy (% top-1)\n")
    header = f"| {'Method':<25} | {'Shots':>5} |"
    separator = f"|{'-' * 27}|{'-' * 7}|"
    for ds in datasets:
        header += f" {ds:>15} |"
        separator += f"{'-' * 17}|"
    lines.append(header)
    lines.append(separator)

    for (model, shots), ds_accs in sorted(table.items(), key=lambda x: (x[0][0], x[0][1])):
        row = f"| {model:<25} | {shots:>5} |"
        for ds in datasets:
            accs = ds_accs.get(ds, [])
            if accs:
                mean_acc = np.mean(accs)
                if len(accs) > 1:
                    std_acc = np.std(accs)
                    row += f" {mean_acc:>6.2f}±{std_acc:.1f}  |"
                else:
                    row += f" {mean_acc:>12.2f}    |"
            else:
                row += f" {'—':>12}    |"
        lines.append(row)

    lines.append("")

    ablation_results = [
        r for r in results
        if r.get("model") == "MVPDR+" and "architecture" in r
    ]
    if ablation_results:
        lines.append("## Component Ablation (PlantDoc, 20-shot)\n")
        lines.append(f"| {'Components':<40} | {'Accuracy':>10} | {'F1':>10} |")
        lines.append(f"|{'-' * 42}|{'-' * 12}|{'-' * 12}|")

        for r in sorted(ablation_results, key=lambda x: x["performance"]["accuracy"]):
            arch = r.get("architecture", {})
            parts = []
            if arch.get("prompt_learner"):
                parts.append("CoOp")
            if arch.get("prototype_bank"):
                parts.append("Proto")
            if arch.get("cross_attention"):
                parts.append("CrossAttn")
            name = " + ".join(parts) if parts else "None"
            acc = r["performance"]["accuracy"]
            f1 = r["performance"].get("f1", 0)
            lines.append(f"| {name:<40} | {acc:>9.2f}% | {f1:>9.2f}% |")

    lines.append("")

    backbone_results = defaultdict(list)
    for r in results:
        bb = r.get("backbone", "")
        if bb and r.get("model") == "MVPDR+":
            backbone_results[bb].append(r["performance"]["accuracy"])

    if len(backbone_results) > 1:
        lines.append("## Backbone Comparison (MVPDR+, PlantDoc, 20-shot)\n")
        lines.append(f"| {'Backbone':<15} | {'Accuracy':>10} |")
        lines.append(f"|{'-' * 17}|{'-' * 12}|")
        for bb in ["RN50", "RN101", "ViT-B/32", "ViT-B/16"]:
            if bb in backbone_results:
                accs = backbone_results[bb]
                mean_acc = np.mean(accs)
                lines.append(f"| {bb:<15} | {mean_acc:>9.2f}% |")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Markdown tables saved to {output_path}")
    return "\n".join(lines)


def generate_comparison_figure(results, output_path):
    """Generate accuracy comparison bar chart across datasets."""
    datasets = sorted({r.get("dataset", "") for r in results if r.get("dataset")})
    models = ["CLIP Zero-Shot", "MVPDR", "MVPDR+"]
    shots_target = 20

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(datasets))
    width = 0.25
    colors = ["#4A90D9", "#E8A838", "#2ECC71"]

    for i, model in enumerate(models):
        accs = []
        for ds in datasets:
            matching = [
                r for r in results
                if r.get("dataset") == ds
                and r.get("model", "MVPDR") == model
                and (r.get("training", {}).get("shots", r.get("hyperparameters", {}).get("shots", 0)) == shots_target
                     or model == "CLIP Zero-Shot")
            ]
            if matching:
                accs.append(np.mean([m["performance"]["accuracy"] for m in matching]))
            else:
                accs.append(0)

        bars = ax.bar(x + i * width, accs, width, label=model, color=colors[i], alpha=0.85)
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{acc:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Classification Accuracy Comparison (20-shot)", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([ds.replace("plant", "Plant") for ds in datasets], fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison figure saved to {output_path}")


def generate_shot_curve(results, output_path):
    """Generate few-shot learning curve (accuracy vs num shots)."""
    datasets = sorted({r.get("dataset", "") for r in results if r.get("dataset")})

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        for model, color, marker in [("MVPDR", "#E8A838", "s"), ("MVPDR+", "#2ECC71", "o")]:
            shot_accs = defaultdict(list)
            for r in results:
                if (r.get("dataset") == ds
                        and r.get("model", "MVPDR") == model):
                    shots = r.get("training", {}).get("shots",
                            r.get("hyperparameters", {}).get("shots", 0))
                    if shots > 0:
                        shot_accs[shots].append(r["performance"]["accuracy"])

            if shot_accs:
                shots_sorted = sorted(shot_accs.keys())
                means = [np.mean(shot_accs[s]) for s in shots_sorted]
                ax.plot(shots_sorted, means, f"-{marker}", color=color,
                        label=model, linewidth=2, markersize=8)

        zs = [r for r in results if r.get("dataset") == ds and r.get("model") == "CLIP Zero-Shot"]
        if zs:
            zs_acc = np.mean([r["performance"]["accuracy"] for r in zs])
            ax.axhline(y=zs_acc, color="#4A90D9", linestyle="--", label="CLIP Zero-Shot")

        ax.set_xlabel("Number of Shots", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(ds.replace("plant", "Plant"), fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Shot curve figure saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate MVPDR experiment results")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--output", default="results/summary", help="Summary output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    results = find_result_files(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Found {len(results)} experiment results\n")

    md = format_markdown_tables(results, os.path.join(args.output, "results_table.md"))
    print(md)

    generate_comparison_figure(results, os.path.join(args.output, "accuracy_comparison.png"))
    generate_shot_curve(results, os.path.join(args.output, "fewshot_curve.png"))

    print(f"\nAll summaries saved to {args.output}/")


if __name__ == "__main__":
    main()
