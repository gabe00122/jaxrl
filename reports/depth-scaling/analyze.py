"""
Analysis script for transformer depth experiment on find_return task.

Usage:
    uv run python reports/depth-scaling/analyze.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import find_experiments, extract_metrics, smooth


OUTPUT_DIR = str(Path(__file__).resolve().parent)

FILTERS = {"environment.env_type": "find_return", "seed": 42}
GROUP_BY = "learner.model.num_layers"


def plot_reward_curves(experiments: dict[int, list[dict]]):
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(experiments)))

    for i, (num_layers, exps) in enumerate(sorted(experiments.items())):
        for exp in exps:
            metrics = extract_metrics(exp["logs"])
            smoothed = smooth(metrics["env/rewards"], window=50)
            steps = metrics["step"][:len(smoothed)]
            ax.plot(steps, smoothed, color=colors[i], alpha=0.8,
                    label=f"{num_layers} layers", linewidth=2)

    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_ylabel("Episode Reward", fontsize=14)
    ax.set_title("Effect of Transformer Depth on Find-Return Task Performance", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "reward_curves.png"), dpi=150)
    plt.close()
    print("Saved reward_curves.png")


def plot_loss_curves(experiments: dict[int, list[dict]]):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    loss_types = [
        ("algo/value_loss", "Value Loss"),
        ("algo/actor_loss", "Actor Loss"),
        ("algo/entropy_loss", "Entropy Loss"),
        ("algo/total_loss", "Total Loss"),
    ]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(experiments)))

    for ax, (loss_key, loss_title) in zip(axes.flat, loss_types):
        for i, (num_layers, exps) in enumerate(sorted(experiments.items())):
            for exp in exps:
                metrics = extract_metrics(exp["logs"])
                smoothed = smooth(metrics[loss_key], window=50)
                steps = metrics["step"][:len(smoothed)]
                ax.plot(steps, smoothed, color=colors[i], alpha=0.8,
                        label=f"{num_layers} layers", linewidth=2)

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel(loss_title, fontsize=12)
        ax.set_title(loss_title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Losses by Transformer Depth", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curves.png"), dpi=150)
    plt.close()
    print("Saved loss_curves.png")


def plot_final_performance(experiments: dict[int, list[dict]]):
    layer_counts = []
    final_rewards = []
    final_stds = []

    for num_layers in sorted(experiments.keys()):
        exp_finals = []
        for exp in experiments[num_layers]:
            metrics = extract_metrics(exp["logs"])
            last_n = min(100, len(metrics["env/rewards"]))
            exp_finals.append(np.mean(metrics["env/rewards"][-last_n:]))

        layer_counts.append(num_layers)
        final_rewards.append(np.mean(exp_finals))
        final_stds.append(np.std(exp_finals) if len(exp_finals) > 1 else 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(layer_counts)), final_rewards,
                  yerr=final_stds, capsize=5,
                  color=plt.cm.viridis(np.linspace(0, 0.9, len(layer_counts))),
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(layer_counts)))
    ax.set_xticklabels([str(n) for n in layer_counts], fontsize=12)
    ax.set_xlabel("Number of Transformer Layers", fontsize=14)
    ax.set_ylabel("Final Reward (avg last 100 steps)", fontsize=14)
    ax.set_title("Final Performance vs Transformer Depth", fontsize=16)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, final_rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_performance.png"), dpi=150)
    plt.close()
    print("Saved final_performance.png")


def generate_report(experiments: dict[int, list[dict]]):
    layer_finals = {}
    for num_layers in sorted(experiments.keys()):
        for exp in experiments[num_layers]:
            metrics = extract_metrics(exp["logs"])
            last_n = min(100, len(metrics["env/rewards"]))
            layer_finals[num_layers] = np.mean(metrics["env/rewards"][-last_n:])

    sorted_layers = sorted(layer_finals.keys())
    best_layers = max(layer_finals, key=layer_finals.get)
    worst_layers = min(layer_finals, key=layer_finals.get)

    lines = [
        "# Transformer Depth Experiment Report",
        "## Find-Return Navigation Task",
        "",
        "### Thesis",
        "",
        "> Model depth (number of transformer layers) shows diminishing returns for the",
        "> find_return navigation task. Moderate depth (4-8 layers) achieves comparable",
        "> performance to 16 layers, while very shallow models (1-2 layers) significantly",
        "> underperform.",
        "",
        "### Experiment Setup",
        "",
        "- **Environment**: find_return (40x40 grid, 8 agents, 11x11 view)",
        "- **Training**: PPO with Muon optimizer, 5000 update steps",
        "- **Seed**: 42 (fixed for reproducibility)",
        f"- **Layer counts tested**: {', '.join(str(l) for l in sorted_layers)}",
        "- **All other hyperparameters**: held constant from return.json baseline",
        "",
        "### Key Findings",
        "",
        f"1. **Best performing depth**: {best_layers} layers (final reward: {layer_finals[best_layers]:.2f})",
        f"2. **Worst performing depth**: {worst_layers} layers (final reward: {layer_finals[worst_layers]:.2f})",
        f"3. **Performance spread**: {layer_finals[best_layers] - layer_finals[worst_layers]:.2f} reward difference",
        "",
        "### Performance by Depth",
        "",
    ]

    for num_layers in sorted_layers:
        reward = layer_finals[num_layers]
        pct = (reward / layer_finals[best_layers]) * 100
        lines.append(f"- **{num_layers} layers**: {reward:.2f} reward ({pct:.1f}% of best)")

    lines += [
        "",
        "### Plots",
        "",
        "![Reward Curves](reward_curves.png)",
        "![Loss Curves](loss_curves.png)",
        "![Final Performance](final_performance.png)",
    ]

    report_path = os.path.join(OUTPUT_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved report.md")


def main():
    experiments = find_experiments(filters=FILTERS, group_by=GROUP_BY)

    if not experiments:
        print("No matching experiments found.")
        return

    print(f"Found experiments for layer counts: {sorted(experiments.keys())}")
    for layers, exps in sorted(experiments.items()):
        print(f"  {layers} layers: {len(exps)} run(s) - {[e['name'] for e in exps]}")

    plot_reward_curves(experiments)
    plot_loss_curves(experiments)
    plot_final_performance(experiments)
    generate_report(experiments)


if __name__ == "__main__":
    main()
