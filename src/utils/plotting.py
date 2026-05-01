import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend (safe for headless environments)


def plot_learning_curves(
    results,
    title="Learning Curves",
    xlabel="Episode",
    ylabel="Smoothed Reward",
    window=100,
    save_path=None,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, metrics in results.items():
        smoothed = metrics.smoothed_rewards(window=window)
        ax.plot(smoothed, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_success_rates(
    results,
    title="Final Success Rate",
    last_n=100,
    save_path=None,
):
    labels = list(results.keys())
    rates = [m.success_rate(last_n=last_n) for m in results.values()]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    bars = ax.bar(labels, rates, color=colors)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.set_ylabel(f"Success Rate (last {last_n} episodes)")
    ax.set_xlabel("Configuration")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_help_usage(
    results,
    title="Help Usage Fraction",
    last_n=100,
    save_path=None,
):
    labels = list(results.keys())
    fractions = [m.mean_help_fraction(last_n=last_n) for m in results.values()]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    bars = ax.bar(labels, fractions, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylim(0, max(max(fractions) * 1.2, 0.1))
    ax.set_title(title)
    ax.set_ylabel(f"Mean Help Fraction (last {last_n} episodes)")
    ax.set_xlabel("Configuration")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_multi_panel(
    results_dict,
    env_name,
    algo_name,
    output_dir="results",
    window=100,
    last_n=100,
):
    labels = list(results_dict.keys())
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{env_name} – {algo_name}", fontsize=13, fontweight="bold")

    # --- Left: smoothed learning curves ---
    ax0 = axes[0]
    for (label, metrics), color in zip(results_dict.items(), colors):
        smoothed = metrics.smoothed_rewards(window=window)
        ax0.plot(smoothed, label=label, color=color)
    ax0.set_title("Learning Curves")
    ax0.set_xlabel("Episode")
    ax0.set_ylabel("Smoothed Reward")
    ax0.legend(loc="lower right", fontsize=7)
    ax0.grid(True, alpha=0.3)

    # --- Right: help usage bar chart ---
    ax2 = axes[1]
    fractions = [m.mean_help_fraction(last_n=last_n) for m in results_dict.values()]
    bars2 = ax2.bar(labels, fractions, color=colors)
    ax2.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    ax2.set_ylim(0, max(max(fractions) * 1.2, 0.1))
    ax2.set_title(f"Help Usage (last {last_n} eps)")
    ax2.set_ylabel("Mean Help Fraction")
    ax2.set_xlabel("Configuration")
    ax2.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    save_path = os.path.join(output_dir, f"{env_name}_{algo_name}_summary.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    return fig