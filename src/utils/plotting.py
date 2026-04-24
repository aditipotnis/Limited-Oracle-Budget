"""Plotting utilities for RL experiment results."""

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
    """Plot smoothed episode-reward curves for multiple configurations.

    Args:
        results (dict[str, EpisodeMetrics]): Mapping from configuration label
            to metrics object.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        window (int): Smoothing window size.
        save_path (str | None): File path to save the figure.  If ``None``,
            the figure is returned without saving.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
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
    """Bar chart of final success rates across configurations.

    Args:
        results (dict[str, EpisodeMetrics]): Mapping from label to metrics.
        title (str): Plot title.
        last_n (int): Number of final episodes used to compute the rate.
        save_path (str | None): Optional file path to save the figure.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
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
    """Bar chart of mean help-usage fractions across configurations.

    Args:
        results (dict[str, EpisodeMetrics]): Mapping from label to metrics.
        title (str): Plot title.
        last_n (int): Number of final episodes used to compute the mean.
        save_path (str | None): Optional file path to save the figure.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
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
    """Produce and save all three standard plots for one (env, algorithm) pair.

    Args:
        results_dict (dict[str, EpisodeMetrics]): Mapping from config label to metrics.
        env_name (str): Environment name (used in titles and file names).
        algo_name (str): Algorithm name (used in titles and file names).
        output_dir (str): Directory where figures are saved.
        window (int): Reward smoothing window.
        last_n (int): Window for computing final metrics.

    Returns:
        tuple[Figure, Figure, Figure]: Learning-curve, success-rate, and
            help-usage figures.
    """
    prefix = os.path.join(output_dir, f"{env_name}_{algo_name}")
    fig1 = plot_learning_curves(
        results_dict,
        title=f"{env_name} – {algo_name}: Learning Curves",
        window=window,
        save_path=f"{prefix}_learning_curves.png",
    )
    fig2 = plot_success_rates(
        results_dict,
        title=f"{env_name} – {algo_name}: Final Success Rate",
        last_n=last_n,
        save_path=f"{prefix}_success_rates.png",
    )
    fig3 = plot_help_usage(
        results_dict,
        title=f"{env_name} – {algo_name}: Help Usage",
        last_n=last_n,
        save_path=f"{prefix}_help_usage.png",
    )
    return fig1, fig2, fig3
