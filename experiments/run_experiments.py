"""Main experiment runner for the Limited Oracle Budget project.

Run all experiments comparing Q-learning and SARSA with and without oracle
access under various budget, penalty, and oracle-quality conditions.

Usage::

    python experiments/run_experiments.py [--env ENV] [--algo ALGO]
                                          [--episodes N] [--seeds N]
                                          [--output DIR] [--quick]

Examples::

    # Full experiment suite (both envs, both algorithms, 5 seeds)
    python experiments/run_experiments.py

    # Quick smoke test (fewer episodes and seeds)
    python experiments/run_experiments.py --quick

    # Only CliffWalking with Q-learning
    python experiments/run_experiments.py --env CliffWalking-v1 --algo qlearning
"""

import argparse
import os
import sys
import time

# Ensure src is importable when running from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym

from experiments.training import (
    compute_optimal_policy,
    make_oracle,
    run_seeds,
    train_qlearning,
    train_sarsa,
)
from src.environments.oracle_env import OracleEnv
from src.utils.plotting import plot_multi_panel

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

ENVIRONMENTS = {
    "CliffWalking-v1": {"max_steps": 200, "env_kwargs": {}},
    "FrozenLake-v1": {"max_steps": 200, "env_kwargs": {"is_slippery": False}},
}

ALGORITHMS = {
    "qlearning": train_qlearning,
    "sarsa": train_sarsa,
}

# Hyperparameters shared across all runs
AGENT_KWARGS = dict(
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
)

# Budget variants: None means unlimited
BUDGETS = [0, 5, 10, 20, None]

# Oracle quality variants (type, accuracy)
ORACLE_TYPES = [
    ("perfect", 1.0),
    ("noisy", 0.8),
    ("noisy", 0.5),
    ("random", 0.0),
]

# Help-penalty variants applied when budget > 0
HELP_PENALTIES = [0.0, -0.1, -0.5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _budget_label(budget):
    return "unlimited" if budget is None else str(budget)


def _config_label(oracle_type, accuracy, budget, penalty):
    oracle_str = (
        f"{oracle_type}"
        if oracle_type == "perfect"
        else f"{oracle_type}({accuracy:.0%})"
    )
    return (
        f"budget={_budget_label(budget)} "
        f"oracle={oracle_str} "
        f"penalty={penalty}"
    )


def make_env_factory(env_name, oracle, budget, penalty, env_kwargs):
    """Return a zero-argument callable that constructs a fresh OracleEnv."""
    def factory():
        return OracleEnv(
            env_name=env_name,
            oracle=oracle,
            max_budget=budget,
            help_penalty=penalty,
            no_budget_penalty=-1.0,
            **env_kwargs,
        )
    return factory


def make_baseline_factory(env_name, env_kwargs):
    """Return a factory for a plain Gymnasium environment (no oracle)."""
    def factory():
        return gym.make(env_name, **env_kwargs)
    return factory


# ---------------------------------------------------------------------------
# Core experiment: vary budgets for one (env, algo, oracle, penalty) setting
# ---------------------------------------------------------------------------

def run_budget_comparison(
    env_name,
    algo_name,
    train_fn,
    optimal_policy,
    n_base_actions,
    oracle_type,
    accuracy,
    help_penalty,
    n_episodes,
    max_steps,
    n_seeds,
    env_kwargs,
    output_dir,
):
    """Train with multiple budget levels and collect metrics for comparison.

    Returns:
        dict[str, EpisodeMetrics]
    """
    results = {}

    # --- No-oracle baseline (standard RL, budget=0 means no help possible) ---
    label_base = "no oracle (baseline)"
    print(f"  [{label_base}]")
    oracle = make_oracle(oracle_type, optimal_policy, n_base_actions,
                         accuracy=accuracy, seed=0)
    env_factory = make_env_factory(
        env_name, oracle, budget=0, penalty=0.0, env_kwargs=env_kwargs
    )
    results[label_base] = run_seeds(
        train_fn,
        env_factory,
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        max_steps=max_steps,
        env_name=env_name,
        **AGENT_KWARGS,
    )

    # --- Oracle variants with different budgets ---
    for budget in BUDGETS:
        if budget == 0:
            continue  # already captured as baseline above
        oracle = make_oracle(oracle_type, optimal_policy, n_base_actions,
                             accuracy=accuracy, seed=0)
        label = _config_label(oracle_type, accuracy, budget, help_penalty)
        print(f"  [{label}]")
        env_factory = make_env_factory(
            env_name, oracle, budget=budget, penalty=help_penalty,
            env_kwargs=env_kwargs
        )
        results[label] = run_seeds(
            train_fn,
            env_factory,
            n_seeds=n_seeds,
            n_episodes=n_episodes,
            max_steps=max_steps,
            env_name=env_name,
            **AGENT_KWARGS,
        )

    return results


# ---------------------------------------------------------------------------
# Top-level experiment suite
# ---------------------------------------------------------------------------

def run_all_experiments(
    env_names,
    algo_names,
    n_episodes,
    n_seeds,
    output_dir,
):
    """Execute the full experiment grid and save plots.

    Experiment dimensions:
    * Environments  : CliffWalking-v1, FrozenLake-v1
    * Algorithms    : Q-learning, SARSA
    * Oracle types  : perfect, noisy (80 %), noisy (50 %), random
    * Help penalties: 0.0, -0.1, -0.5
    * Budgets       : 0 (baseline), 5, 10, 20, unlimited
    """
    os.makedirs(output_dir, exist_ok=True)

    for env_name in env_names:
        env_cfg = ENVIRONMENTS[env_name]
        max_steps = env_cfg["max_steps"]
        env_kwargs = env_cfg["env_kwargs"]

        print(f"\n=== Environment: {env_name} ===")
        print("  Computing optimal policy via value iteration...")
        optimal_policy = compute_optimal_policy(env_name, **env_kwargs)
        n_base_actions = gym.make(env_name, **env_kwargs).action_space.n

        for algo_name in algo_names:
            train_fn = ALGORITHMS[algo_name]
            print(f"\n  Algorithm: {algo_name.upper()}")

            # ------------------------------------------------------------------
            # Experiment 1 – Budget comparison (perfect oracle, no penalty)
            # ------------------------------------------------------------------
            print("\n  [Exp 1] Budget comparison (perfect oracle, no penalty)")
            results_budgets = {}
            oracle = make_oracle("perfect", optimal_policy, n_base_actions)
            # baseline
            env_factory = make_env_factory(
                env_name, oracle, budget=0, penalty=0.0, env_kwargs=env_kwargs
            )
            results_budgets["no oracle"] = run_seeds(
                train_fn, env_factory, n_seeds=n_seeds,
                n_episodes=n_episodes, max_steps=max_steps, env_name=env_name,
                **AGENT_KWARGS,
            )
            for budget in [5, 10, 20, None]:
                oracle = make_oracle("perfect", optimal_policy, n_base_actions)
                lbl = f"budget={_budget_label(budget)}"
                print(f"    {lbl}")
                env_factory = make_env_factory(
                    env_name, oracle, budget=budget, penalty=0.0,
                    env_kwargs=env_kwargs
                )
                results_budgets[lbl] = run_seeds(
                    train_fn, env_factory, n_seeds=n_seeds,
                    n_episodes=n_episodes, max_steps=max_steps,
                    env_name=env_name, **AGENT_KWARGS,
                )
            plot_multi_panel(
                results_budgets,
                env_name=env_name.replace("-", "_"),
                algo_name=f"{algo_name}_budget_comparison",
                output_dir=output_dir,
            )

            # ------------------------------------------------------------------
            # Experiment 2 – Oracle quality comparison (fixed budget=10, no penalty)
            # ------------------------------------------------------------------
            print("\n  [Exp 2] Oracle quality comparison (budget=10, no penalty)")
            results_quality = {}
            oracle = make_oracle("perfect", optimal_policy, n_base_actions)
            env_factory = make_env_factory(
                env_name, oracle, budget=0, penalty=0.0, env_kwargs=env_kwargs
            )
            results_quality["no oracle"] = run_seeds(
                train_fn, env_factory, n_seeds=n_seeds,
                n_episodes=n_episodes, max_steps=max_steps, env_name=env_name,
                **AGENT_KWARGS,
            )
            for oracle_type, accuracy in ORACLE_TYPES:
                oracle = make_oracle(oracle_type, optimal_policy, n_base_actions,
                                     accuracy=accuracy)
                lbl = (
                    oracle_type
                    if oracle_type in ("perfect", "random")
                    else f"noisy({accuracy:.0%})"
                )
                print(f"    {lbl}")
                env_factory = make_env_factory(
                    env_name, oracle, budget=10, penalty=0.0,
                    env_kwargs=env_kwargs
                )
                results_quality[lbl] = run_seeds(
                    train_fn, env_factory, n_seeds=n_seeds,
                    n_episodes=n_episodes, max_steps=max_steps,
                    env_name=env_name, **AGENT_KWARGS,
                )
            plot_multi_panel(
                results_quality,
                env_name=env_name.replace("-", "_"),
                algo_name=f"{algo_name}_oracle_quality",
                output_dir=output_dir,
            )

            # ------------------------------------------------------------------
            # Experiment 3 – Help-penalty comparison (perfect oracle, budget=10)
            # ------------------------------------------------------------------
            print("\n  [Exp 3] Help-penalty comparison (perfect oracle, budget=10)")
            results_penalty = {}
            oracle = make_oracle("perfect", optimal_policy, n_base_actions)
            env_factory = make_env_factory(
                env_name, oracle, budget=0, penalty=0.0, env_kwargs=env_kwargs
            )
            results_penalty["no oracle"] = run_seeds(
                train_fn, env_factory, n_seeds=n_seeds,
                n_episodes=n_episodes, max_steps=max_steps, env_name=env_name,
                **AGENT_KWARGS,
            )
            for penalty in HELP_PENALTIES:
                oracle = make_oracle("perfect", optimal_policy, n_base_actions)
                lbl = f"penalty={penalty}"
                print(f"    {lbl}")
                env_factory = make_env_factory(
                    env_name, oracle, budget=10, penalty=penalty,
                    env_kwargs=env_kwargs
                )
                results_penalty[lbl] = run_seeds(
                    train_fn, env_factory, n_seeds=n_seeds,
                    n_episodes=n_episodes, max_steps=max_steps,
                    env_name=env_name, **AGENT_KWARGS,
                )
            plot_multi_panel(
                results_penalty,
                env_name=env_name.replace("-", "_"),
                algo_name=f"{algo_name}_penalty_comparison",
                output_dir=output_dir,
            )

    print(f"\nAll experiments complete. Figures saved to '{output_dir}/'.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Limited Oracle Budget experiments"
    )
    parser.add_argument(
        "--env",
        default=None,
        choices=list(ENVIRONMENTS.keys()),
        help="Run experiments on a single environment (default: all)",
    )
    parser.add_argument(
        "--algo",
        default=None,
        choices=list(ALGORITHMS.keys()),
        help="Run a single algorithm (default: all)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2000,
        help="Number of training episodes per run (default: 2000)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of random seeds to average over (default: 3)",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for figures (default: results/)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test: 300 episodes, 2 seeds",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    env_names = [args.env] if args.env else list(ENVIRONMENTS.keys())
    algo_names = [args.algo] if args.algo else list(ALGORITHMS.keys())
    n_episodes = 300 if args.quick else args.episodes
    n_seeds = 2 if args.quick else args.seeds

    print("=" * 60)
    print("Limited Oracle Budget – Experiment Runner")
    print("=" * 60)
    print(f"Environments : {env_names}")
    print(f"Algorithms   : {algo_names}")
    print(f"Episodes     : {n_episodes}")
    print(f"Seeds        : {n_seeds}")
    print(f"Output dir   : {args.output}")
    print("=" * 60)

    t0 = time.time()
    run_all_experiments(
        env_names=env_names,
        algo_names=algo_names,
        n_episodes=n_episodes,
        n_seeds=n_seeds,
        output_dir=args.output,
    )
    print(f"\nTotal wall-clock time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
