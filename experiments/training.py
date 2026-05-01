import numpy as np
import gymnasium as gym

from src.agents.q_learning import QLearningAgent
from src.agents.sarsa import SARSAAgent
from src.environments.oracle_env import OracleEnv
from src.oracle.oracle import NoisyOracle, PerfectOracle, RandomOracle
from src.oracle.value_iteration import value_iteration
from src.utils.metrics import EpisodeMetrics


# ---------------------------------------------------------------------------
# Optimal-policy computation
# ---------------------------------------------------------------------------

def compute_optimal_policy(env_name, gamma=0.99, **env_kwargs):
    env = gym.make(env_name, **env_kwargs)
    _, policy = value_iteration(env, gamma=gamma)
    env.close()
    return policy


# ---------------------------------------------------------------------------
# Oracle factory
# ---------------------------------------------------------------------------

def make_oracle(oracle_type, optimal_policy, n_actions, accuracy=0.8, seed=None):
    if oracle_type == "perfect":
        return PerfectOracle(optimal_policy)
    if oracle_type == "noisy":
        return NoisyOracle(optimal_policy, n_actions, accuracy=accuracy, seed=seed)
    if oracle_type == "random":
        return RandomOracle(n_actions, seed=seed)
    raise ValueError(f"Unknown oracle_type: {oracle_type!r}")


# ---------------------------------------------------------------------------
# Single-run training functions
# ---------------------------------------------------------------------------

def _detect_success(env_name, info, terminated):
    # FrozenLake marks success with info['is_success'] == 1
    if "is_success" in info:
        return bool(info["is_success"])
    # For CliffWalking terminated without info key means reaching the goal
    return terminated


def train_qlearning(
    env,
    n_episodes,
    max_steps,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    seed=None,
    env_name="",
):
    is_oracle = isinstance(env, OracleEnv)
    n_states = env.n_states if is_oracle else env.observation_space.n
    n_actions = env.n_actions if is_oracle else env.action_space.n

    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )
    metrics = EpisodeMetrics()

    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        help_count = 0
        step = 0
        success = False

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            if info.get("used_help", False):
                help_count += 1
            total_reward += reward
            state = next_state

            if done:
                success = _detect_success(env_name, info, terminated)
                break

        agent.decay_epsilon()
        metrics.record(total_reward, success, help_count, step + 1)

    return metrics


def train_sarsa(
    env,
    n_episodes,
    max_steps,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    seed=None,
    env_name="",
):
    is_oracle = isinstance(env, OracleEnv)
    n_states = env.n_states if is_oracle else env.observation_space.n
    n_actions = env.n_actions if is_oracle else env.action_space.n

    agent = SARSAAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )
    metrics = EpisodeMetrics()

    for ep in range(n_episodes):
        state, _ = env.reset()
        action = agent.select_action(state)
        total_reward = 0.0
        help_count = 0
        step = 0
        success = False

        for step in range(max_steps):
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_action = agent.select_action(next_state)

            agent.update(state, action, reward, next_state, next_action, done)

            if info.get("used_help", False):
                help_count += 1
            total_reward += reward
            state = next_state
            action = next_action

            if done:
                success = _detect_success(env_name, info, terminated)
                break

        agent.decay_epsilon()
        metrics.record(total_reward, success, help_count, step + 1)

    return metrics


# ---------------------------------------------------------------------------
# Multi-seed averaging
# ---------------------------------------------------------------------------

def run_seeds(train_fn, env_factory, n_seeds, **train_kwargs):
    all_rewards = []
    all_successes = []
    all_help_counts = []
    all_step_counts = []

    for seed in range(n_seeds):
        env = env_factory()
        kw = dict(train_kwargs)
        kw["seed"] = seed
        m = train_fn(env, **kw)
        env.close()
        all_rewards.append(m.rewards)
        all_successes.append(m.successes)
        all_help_counts.append(m.help_counts)
        all_step_counts.append(m.step_counts)

    avg = EpisodeMetrics()
    n_ep = len(all_rewards[0])
    for i in range(n_ep):
        avg.record(
            total_reward=float(np.mean([r[i] for r in all_rewards])),
            success=float(np.mean([s[i] for s in all_successes])) >= 0.5,
            help_count=int(round(np.mean([h[i] for h in all_help_counts]))),
            step_count=int(round(np.mean([s[i] for s in all_step_counts]))),
        )
    return avg
