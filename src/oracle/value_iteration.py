"""Value iteration for computing optimal policies in discrete environments."""

import numpy as np


def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000):
    """Run value iteration on a Gymnasium discrete environment with transition model P.

    Args:
        env: A Gymnasium environment whose unwrapped form exposes ``env.P``,
             ``env.observation_space.n``, and ``env.action_space.n``.
        gamma: Discount factor.
        theta: Convergence threshold (stop when max value change < theta).
        max_iterations: Safety cap on iteration count.

    Returns:
        V (np.ndarray): Optimal state-value function of shape (n_states,).
        policy (np.ndarray): Greedy policy derived from V, shape (n_states,).
    """
    P = env.unwrapped.P
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)

    for _ in range(max_iterations):
        delta = 0.0
        for s in range(n_states):
            v = V[s]
            q_values = _compute_q(P, V, s, n_actions, gamma)
            V[s] = np.max(q_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_values = _compute_q(P, V, s, n_actions, gamma)
        policy[s] = np.argmax(q_values)

    return V, policy


def _compute_q(P, V, state, n_actions, gamma):
    """Compute Q-values for a single state from the transition model."""
    q_values = np.zeros(n_actions)
    for a in range(n_actions):
        for prob, next_state, reward, done in P[state][a]:
            q_values[a] += prob * (reward + gamma * V[next_state] * (1.0 - float(done)))
    return q_values
