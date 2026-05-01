"""
Oracle policy implementations.

Three oracle variants are provided:
- PerfectOracle  : always recommends the optimal action.
- NoisyOracle    : recommends the optimal action with probability ``accuracy``,
                   otherwise a uniformly random action.
- RandomOracle   : always recommends a uniformly random action.
"""

import numpy as np


class PerfectOracle:
    def __init__(self, optimal_policy):
        self.optimal_policy = np.asarray(optimal_policy, dtype=int)

    def get_action(self, state):
        """Return the optimal action for *state*."""
        return int(self.optimal_policy[state])


class NoisyOracle:
    def __init__(self, optimal_policy, n_actions, accuracy=0.8, seed=None):
        self.optimal_policy = np.asarray(optimal_policy, dtype=int)
        self.n_actions = n_actions
        self.accuracy = accuracy
        self.rng = np.random.default_rng(seed)

    def get_action(self, state):
        """Return the oracle's recommended action for *state*."""
        if self.rng.random() < self.accuracy:
            return int(self.optimal_policy[state])
        return int(self.rng.integers(self.n_actions))


class RandomOracle:
    def __init__(self, n_actions, seed=None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def get_action(self, state):  # noqa: ARG002
        """Return a random action (state is ignored)."""
        return int(self.rng.integers(self.n_actions))
