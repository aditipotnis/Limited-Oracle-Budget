"""Tabular SARSA agent for discrete state-action spaces."""

import numpy as np


class SARSAAgent:
    """On-policy tabular SARSA agent.

    Maintains a Q-table of shape ``(n_states, n_actions)`` and uses an
    epsilon-greedy exploration strategy with geometric decay.  Unlike
    Q-learning, the update uses the *actual* next action chosen by the policy
    rather than the greedy maximum.

    Args:
        n_states (int): Number of discrete states.
        n_actions (int): Number of discrete actions.
        alpha (float): Learning rate in (0, 1].
        gamma (float): Discount factor in [0, 1].
        epsilon_start (float): Initial exploration probability.
        epsilon_end (float): Minimum exploration probability.
        epsilon_decay (float): Multiplicative decay factor applied per episode.
        seed (int | None): Optional random seed for reproducibility.
    """

    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        seed=None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state, greedy=False):
        """Select an action using epsilon-greedy policy.

        Args:
            state (int): Current state index.
            greedy (bool): If True, always choose the greedy action.

        Returns:
            int: Selected action index.
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, next_action, done):
        """Apply a single SARSA (on-policy TD) update.

        Args:
            state (int): State in which the action was taken.
            action (int): Action taken.
            reward (float): Observed reward.
            next_state (int): State reached after the action.
            next_action (int): Action selected in *next_state* (on-policy).
            done (bool): Whether the episode ended.
        """
        next_q = 0.0 if done else float(self.Q[next_state, next_action])
        td_target = reward + self.gamma * next_q
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        """Apply one step of epsilon decay (call once per episode)."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        """Return the greedy policy as an integer array of shape (n_states,)."""
        return np.argmax(self.Q, axis=1)

    def reset(self):
        """Re-initialize the Q-table and reset epsilon to its starting value."""
        self.Q[:] = 0.0
        self.epsilon = 1.0
