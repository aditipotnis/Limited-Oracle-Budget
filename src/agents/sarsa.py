import numpy as np


class SARSAAgent:
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
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, next_action, done):
        next_q = 0.0 if done else float(self.Q[next_state, next_action])
        td_target = reward + self.gamma * next_q
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        return np.argmax(self.Q, axis=1)

    def reset(self):
        self.Q[:] = 0.0
        self.epsilon = 1.0
