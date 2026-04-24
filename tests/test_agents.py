"""Tests for QLearningAgent and SARSAAgent."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.agents.q_learning import QLearningAgent
from src.agents.sarsa import SARSAAgent


N_STATES = 10
N_ACTIONS = 4


# ---------------------------------------------------------------------------
# QLearningAgent tests
# ---------------------------------------------------------------------------

class TestQLearningAgent:
    def _make_agent(self, **kwargs):
        defaults = dict(n_states=N_STATES, n_actions=N_ACTIONS, seed=0)
        defaults.update(kwargs)
        return QLearningAgent(**defaults)

    def test_q_table_initialised_to_zero(self):
        agent = self._make_agent()
        assert np.all(agent.Q == 0.0)

    def test_q_table_shape(self):
        agent = self._make_agent()
        assert agent.Q.shape == (N_STATES, N_ACTIONS)

    def test_select_action_range(self):
        agent = self._make_agent()
        for s in range(N_STATES):
            a = agent.select_action(s)
            assert 0 <= a < N_ACTIONS

    def test_greedy_action_is_argmax(self):
        agent = self._make_agent()
        # Set a known Q-value
        agent.Q[3, 2] = 10.0
        assert agent.select_action(3, greedy=True) == 2

    def test_update_increases_q_for_positive_reward(self):
        agent = self._make_agent(alpha=1.0, gamma=0.0)
        agent.update(state=0, action=0, reward=5.0, next_state=1, done=True)
        assert agent.Q[0, 0] == pytest.approx(5.0)

    def test_update_td_target(self):
        agent = self._make_agent(alpha=0.5, gamma=0.9)
        agent.Q[1, :] = [0.0, 0.0, 2.0, 0.0]  # best next Q = 2.0
        agent.update(state=0, action=0, reward=1.0, next_state=1, done=False)
        # td_target = 1.0 + 0.9 * 2.0 = 2.8; update: 0 + 0.5 * 2.8 = 1.4
        assert agent.Q[0, 0] == pytest.approx(1.4)

    def test_update_done_ignores_next_state(self):
        agent = self._make_agent(alpha=1.0, gamma=0.9)
        agent.Q[1, :] = [100.0, 100.0, 100.0, 100.0]
        agent.update(state=0, action=0, reward=3.0, next_state=1, done=True)
        assert agent.Q[0, 0] == pytest.approx(3.0)

    def test_epsilon_decays(self):
        agent = self._make_agent(epsilon_start=1.0, epsilon_decay=0.5, epsilon_end=0.0)
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.5)
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.25)

    def test_epsilon_never_below_min(self):
        agent = self._make_agent(epsilon_start=1.0, epsilon_decay=0.1, epsilon_end=0.05)
        for _ in range(100):
            agent.decay_epsilon()
        assert agent.epsilon >= 0.05

    def test_get_policy_shape(self):
        agent = self._make_agent()
        policy = agent.get_policy()
        assert policy.shape == (N_STATES,)
        assert all(0 <= a < N_ACTIONS for a in policy)

    def test_reset_clears_q_table(self):
        agent = self._make_agent()
        agent.Q[0, 0] = 99.0
        agent.epsilon = 0.01
        agent.reset()
        assert agent.Q[0, 0] == 0.0
        assert agent.epsilon == 1.0


# ---------------------------------------------------------------------------
# SARSAAgent tests
# ---------------------------------------------------------------------------

class TestSARSAAgent:
    def _make_agent(self, **kwargs):
        defaults = dict(n_states=N_STATES, n_actions=N_ACTIONS, seed=0)
        defaults.update(kwargs)
        return SARSAAgent(**defaults)

    def test_q_table_initialised_to_zero(self):
        agent = self._make_agent()
        assert np.all(agent.Q == 0.0)

    def test_select_action_range(self):
        agent = self._make_agent()
        for s in range(N_STATES):
            a = agent.select_action(s)
            assert 0 <= a < N_ACTIONS

    def test_greedy_action_is_argmax(self):
        agent = self._make_agent()
        agent.Q[5, 1] = 7.0
        assert agent.select_action(5, greedy=True) == 1

    def test_update_uses_next_action_not_max(self):
        """SARSA uses Q[next_state, next_action], not max Q[next_state]."""
        agent = self._make_agent(alpha=1.0, gamma=0.9)
        agent.Q[1, :] = [0.0, 0.0, 2.0, 5.0]  # max = 5 at action 3
        next_action = 2  # deliberately NOT the greedy action (action 3)
        agent.update(
            state=0, action=0, reward=1.0,
            next_state=1, next_action=next_action, done=False
        )
        # td_target = 1.0 + 0.9 * Q[1, 2] = 1.0 + 0.9 * 2.0 = 2.8
        assert agent.Q[0, 0] == pytest.approx(2.8)

    def test_update_done_ignores_next(self):
        agent = self._make_agent(alpha=1.0, gamma=0.9)
        agent.Q[1, :] = [100.0] * N_ACTIONS
        agent.update(
            state=0, action=0, reward=4.0,
            next_state=1, next_action=0, done=True
        )
        assert agent.Q[0, 0] == pytest.approx(4.0)

    def test_epsilon_decays(self):
        agent = self._make_agent(epsilon_start=1.0, epsilon_decay=0.5, epsilon_end=0.0)
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.5)

    def test_epsilon_never_below_min(self):
        agent = self._make_agent(epsilon_start=1.0, epsilon_decay=0.1, epsilon_end=0.05)
        for _ in range(100):
            agent.decay_epsilon()
        assert agent.epsilon >= 0.05

    def test_get_policy_shape(self):
        agent = self._make_agent()
        policy = agent.get_policy()
        assert policy.shape == (N_STATES,)

    def test_reset_clears_q_table(self):
        agent = self._make_agent()
        agent.Q[3, 2] = 42.0
        agent.reset()
        assert agent.Q[3, 2] == 0.0
