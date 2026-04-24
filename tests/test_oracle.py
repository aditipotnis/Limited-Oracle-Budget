"""Tests for oracle implementations and value iteration."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import pytest

from src.oracle.oracle import NoisyOracle, PerfectOracle, RandomOracle
from src.oracle.value_iteration import value_iteration


N_STATES = 16
N_ACTIONS = 4

# A simple fixed policy used across oracle tests
FIXED_POLICY = np.array([i % N_ACTIONS for i in range(N_STATES)], dtype=int)


# ---------------------------------------------------------------------------
# PerfectOracle
# ---------------------------------------------------------------------------

class TestPerfectOracle:
    def test_returns_policy_action(self):
        oracle = PerfectOracle(FIXED_POLICY)
        for s in range(N_STATES):
            assert oracle.get_action(s) == FIXED_POLICY[s]

    def test_return_type_is_int(self):
        oracle = PerfectOracle(FIXED_POLICY)
        assert isinstance(oracle.get_action(0), int)


# ---------------------------------------------------------------------------
# NoisyOracle
# ---------------------------------------------------------------------------

class TestNoisyOracle:
    def test_accuracy_one_always_correct(self):
        oracle = NoisyOracle(FIXED_POLICY, N_ACTIONS, accuracy=1.0, seed=0)
        for s in range(N_STATES):
            assert oracle.get_action(s) == FIXED_POLICY[s]

    def test_accuracy_zero_never_correct(self):
        """With accuracy=0 the oracle should match the policy very rarely."""
        oracle = NoisyOracle(FIXED_POLICY, N_ACTIONS, accuracy=0.0, seed=0)
        # With accuracy 0.0 actions are random; on a 4-action space we'd
        # expect ~25% to accidentally match.  Over 2000 calls the chance of
        # *always* matching is essentially 0 (checking at least one mismatch).
        actions = [oracle.get_action(0) for _ in range(2000)]
        # At least some actions must differ from FIXED_POLICY[0]
        assert any(a != FIXED_POLICY[0] for a in actions)

    def test_accuracy_80_approx(self):
        """Actions should match the policy roughly 80% of the time."""
        policy = np.zeros(N_STATES, dtype=int)  # all actions 0
        oracle = NoisyOracle(policy, N_ACTIONS, accuracy=0.8, seed=42)
        n_trials = 5000
        matches = sum(oracle.get_action(0) == 0 for _ in range(n_trials))
        observed = matches / n_trials
        # Allow ±8% tolerance (statistical variation with 5000 trials)
        assert 0.72 <= observed <= 0.88, f"Observed accuracy {observed:.2%}"

    def test_return_type_is_int(self):
        oracle = NoisyOracle(FIXED_POLICY, N_ACTIONS, accuracy=0.8, seed=0)
        assert isinstance(oracle.get_action(0), int)

    def test_action_in_range(self):
        oracle = NoisyOracle(FIXED_POLICY, N_ACTIONS, accuracy=0.5, seed=0)
        for _ in range(100):
            a = oracle.get_action(np.random.randint(N_STATES))
            assert 0 <= a < N_ACTIONS


# ---------------------------------------------------------------------------
# RandomOracle
# ---------------------------------------------------------------------------

class TestRandomOracle:
    def test_action_in_range(self):
        oracle = RandomOracle(N_ACTIONS, seed=0)
        for _ in range(500):
            a = oracle.get_action(0)
            assert 0 <= a < N_ACTIONS

    def test_all_actions_used(self):
        oracle = RandomOracle(N_ACTIONS, seed=0)
        actions = {oracle.get_action(0) for _ in range(500)}
        assert actions == set(range(N_ACTIONS))

    def test_return_type_is_int(self):
        oracle = RandomOracle(N_ACTIONS, seed=0)
        assert isinstance(oracle.get_action(0), int)


# ---------------------------------------------------------------------------
# Value iteration
# ---------------------------------------------------------------------------

class TestValueIteration:
    def test_frozen_lake_converges(self):
        env = gym.make("FrozenLake-v1", is_slippery=False)
        V, policy = value_iteration(env, gamma=0.99)
        env.close()
        assert V.shape == (16,)
        assert policy.shape == (16,)
        assert all(0 <= a < 4 for a in policy)

    def test_frozen_lake_policy_reaches_goal(self):
        """Simulate the greedy policy in the non-slippery FrozenLake.

        The optimal policy should always reach the goal from state 0."""
        env = gym.make("FrozenLake-v1", is_slippery=False)
        _, policy = value_iteration(env, gamma=0.99)

        sim_env = gym.make("FrozenLake-v1", is_slippery=False)
        obs, _ = sim_env.reset(seed=0)
        reached_goal = False
        for _ in range(100):
            action = int(policy[obs])
            obs, reward, terminated, truncated, _ = sim_env.step(action)
            if terminated:
                reached_goal = reward > 0
                break
        sim_env.close()
        env.close()
        assert reached_goal

    def test_cliff_walking_converges(self):
        env = gym.make("CliffWalking-v1")
        V, policy = value_iteration(env, gamma=0.99)
        env.close()
        assert V.shape == (48,)
        assert policy.shape == (48,)
        assert all(0 <= a < 4 for a in policy)

    def test_value_of_goal_state_is_nonnegative(self):
        """Goal/terminal states should have non-negative values."""
        env = gym.make("FrozenLake-v1", is_slippery=False)
        V, _ = value_iteration(env, gamma=0.99)
        env.close()
        # In FrozenLake the goal is state 15 (bottom-right)
        assert V[15] >= 0.0

    def test_custom_gamma(self):
        env = gym.make("FrozenLake-v1", is_slippery=False)
        V_high, _ = value_iteration(env, gamma=0.99)
        V_low, _ = value_iteration(env, gamma=0.5)
        env.close()
        # Higher gamma should yield higher or equal values overall
        assert np.sum(V_high) >= np.sum(V_low)
