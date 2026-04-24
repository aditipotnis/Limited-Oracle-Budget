"""Tests for OracleEnv."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.environments.oracle_env import OracleEnv
from src.oracle.oracle import PerfectOracle, RandomOracle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_frozen_oracle():
    """Return a PerfectOracle with a fixed policy for FrozenLake-v1 (4x4)."""
    # FrozenLake-v1 has 16 states, 4 actions
    policy = np.zeros(16, dtype=int)
    return PerfectOracle(policy)


def _make_env(budget, unlimited=False, penalty=0.0, no_budget_penalty=-1.0):
    oracle = _make_frozen_oracle()
    return OracleEnv(
        env_name="FrozenLake-v1",
        oracle=oracle,
        max_budget=None if unlimited else budget,
        help_penalty=penalty,
        no_budget_penalty=no_budget_penalty,
        is_slippery=False,
    )


# ---------------------------------------------------------------------------
# State encoding / decoding
# ---------------------------------------------------------------------------

class TestStateEncoding:
    def test_encode_decode_roundtrip(self):
        env = _make_env(budget=5)
        for obs in range(16):
            for b in range(6):  # 0..5
                state = env.encode_state(obs, b)
                decoded_obs, decoded_b = env.decode_state(state)
                assert decoded_obs == obs
                assert decoded_b == b
        env.close()

    def test_state_space_size(self):
        env = _make_env(budget=10)
        assert env.n_states == 16 * 11  # 16 states * (10+1) budget levels
        env.close()

    def test_unlimited_state_space(self):
        env = _make_env(budget=0, unlimited=True)
        assert env.n_states == 16
        env.close()

    def test_unlimited_encode_decode(self):
        env = _make_env(budget=0, unlimited=True)
        for obs in range(16):
            state = env.encode_state(obs, budget=None)
            decoded_obs, decoded_b = env.decode_state(state)
            assert decoded_obs == obs
            assert decoded_b is None
        env.close()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_encoded_state(self):
        env = _make_env(budget=5)
        state, info = env.reset(seed=0)
        obs, budget = env.decode_state(state)
        assert 0 <= obs < 16
        assert budget == 5
        env.close()

    def test_reset_restores_budget(self):
        env = _make_env(budget=3)
        env.reset(seed=0)
        # Use some budget
        for _ in range(2):
            env.step(env.help_action)
        assert env._remaining_budget == 1
        # Reset should restore
        env.reset(seed=0)
        assert env._remaining_budget == 3
        env.close()

    def test_unlimited_reset(self):
        env = _make_env(budget=0, unlimited=True)
        state, _ = env.reset(seed=0)
        assert 0 <= state < 16
        env.close()


# ---------------------------------------------------------------------------
# Step with regular actions
# ---------------------------------------------------------------------------

class TestRegularActions:
    def test_regular_action_doesnt_consume_budget(self):
        env = _make_env(budget=5)
        env.reset(seed=0)
        _, _, _, _, info = env.step(0)  # move left
        assert env._remaining_budget == 5
        assert info["used_help"] is False
        env.close()

    def test_state_reflects_current_budget(self):
        env = _make_env(budget=5)
        env.reset(seed=0)
        state, _, _, _, _ = env.step(0)
        _, budget = env.decode_state(state)
        assert budget == env._remaining_budget
        env.close()


# ---------------------------------------------------------------------------
# Help action with positive budget
# ---------------------------------------------------------------------------

class TestHelpAction:
    def test_help_consumes_budget(self):
        env = _make_env(budget=3)
        env.reset(seed=0)
        env.step(env.help_action)
        assert env._remaining_budget == 2
        env.close()

    def test_help_sets_used_help_true(self):
        env = _make_env(budget=3)
        env.reset(seed=0)
        _, _, _, _, info = env.step(env.help_action)
        assert info["used_help"] is True
        assert "oracle_action" in info
        env.close()

    def test_help_applies_penalty(self):
        penalty = -0.5
        # We'll compare reward with and without oracle penalty.
        # Make a reference env with penalty 0.
        oracle = _make_frozen_oracle()
        env_ref = OracleEnv("FrozenLake-v1", oracle, max_budget=5,
                             help_penalty=0.0, is_slippery=False)
        env_ref.reset(seed=42)
        _, r_ref, _, _, _ = env_ref.step(env_ref.help_action)

        oracle2 = _make_frozen_oracle()
        env_pen = OracleEnv("FrozenLake-v1", oracle2, max_budget=5,
                             help_penalty=penalty, is_slippery=False)
        env_pen.reset(seed=42)
        _, r_pen, _, _, _ = env_pen.step(env_pen.help_action)

        assert abs(r_pen - (r_ref + penalty)) < 1e-9
        env_ref.close()
        env_pen.close()

    def test_budget_depletes_to_zero(self):
        env = _make_env(budget=2)
        env.reset(seed=0)
        for _ in range(2):
            _, _, done, trunc, _ = env.step(env.help_action)
            if done or trunc:
                env.reset(seed=0)
        assert env._remaining_budget == 0
        env.close()


# ---------------------------------------------------------------------------
# Help action with zero budget (exhausted)
# ---------------------------------------------------------------------------

class TestNoBudget:
    def test_no_budget_penalty_applied(self):
        no_budget_pen = -1.0
        env = _make_env(budget=0, no_budget_penalty=no_budget_pen)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(env.help_action)
        assert info.get("no_budget") is True
        assert info["used_help"] is False
        # Reward should be base_reward + no_budget_penalty
        # (base reward is typically 0 in FrozenLake non-terminal)
        assert reward <= 0.0
        env.close()

    def test_no_budget_doesnt_decrement_below_zero(self):
        env = _make_env(budget=0)
        env.reset(seed=0)
        env.step(env.help_action)
        assert env._remaining_budget == 0
        env.close()


# ---------------------------------------------------------------------------
# Unlimited budget
# ---------------------------------------------------------------------------

class TestUnlimitedBudget:
    def test_unlimited_help_always_works(self):
        env = _make_env(budget=0, unlimited=True)
        env.reset(seed=0)
        for _ in range(10):
            _, _, done, trunc, info = env.step(env.help_action)
            assert info["used_help"] is True
            if done or trunc:
                env.reset(seed=0)
        env.close()

    def test_unlimited_budget_not_decremented(self):
        env = _make_env(budget=0, unlimited=True)
        env.reset(seed=0)
        env.step(env.help_action)
        assert env._remaining_budget == 0  # internal budget field unused
        env.close()


# ---------------------------------------------------------------------------
# Action / state space sizes
# ---------------------------------------------------------------------------

class TestSpaceSizes:
    def test_n_actions_one_more_than_base(self):
        env = _make_env(budget=5)
        # FrozenLake has 4 base actions -> 5 total
        assert env.n_actions == 5
        assert env.help_action == 4
        env.close()
