"""OracleEnv – a Gymnasium wrapper that augments a discrete environment with
a limited oracle help mechanism.

Design
------
* The base action space is extended with one additional *help* action
  (index n_base_actions).
* The observation is augmented to encode the remaining help budget:
    encoded_state = env_obs * (max_budget + 1) + remaining_budget
* When the agent takes the *help* action:
    - If budget > 0: the oracle is queried, its recommended base action is
      executed, one budget unit is consumed, and help_penalty is added to
      the step reward.
    - If budget == 0: a no_budget_penalty is applied to the reward and a
      random base action is executed (the agent wasted a turn).
* For the *unlimited budget* case set max_budget=None.  Budget is never
  depleted; the observation is just the raw environment observation.
"""

import gymnasium as gym
import numpy as np


class OracleEnv:
    UNLIMITED = None  # sentinel for unlimited budget

    def __init__(
        self,
        env_name,
        oracle,
        max_budget,
        help_penalty=0.0,
        no_budget_penalty=-1.0,
        **env_kwargs,
    ):
        self.base_env = gym.make(env_name, **env_kwargs)
        self.oracle = oracle
        self.max_budget = max_budget
        self.help_penalty = help_penalty
        self.no_budget_penalty = no_budget_penalty

        self.unlimited = max_budget is None

        self.n_base_actions = self.base_env.action_space.n
        self.n_base_states = self.base_env.observation_space.n
        # Action space: base actions + 1 help action
        self.n_actions = self.n_base_actions + 1
        self.help_action = self.n_base_actions

        # State-space size
        if self.unlimited:
            self.n_states = self.n_base_states
        else:
            self.n_states = self.n_base_states * (self.max_budget + 1)

        self._current_obs = None
        self._remaining_budget = 0

    # ------------------------------------------------------------------
    # State encoding helpers
    # ------------------------------------------------------------------

    def encode_state(self, obs, budget):
        if self.unlimited:
            return int(obs)
        return int(obs) * (self.max_budget + 1) + int(budget)

    def decode_state(self, state):
        if self.unlimited:
            return int(state), None
        budget = state % (self.max_budget + 1)
        obs = state // (self.max_budget + 1)
        return int(obs), int(budget)

    # ------------------------------------------------------------------
    # Gymnasium-like interface
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        obs, info = self.base_env.reset(seed=seed)
        self._current_obs = int(obs)
        self._remaining_budget = 0 if self.unlimited else self.max_budget
        return self.encode_state(self._current_obs, self._remaining_budget), info

    def step(self, action):
        if action == self.help_action:
            if self.unlimited or self._remaining_budget > 0:
                oracle_action = self.oracle.get_action(self._current_obs)
                if not self.unlimited:
                    self._remaining_budget -= 1
                obs, reward, terminated, truncated, info = self.base_env.step(
                    oracle_action
                )
                reward = float(reward) + self.help_penalty
                info["used_help"] = True
                info["oracle_action"] = oracle_action
            else:
                fallback_action = int(self.base_env.action_space.sample())
                obs, reward, terminated, truncated, info = self.base_env.step(
                    fallback_action
                )
                reward = float(reward) + self.no_budget_penalty
                info["used_help"] = False
                info["no_budget"] = True
        else:
            obs, reward, terminated, truncated, info = self.base_env.step(action)
            reward = float(reward)
            info["used_help"] = False

        self._current_obs = int(obs)
        state = self.encode_state(self._current_obs, self._remaining_budget)
        return state, reward, terminated, truncated, info

    def close(self):
        self.base_env.close()

    # ------------------------------------------------------------------
    # Properties forwarded from base env
    # ------------------------------------------------------------------

    @property
    def observation_space(self):
        return self.base_env.observation_space

    @property
    def action_space(self):
        return self.base_env.action_space
