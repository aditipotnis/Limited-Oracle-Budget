"""Metrics tracking utilities for RL experiments."""

import numpy as np


class EpisodeMetrics:
    """Records per-episode statistics for a single training run.

    Attributes:
        rewards (list[float]): Total undiscounted reward per episode.
        successes (list[bool]): Whether the agent reached the goal each episode.
        help_counts (list[int]): Number of help actions taken per episode.
        step_counts (list[int]): Number of steps taken per episode.
    """

    def __init__(self):
        self.rewards = []
        self.successes = []
        self.help_counts = []
        self.step_counts = []

    def record(self, total_reward, success, help_count, step_count):
        """Append statistics for one completed episode.

        Args:
            total_reward (float): Cumulative undiscounted episode reward.
            success (bool): True if the goal was reached.
            help_count (int): Number of help actions used this episode.
            step_count (int): Number of environment steps this episode.
        """
        self.rewards.append(float(total_reward))
        self.successes.append(bool(success))
        self.help_counts.append(int(help_count))
        self.step_counts.append(int(step_count))

    def smoothed_rewards(self, window=100):
        """Return a moving-average of episode rewards.

        Args:
            window (int): Size of the smoothing window.

        Returns:
            np.ndarray: Smoothed reward array of the same length as
                ``self.rewards``.
        """
        rewards = np.array(self.rewards)
        if len(rewards) < window:
            return rewards
        kernel = np.ones(window) / window
        return np.convolve(rewards, kernel, mode="same")

    def success_rate(self, last_n=100):
        """Fraction of the last *last_n* episodes in which the goal was reached.

        Args:
            last_n (int): Number of recent episodes to average over.

        Returns:
            float: Success rate in [0, 1].
        """
        recent = self.successes[-last_n:]
        return float(np.mean(recent)) if recent else 0.0

    def mean_help_fraction(self, last_n=100):
        """Mean fraction of steps in which help was used, over the last *last_n* episodes.

        Args:
            last_n (int): Number of recent episodes to average over.

        Returns:
            float: Mean help fraction in [0, 1].
        """
        recent_help = self.help_counts[-last_n:]
        recent_steps = self.step_counts[-last_n:]
        if not recent_steps or sum(recent_steps) == 0:
            return 0.0
        return float(sum(recent_help) / sum(recent_steps))

    def __len__(self):
        return len(self.rewards)
