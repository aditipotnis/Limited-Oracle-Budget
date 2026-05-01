import numpy as np


class EpisodeMetrics:

    def __init__(self):
        self.rewards = []
        self.successes = []
        self.help_counts = []
        self.step_counts = []

    def record(self, total_reward, success, help_count, step_count):
        self.rewards.append(float(total_reward))
        self.successes.append(bool(success))
        self.help_counts.append(int(help_count))
        self.step_counts.append(int(step_count))

    def smoothed_rewards(self, window=100):
        rewards = np.array(self.rewards)
        if len(rewards) < window:
            return rewards
        kernel = np.ones(window) / window
        half = window // 2
        padded = np.pad(rewards, (half, half), mode="edge")
        return np.convolve(padded, kernel, mode="valid")[: len(rewards)]

    def success_rate(self, last_n=100):
        recent = self.successes[-last_n:]
        return float(np.mean(recent)) if recent else 0.0

    def mean_help_fraction(self, last_n=100):
        recent_help = self.help_counts[-last_n:]
        recent_steps = self.step_counts[-last_n:]
        if not recent_steps or sum(recent_steps) == 0:
            return 0.0
        return float(sum(recent_help) / sum(recent_steps))

    def __len__(self):
        return len(self.rewards)
