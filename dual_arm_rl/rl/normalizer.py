"""Running mean/std statistics for observation and reward normalisation."""
from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Parallel (Chan et al.) running mean and variance."""

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        batch_mean, batch_var, batch_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d: dict):
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])


class ObsNormalizer:
    """z-score observations with running statistics, clipped."""

    def __init__(self, obs_dim: int, clip: float = 10.0):
        self.rms = RunningMeanStd((obs_dim,))
        self.clip = clip

    def __call__(self, obs: np.ndarray, update: bool = False) -> np.ndarray:
        if update:
            self.rms.update(np.atleast_2d(obs))
        z = (obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return self.rms.state_dict()

    def load_state_dict(self, d):
        self.rms.load_state_dict(d)


class RewardScaler:
    """Divide rewards by the running std of the discounted return (per env)."""

    def __init__(self, n_envs: int, gamma: float, clip: float = 10.0):
        self.rms = RunningMeanStd(())
        self.ret = np.zeros(n_envs)
        self.gamma = gamma
        self.clip = clip

    def __call__(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        self.ret = self.ret * self.gamma + reward
        self.rms.update(self.ret.reshape(-1, 1).ravel())
        self.ret[done] = 0.0
        return np.clip(reward / np.sqrt(self.rms.var + 1e-8), -self.clip, self.clip)

    def state_dict(self):
        return self.rms.state_dict()

    def load_state_dict(self, d):
        self.rms.load_state_dict(d)
