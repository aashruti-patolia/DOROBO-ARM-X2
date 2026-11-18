"""
Actor-critic network for PPO.

Separate actor and critic MLPs (sharing a trunk couples the value and policy
gradients, which tends to hurt PPO in continuous control). The actor outputs
the mean of a diagonal Gaussian; the log standard deviation is a learned,
state-independent parameter. The skill being executed (GRASP / PLACE) is part
of the observation, so one network learns both skills.

References: Schulman et al. 2017 (PPO); Andrychowicz et al. 2021, "What
matters in on-policy RL" (separate networks, orthogonal init, small last layer).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _init_orthogonal(module: nn.Module, gain: float = np.sqrt(2)):
    """Orthogonal weights, zero bias."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


def _mlp(in_dim: int, hidden: tuple, out_dim: int, out_gain: float) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [_init_orthogonal(nn.Linear(d, h)), nn.Tanh()]
        d = h
    layers.append(_init_orthogonal(nn.Linear(d, out_dim), gain=out_gain))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(256, 256), init_log_std: float = -0.5):
        super().__init__()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.actor = _mlp(obs_dim, tuple(hidden_sizes), action_dim, out_gain=0.01)
        self.critic = _mlp(obs_dim, tuple(hidden_sizes), 1, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = torch.exp(self.log_std.clamp(-5.0, 1.0)).expand_as(mean)
        return Normal(mean, std)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None,
                             deterministic: bool = False):
        """Sample (or evaluate) an action; returns action, log_prob, entropy, value."""
        dist = self.distribution(obs)
        if action is None:
            action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.get_value(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        dist = self.distribution(obs)
        return dist.mean if deterministic else dist.sample()
