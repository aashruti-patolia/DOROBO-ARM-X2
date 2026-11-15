"""
Skill-Conditioned Actor-Critic Network for Bimanual PPO
=========================================================

Implements a PyTorch PPO Actor-Critic architecture that is conditioned
on both the current manipulation skill_id and the active_arm_id.

Architecture:
  - Shared MLP backbone with LayerNorm
  - Separate Actor head (continuous Gaussian policy)
  - Separate Critic head (state-value function V(s))
  - Orthogonal initialization on all linear layers
  - Continuous action space with learnable log-std

References:
  - PPO (Schulman et al., 2017)
  - Orthogonal init (Saxe et al., 2013)
  - Layer Normalization (Ba et al., 2016)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple


def _init_orthogonal(module: nn.Module, gain: float = np.sqrt(2)):
    """Apply orthogonal initialization to a linear layer."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


class SkillConditionedActorCritic(nn.Module):
    """PPO Actor-Critic network conditioned on skill_id and active_arm_id.

    The observation vector is expected to contain:
      - skill_id (3,) one-hot: [PICK, GRASP, DROP]
      - arm_id (2,) one-hot: [LEFT, RIGHT]
      - proprioception: joint_pos (7,), joint_vel (7,), ee_pos (3,), ee_quat (4,), gripper_state (1,)
      - exteroception: ball_ee_vec (3,), ball_vel (3,), ball_basket_vec (3,), basket_clearance (1,)
      - prev_action (7,)

    Action space: 7 continuous dims = delta_pos (3,) + delta_orient (3,) + gripper (1,)
    All outputs are in [-1, 1] (scaled externally).

    Parameters
    ----------
    obs_dim : int
        Total flattened observation dimensionality.
    action_dim : int
        Number of continuous action dimensions (default: 7).
    hidden_sizes : tuple
        Sizes of shared backbone hidden layers.
    actor_hidden : int
        Size of actor-specific hidden layer.
    critic_hidden : int
        Size of critic-specific hidden layer.
    init_log_std : float
        Initial value for the log standard deviation parameter.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 7,
        hidden_sizes: tuple[int, ...] = (256, 256),
        actor_hidden: int = 128,
        critic_hidden: int = 128,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # ---- Shared backbone ----
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(_init_orthogonal(nn.Linear(in_dim, h)))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.backbone_out_dim = in_dim

        # ---- Actor head (Gaussian policy) ----
        self.actor_head = nn.Sequential(
            _init_orthogonal(nn.Linear(self.backbone_out_dim, actor_hidden)),
            nn.LayerNorm(actor_hidden),
            nn.ReLU(),
            _init_orthogonal(nn.Linear(actor_hidden, action_dim), gain=0.01),
        )

        # Learnable log standard deviation (state-independent)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), init_log_std, dtype=torch.float32)
        )

        # ---- Critic head (value function) ----
        self.critic_head = nn.Sequential(
            _init_orthogonal(nn.Linear(self.backbone_out_dim, critic_hidden)),
            nn.LayerNorm(critic_hidden),
            nn.ReLU(),
            _init_orthogonal(nn.Linear(critic_hidden, 1), gain=1.0),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean and state value.

        Parameters
        ----------
        obs : Tensor (batch, obs_dim)

        Returns
        -------
        action_mean : Tensor (batch, action_dim)
        value : Tensor (batch, 1)
        """
        features = self.backbone(obs)
        action_mean = self.actor_head(features)
        value = self.critic_head(features)
        return action_mean, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action (or evaluate a given action) and return value + log_prob.

        Parameters
        ----------
        obs : Tensor (batch, obs_dim)
        action : Tensor (batch, action_dim), optional
            If provided, compute log_prob of this action instead of sampling.
        deterministic : bool
            If True, use the mean action (no sampling).

        Returns
        -------
        action : Tensor (batch, action_dim)
        log_prob : Tensor (batch,)
        entropy : Tensor (batch,)
        value : Tensor (batch, 1)
        """
        features = self.backbone(obs)
        action_mean = self.actor_head(features)
        value = self.critic_head(features)

        # Build Gaussian distribution
        std = torch.exp(self.log_std.clamp(-5, 2))
        dist = Normal(action_mean, std)

        if action is None:
            if deterministic:
                action = action_mean
            else:
                action = dist.rsample()

        # Sum log-probs across action dimensions
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute state value only (no actor forward pass).

        Parameters
        ----------
        obs : Tensor (batch, obs_dim)

        Returns
        -------
        value : Tensor (batch, 1)
        """
        features = self.backbone(obs)
        return self.critic_head(features)

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the mean action (for evaluation/deployment).

        Parameters
        ----------
        obs : Tensor (batch, obs_dim) or (obs_dim,)

        Returns
        -------
        action : Tensor (same leading shape, action_dim)
        """
        squeeze = obs.dim() == 1
        if squeeze:
            obs = obs.unsqueeze(0)

        features = self.backbone(obs)
        action_mean = self.actor_head(features)

        if squeeze:
            action_mean = action_mean.squeeze(0)
        return action_mean

