#!/usr/bin/env python3
"""
PPO Training Pipeline for Bimanual Pick-and-Place
====================================================

CleanRL-style single-file PPO implementation featuring:
  - Generalized Advantage Estimation (GAE)
  - KL-divergence early stopping per update epoch
  - Curriculum scheduling for basket collision penalty
  - Alternating-arm continuous training
  - TensorBoard logging and model checkpointing
  - Domain randomization hooks

Usage:
    python train_ppo.py --total-timesteps 500000 --headless
    python train_ppo.py --total-timesteps 100000 --seed 42 --lr 3e-4

    # Quick smoke test (10k steps):
    python train_ppo.py --total-timesteps 10000 --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure local imports work
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bimanual_env import BimanualPickPlaceEnv, MAX_EPISODE_STEPS
from ppo_network import SkillConditionedActorCritic
from reward_calculator import RewardConfig


# =====================================================================
# Rollout Buffer
# =====================================================================
class RolloutBuffer:
    """Fixed-size buffer for storing PPO rollout data."""

    def __init__(self, n_steps: int, obs_dim: int, action_dim: int):
        self.n_steps = n_steps
        self.obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)
        self.ptr = 0

    def store(self, obs, action, log_prob, reward, done, value):
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.values[idx] = value
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute Generalized Advantage Estimation."""
        last_gae = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            self.advantages[t] = last_gae = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae
            )

        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get_batches(self, batch_size: int, device: torch.device):
        """Yield shuffled mini-batches for SGD updates.

        The rollout is moved to ``device`` once per pass (a zero-copy view
        on CPU) and mini-batches are sliced from those tensors, rather than
        building five new tensors from NumPy for every batch.
        """
        n = self.ptr
        obs = torch.as_tensor(self.obs[:n], device=device)
        actions = torch.as_tensor(self.actions[:n], device=device)
        log_probs = torch.as_tensor(self.log_probs[:n], device=device)
        advantages = torch.as_tensor(self.advantages[:n], device=device)
        returns = torch.as_tensor(self.returns[:n], device=device)

        indices = torch.as_tensor(
            np.random.permutation(n), dtype=torch.long, device=device)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]

            yield (
                obs[batch_idx],
                actions[batch_idx],
                log_probs[batch_idx],
                advantages[batch_idx],
                returns[batch_idx],
            )

    def reset(self):
        self.ptr = 0


# =====================================================================
# PPO Update
# =====================================================================
def ppo_update(
    policy: SkillConditionedActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    *,
    n_epochs: int,
    batch_size: int,
    clip_ratio: float,
    entropy_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    target_kl: float | None,
    device: torch.device,
) -> dict:
    """Perform PPO policy update over the buffer data.

    Returns
    -------
    stats : dict
        Training statistics for logging.
    """
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    n_updates = 0
    early_stopped = False

    # Normalize advantages
    adv = buffer.advantages[:buffer.ptr]
    adv_mean, adv_std = adv.mean(), adv.std() + 1e-8
    buffer.advantages[:buffer.ptr] = (adv - adv_mean) / adv_std

    for epoch in range(n_epochs):
        epoch_kl = 0.0
        epoch_batches = 0

        for obs_b, act_b, old_logp_b, adv_b, ret_b in buffer.get_batches(
            batch_size, device
        ):
            _, new_logp, entropy, values = policy.get_action_and_value(
                obs_b, action=act_b
            )

            # Policy loss (clipped surrogate)
            ratio = torch.exp(new_logp - old_logp_b)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (MSE)
            value_loss = 0.5 * ((values.squeeze(-1) - ret_b) ** 2).mean()

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + vf_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            # Approximate KL divergence
            with torch.no_grad():
                approx_kl = (old_logp_b - new_logp).mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            total_approx_kl += approx_kl
            n_updates += 1
            epoch_kl += approx_kl
            epoch_batches += 1

        # KL early stopping
        avg_epoch_kl = epoch_kl / max(epoch_batches, 1)
        if target_kl is not None and avg_epoch_kl > target_kl:
            early_stopped = True
            break

    n_updates = max(n_updates, 1)
    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entropy / n_updates,
        "approx_kl": total_approx_kl / n_updates,
        "n_epochs_actual": epoch + 1 if not early_stopped else epoch + 1,
        "kl_early_stopped": early_stopped,
    }


# =====================================================================
# Training Loop
# =====================================================================
def train(args):
    """Main PPO training loop."""
    print("=" * 65)
    print("PPO Training Pipeline -- Bimanual Pick-and-Place")
    print("=" * 65)
    print(f"  Total timesteps:  {args.total_timesteps}")
    print(f"  Rollout steps:    {args.n_steps}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  Seed:             {args.seed}")
    print(f"  Device:           {args.device}")
    print("=" * 65)

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    # ---- Create environment ----
    reward_config = RewardConfig()
    env = BimanualPickPlaceEnv(
        model_path=args.model,
        reward_config=reward_config,
        domain_randomization=args.dr_level != "off",
        randomization_level=args.dr_level,
        obs_noise_scale=0.002 if args.dr_level != "off" else 0.0,
        render_mode=None,
    )

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"  Obs dim:          {obs_dim}")
    print(f"  Action dim:       {action_dim}")
    print("=" * 65)

    # ---- Create policy network ----
    policy = SkillConditionedActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=(256, 256),
        actor_hidden=128,
        critic_hidden=128,
        init_log_std=-0.5,
    ).to(device)

    # Fused Adam applies the update for all parameter tensors in one call
    # instead of a Python loop over ~20 small tensors, which is a large part
    # of each SGD step for a network this size. Same algorithm; fall back to
    # the default implementation on torch builds without fused support.
    try:
        optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5, fused=True)
    except (RuntimeError, TypeError):
        optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    # ---- Rollout buffer ----
    buffer = RolloutBuffer(args.n_steps, obs_dim, action_dim)

    # ---- TensorBoard (optional) ----
    writer = None
    if args.log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_path = Path(args.log_dir) / f"ppo_bimanual_seed{args.seed}"
            writer = SummaryWriter(str(log_path))
            print(f"  Logging to:       {log_path}")
        except ImportError:
            print("  [WARN] TensorBoard not available, skipping logging.")

    # ---- Checkpoint directory ----
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training state ----
    global_step = 0
    best_success_rate = 0.0
    episode_rewards = []
    episode_successes = []
    episode_lengths = []
    collision_curriculum_warmup = args.collision_warmup_steps

    obs, info = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode_length = 0

    print("\n[TRAINING] Starting PPO rollout collection...\n")
    t_start = time.time()

    while global_step < args.total_timesteps:
        # ---- Curriculum scheduling ----
        collision_weight = min(1.0, global_step / max(collision_curriculum_warmup, 1))
        env.reward_calc.set_curriculum(collision_weight)

        # ---- Collect rollout ----
        buffer.reset()
        policy.eval()

        for step in range(args.n_steps):
            global_step += 1

            with torch.inference_mode():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, log_prob, _, value = policy.get_action_and_value(obs_t)

                action_np = action.squeeze(0).cpu().numpy()
                log_prob_np = log_prob.item()
                value_np = value.item()

            # Step environment
            next_obs, reward, terminated, truncated, step_info = env.step(action_np)
            done = terminated or truncated

            # A truncation (step limit, ball off the table) is not a real
            # terminal state: the task would have continued. Bootstrap from the
            # value of the final observation so the critic doesn't learn that
            # running out of time is worth zero. GAE treats the step as an
            # episode boundary, so the bootstrap is folded into the reward.
            if truncated and not terminated:
                with torch.inference_mode():
                    final_obs_t = torch.as_tensor(
                        next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    reward += args.gamma * policy.get_value(final_obs_t).item()

            buffer.store(obs, action_np, log_prob_np, reward, float(done), value_np)

            episode_reward += reward
            episode_length += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_successes.append(1.0 if step_info.get("is_success", False) else 0.0)
                episode_lengths.append(episode_length)

                # Log episode
                if len(episode_rewards) % 10 == 0:
                    recent_rewards = episode_rewards[-20:]
                    recent_successes = episode_successes[-20:]
                    avg_reward = np.mean(recent_rewards)
                    success_rate = np.mean(recent_successes)
                    print(
                        f"  Step {global_step:>8d} | "
                        f"Ep {len(episode_rewards):>5d} | "
                        f"Avg R: {avg_reward:>8.2f} | "
                        f"Succ: {success_rate*100:>5.1f}% | "
                        f"Phase: {step_info.get('phase', '?')} | "
                        f"Arm: {step_info.get('active_arm', '?')} | "
                        f"Collision W: {collision_weight:.2f}"
                    )

                    if writer:
                        writer.add_scalar("charts/episodic_return", avg_reward, global_step)
                        writer.add_scalar("charts/success_rate", success_rate, global_step)
                        writer.add_scalar("charts/episode_length", np.mean(episode_lengths[-20:]), global_step)
                        writer.add_scalar("charts/collision_curriculum", collision_weight, global_step)
                        writer.add_scalar("scenario/table_z", step_info["scenario"]["table_z"], global_step)

                episode_reward = 0.0
                episode_length = 0
                obs, info = env.reset()
            else:
                obs = next_obs

            if global_step >= args.total_timesteps:
                break

        # ---- Compute GAE ----
        with torch.inference_mode():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = policy.get_value(obs_t).item()

        buffer.compute_gae(last_value, args.gamma, args.gae_lambda)

        # ---- PPO update ----
        policy.train()
        update_stats = ppo_update(
            policy,
            optimizer,
            buffer,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            clip_ratio=args.clip_ratio,
            entropy_coef=args.entropy_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            device=device,
        )

        # Log update stats
        if writer:
            writer.add_scalar("losses/policy_loss", update_stats["policy_loss"], global_step)
            writer.add_scalar("losses/value_loss", update_stats["value_loss"], global_step)
            writer.add_scalar("losses/entropy", update_stats["entropy"], global_step)
            writer.add_scalar("losses/approx_kl", update_stats["approx_kl"], global_step)

        # ---- Checkpointing ----
        if len(episode_successes) >= 20:
            current_success = np.mean(episode_successes[-50:])
            if current_success > best_success_rate:
                best_success_rate = current_success
                ckpt_path = ckpt_dir / "best_model.pt"
                torch.save({
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "best_success_rate": best_success_rate,
                }, str(ckpt_path))
                print(f"  [CKPT] New best model saved (success rate: {best_success_rate*100:.1f}%)")

        # Periodic checkpoint
        if global_step % (args.n_steps * 20) < args.n_steps:
            ckpt_path = ckpt_dir / f"checkpoint_{global_step}.pt"
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            }, str(ckpt_path))

    # ---- Training complete ----
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Total steps:      {global_step}")
    print(f"  Total episodes:   {len(episode_rewards)}")
    print(f"  Total time:       {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Steps/sec:        {global_step / elapsed:.0f}")
    if episode_rewards:
        print(f"  Final avg reward: {np.mean(episode_rewards[-50:]):.2f}")
    if episode_successes:
        print(f"  Final success:    {np.mean(episode_successes[-50:])*100:.1f}%")
    print(f"  Best success:     {best_success_rate*100:.1f}%")
    print("=" * 65)

    # Save final model
    final_path = ckpt_dir / "final_model.pt"
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "args": vars(args),
    }, str(final_path))
    print(f"  Final model saved to: {final_path}")

    env.close()
    if writer:
        writer.close()


# =====================================================================
# CLI
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="PPO Training for Bimanual Pick-and-Place"
    )
    # Environment
    parser.add_argument(
        "--model", type=str,
        default="openarm_pick_and_place.xml",
        help="Path to MuJoCo XML model"
    )
    parser.add_argument("--headless", action="store_true", help="No rendering")
    parser.add_argument("--no-domain-rand", action="store_true",
                        help="Disable domain randomization (same as --dr-level off)")
    parser.add_argument("--dr-level", choices=["off", "mild", "full"], default="mild",
                        help="How strongly each episode's world varies: table height, object "
                             "sizes, friction, sensor noise, distractors, disturbances "
                             "(plus physics/latency/perception randomisation)")

    # PPO hyperparameters
    parser.add_argument("--total-timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Steps per rollout")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="SGD epochs per update")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip-ratio", type=float, default=0.2,
                        help="PPO clip ratio")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Max gradient norm")
    parser.add_argument("--target-kl", type=float, default=0.02,
                        help="KL divergence early stopping threshold")

    # Curriculum
    parser.add_argument("--collision-warmup-steps", type=int, default=100_000,
                        help="Steps to linearly ramp collision penalty to full")

    # Logging & checkpointing
    parser.add_argument("--log-dir", type=str, default="runs",
                        help="TensorBoard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Model checkpoint directory")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device (cpu, cuda, auto)")

    args = parser.parse_args()
    if args.no_domain_rand:
        args.dr_level = "off"

    # Auto-detect device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)

