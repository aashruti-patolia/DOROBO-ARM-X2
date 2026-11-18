"""Base knowledge pipeline: teacher demos -> behaviour cloning -> PPO fine-tuning."""
import numpy as np
import pytest
import torch

from dual_arm_rl.config import Config
from dual_arm_rl.rl.ppo import PPOTrainer, load_policy, save_checkpoint
from dual_arm_rl.rl.pretrain import _worker_stages, behaviour_clone, collect_demos


def test_stage_mix_is_spread_over_the_workers():
    assert _worker_stages((0.6, 0.25, 0.15), 8) == [0, 0, 0, 0, 0, 1, 1, 2]
    assert len(_worker_stages((0.6, 0.25, 0.15), 3)) == 3


def _tiny_cfg():
    cfg = Config()
    cfg.prior.demo_envs = 2
    cfg.prior.bc_epochs = 4
    cfg.prior.bc_batch = 128
    cfg.prior.bc_decay_steps = 128
    cfg.prior.value_warmup_updates = 1
    cfg.ppo.n_envs, cfg.ppo.n_steps, cfg.ppo.n_epochs = 2, 32, 2
    cfg.ppo.total_timesteps = 128
    cfg.eval.train_episodes = 1
    cfg.eval.interval_steps = 10 ** 9
    return cfg


@pytest.mark.slow
def test_imitation_then_fine_tuning_runs_end_to_end(tmp_path):
    cfg = _tiny_cfg()
    demos = collect_demos(cfg, episodes=6, seed=0, log=lambda *a: None, use_async=False)
    assert demos["obs"].ndim == 3 and demos["act"].shape[-1] == 4
    policy, obs_norm, scaler, stats = behaviour_clone(cfg, demos, log=lambda *a: None)
    assert stats["final_train_loss"] < stats["first_train_loss"]          # it learns to copy
    assert float(torch.exp(policy.log_std.detach()).mean()) == pytest.approx(np.exp(cfg.prior.init_log_std), rel=1e-5)
    ckpt = tmp_path / "checkpoints" / "pretrained.pt"
    save_checkpoint(ckpt, policy, None, obs_norm, scaler, cfg, 0, 0, -1.0, {"pretrained": True})
    pol2, norm2, _, _ = load_policy(ckpt)
    trainer = PPOTrainer(cfg, tmp_path, use_async=False, log_fn=lambda *a: None, init_from=ckpt)
    assert trainer.from_prior and trainer.bc_coef() == pytest.approx(cfg.prior.bc_coef)
    trainer.train()
    assert trainer.global_step == 128 and trainer.bc_coef() == 0.0          # guidance faded out
