"""The training loop runs end to end: rollout, PPO update, checkpoint, reload, evaluation."""
import numpy as np
import pytest
import torch

from dual_arm_rl.config import Config
from dual_arm_rl.rl.evaluate import evaluate_policy, summarise
from dual_arm_rl.rl.normalizer import RunningMeanStd
from dual_arm_rl.rl.ppo import PPOTrainer, load_policy


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, size=(1000, 4))
    rms = RunningMeanStd((4,))
    for chunk in np.array_split(x, 7):
        rms.update(chunk)
    np.testing.assert_allclose(rms.mean, x.mean(0), atol=1e-3)
    np.testing.assert_allclose(rms.var, x.var(0), rtol=1e-2)


def test_config_round_trips_and_accepts_overrides():
    cfg = Config().apply_overrides(["ppo.lr=1e-4", "ppo.n_envs=3", "curriculum.advance_success=0.5"])
    back = Config.from_dict(cfg.to_dict())
    assert back.ppo.lr == 1e-4 and back.ppo.n_envs == 3 and back.curriculum.advance_success == 0.5
    assert back.curriculum.stages[2].max_distance == float("inf")
    with pytest.raises(ValueError):
        Config().apply_overrides(["ppo.not_a_field=1"])


@pytest.mark.slow
def test_a_tiny_training_run_saves_a_usable_checkpoint(tmp_path):
    cfg = Config()
    cfg.ppo.n_envs, cfg.ppo.n_steps, cfg.ppo.n_epochs = 2, 32, 2
    cfg.ppo.total_timesteps = 64
    cfg.eval.train_episodes = 1
    cfg.eval.interval_steps = 10 ** 9
    trainer = PPOTrainer(cfg, tmp_path, use_async=False, log_fn=lambda *a: None)
    summary = trainer.train()
    assert summary["global_step"] == 64
    ckpt = tmp_path / "checkpoints" / "latest.pt"
    assert ckpt.exists() and (tmp_path / "checkpoints" / "best.pt").exists()
    policy, obs_norm, cfg2, raw = load_policy(ckpt)
    assert raw["global_step"] == 64 and obs_norm.rms.count > 1
    res = evaluate_policy(policy, obs_norm, cfg2, episodes=1, stage=0, seed=0)
    report = summarise(res["summaries"])
    assert report["episodes"] == 1 and 0.0 <= report["success_rate"] <= 1.0
    # resuming continues from the saved step
    cfg2.ppo.total_timesteps = 128
    t2 = PPOTrainer(cfg2, tmp_path, resume=ckpt, use_async=False, log_fn=lambda *a: None)
    assert t2.global_step == 64
    torch.testing.assert_close(t2.policy.log_std, policy.log_std)
