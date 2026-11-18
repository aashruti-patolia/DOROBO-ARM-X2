"""
Base knowledge for the policy: demonstrations from the scripted teacher, then
behaviour cloning (BC).

1. ``collect_demos``: the teacher plays GRASP and PLACE in parallel
   environments, across the curriculum stages. With DART-style noise
   (Laskey et al. 2017) the *executed* action is the teacher's plus Gaussian
   noise while the *recorded* label is the clean teacher action, so the data
   also covers how to recover from small mistakes.
2. ``behaviour_clone``: the actor learns to output the teacher's action
   (MSE). The critic learns the discounted return of the demos, in the same
   scaled units PPO uses, so fine-tuning starts with a sensible value function.
3. The result is saved as ``checkpoints/pretrained.pt`` in the normal
   checkpoint format; PPO fine-tuning (``rl/ppo.py``) starts from it.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import Config
from .network import ActorCritic
from .normalizer import ObsNormalizer, RewardScaler
from .ppo import finished_episodes, make_vector_env, save_checkpoint


def _worker_stages(mix, n_envs: int) -> list[int]:
    """Spread the stage mix over the workers (largest-remainder rounding)."""
    mix = np.asarray(mix, dtype=float)
    mix = mix / mix.sum()
    counts = np.floor(mix * n_envs).astype(int)
    for k in np.argsort(-(mix * n_envs - counts))[: n_envs - counts.sum()]:
        counts[k] += 1
    return [k for k, c in enumerate(counts) for _ in range(c)]


def collect_demos(cfg: Config, episodes: int, seed: int = 0, log=print, use_async: bool = True) -> dict:
    """Run the teacher; returns arrays obs, act (teacher labels), rew, done, env, plus stats."""
    p = cfg.prior
    n = max(1, min(p.demo_envs, episodes))
    stages = _worker_stages(p.demo_stage_mix, n)
    envs = make_vector_env(cfg, stages, n, seed, use_async)
    obs, infos = envs.reset(seed=seed)
    teacher = np.asarray(infos["teacher_action"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    buf = defaultdict(list)
    done_eps, per_stage = [], defaultdict(list)
    t0 = time.time()
    log(f"[demos] teacher plays {episodes} episodes in {n} environments "
        f"(stages per worker {stages}, DART noise {p.demo_noise})")
    while len(done_eps) < episodes:
        executed = np.clip(teacher + rng.normal(0, p.demo_noise, teacher.shape), -1, 1).astype(np.float32)
        nxt, rew, term, trunc, infos = envs.step(executed)
        buf["obs"].append(obs.astype(np.float32))
        buf["act"].append(teacher.copy())
        buf["rew"].append(rew.astype(np.float32))
        buf["done"].append((term | trunc).astype(np.float32))
        for ep in finished_episodes(infos):
            done_eps.append(ep)
            per_stage[int(ep["stage"])].append(ep)
        obs = nxt
        teacher = np.asarray(infos["teacher_action"], dtype=np.float32)
    envs.close()
    out = {k: np.stack(v) for k, v in buf.items()}            # (T, N, ...)
    stats = {"episodes": len(done_eps), "samples": int(out["obs"].shape[0] * out["obs"].shape[1]),
             "minutes": (time.time() - t0) / 60,
             "success_by_stage": {int(k): float(np.mean([e["success"] for e in v])) for k, v in per_stage.items()},
             "outcomes": dict(Counter(e["end"] for e in done_eps))}
    log(f"[demos] {stats['samples']:,} samples in {stats['minutes']:.1f} min | teacher success (with noise) "
        + ", ".join(f"stage {k}: {v * 100:.0f}%" for k, v in sorted(stats['success_by_stage'].items())))
    out["stats"] = stats
    return out


def _flatten_with_returns(demos: dict, gamma: float):
    """Flatten (T, N) arrays per env and compute discounted returns and reward scaling."""
    obs, act, rew, done = demos["obs"], demos["act"], demos["rew"], demos["done"]
    T, N = rew.shape
    scaler = RewardScaler(N, gamma)
    for t in range(T):                                        # same statistic PPO keeps
        scaler(rew[t].astype(np.float64), done[t].astype(bool))
    ret = np.zeros((T, N), np.float32)
    g = np.zeros(N, np.float32)
    for t in reversed(range(T)):
        g = rew[t] + gamma * g * (1.0 - done[t])
        ret[t] = g
    # drop the unfinished tail of each env (no return known after the last done)
    keep = np.zeros((T, N), bool)
    for i in range(N):
        ends = np.flatnonzero(done[:, i])
        if ends.size:
            keep[: ends[-1] + 1, i] = True
    scale = float(np.sqrt(scaler.rms.var + 1e-8))
    return (obs[keep], act[keep], ret[keep] / scale, scaler)


def behaviour_clone(cfg: Config, demos: dict, log=print, writer=None):
    """Train an ActorCritic to imitate the teacher. Returns (policy, obs_norm, reward_scaler, stats)."""
    p, pp = cfg.prior, cfg.ppo
    X_raw, Y, R, scaler = _flatten_with_returns(demos, pp.gamma)
    obs_norm = ObsNormalizer(X_raw.shape[1], pp.obs_clip)
    obs_norm.rms.update(X_raw)
    X = obs_norm(X_raw)
    policy = ActorCritic(X.shape[1], Y.shape[1], pp.hidden_sizes, p.init_log_std)
    opt = torch.optim.Adam(list(policy.actor.parameters()) + list(policy.critic.parameters()), lr=p.bc_lr)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_val = max(1, len(X) // 10)
    val, tr = idx[:n_val], idx[n_val:]
    Xt, Yt, Rt = (torch.as_tensor(a, dtype=torch.float32) for a in (X, Y, R))
    hist = []
    for epoch in range(p.bc_epochs):
        policy.train()
        perm = torch.as_tensor(rng.permutation(tr))
        a_loss = v_loss = 0.0
        nb = 0
        for s in range(0, len(perm), p.bc_batch):
            b = perm[s:s + p.bc_batch]
            la = ((policy.actor(Xt[b]) - Yt[b]) ** 2).sum(-1).mean()
            lv = ((policy.get_value(Xt[b]) - Rt[b]) ** 2).mean()
            opt.zero_grad()
            (la + 0.5 * lv).backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            a_loss += la.item()
            v_loss += lv.item()
            nb += 1
        policy.eval()
        with torch.no_grad():
            va = ((policy.actor(Xt[val]) - Yt[val]) ** 2).sum(-1).mean().item()
        hist.append((a_loss / nb, va, v_loss / nb))
        if writer is not None:
            writer.add_scalar("pretrain/bc_train_loss", a_loss / nb, epoch)
            writer.add_scalar("pretrain/bc_val_loss", va, epoch)
            writer.add_scalar("pretrain/value_loss", v_loss / nb, epoch)
        if epoch % 5 == 0 or epoch == p.bc_epochs - 1:
            log(f"[bc] epoch {epoch + 1:>3d}/{p.bc_epochs}  action loss train {a_loss / nb:.4f} "
                f"val {va:.4f} | value loss {v_loss / nb:.4f}")
    with torch.no_grad():
        policy.log_std.fill_(p.init_log_std)
    stats = {"samples": int(len(X)), "final_train_loss": hist[-1][0], "final_val_loss": hist[-1][1],
             "first_train_loss": hist[0][0]}
    return policy, obs_norm, scaler, stats


def run_pretrain(cfg: Config, run_dir: Path, log=print, use_async: bool = True) -> Path:
    """Teacher demos -> behaviour cloning -> checkpoints/pretrained.pt (+ evaluation)."""
    from .evaluate import evaluate_policy
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run_dir / "tb"))
    except Exception:                                          # pragma: no cover
        pass
    demos = collect_demos(cfg, cfg.prior.demo_episodes, seed=cfg.run.seed + 77, log=log, use_async=use_async)
    np.savez_compressed(run_dir / "demos.npz", **{k: v for k, v in demos.items() if k != "stats"})
    policy, obs_norm, scaler, bc_stats = behaviour_clone(cfg, demos, log=log, writer=writer)

    n_eval = max(4, cfg.eval.train_episodes)
    res = evaluate_policy(policy, obs_norm, cfg, episodes=n_eval, stage=0, seed=4242)
    log(f"[pretrain] imitation policy (deterministic, stage 0, full pipeline): "
        f"success {res['success_rate'] * 100:.0f}% over {n_eval} episodes")
    report = {"teacher": demos["stats"], "bc": bc_stats, "bc_policy_success_stage0": res["success_rate"]}
    (run_dir / "pretrain.json").write_text(json.dumps(report, indent=2))
    if writer is not None:
        writer.add_scalar("pretrain/bc_policy_success_stage0", res["success_rate"], 0)
        writer.close()
    path = run_dir / "checkpoints" / "pretrained.pt"
    save_checkpoint(path, policy, None, obs_norm, scaler, cfg, global_step=0, stage=0, best_success=-1.0,
                    extra={"pretrained": True, "pretrain_report": report})
    log(f"[pretrain] saved {path}")
    return path
