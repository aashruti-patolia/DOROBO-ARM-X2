"""
Vectorised PPO trainer (CleanRL-style, single file) for the hybrid pick-and-place.

* ``n_envs`` environments run in parallel worker processes (AsyncVectorEnv,
  falls back to in-process SyncVectorEnv). Autoreset is "same step", so a
  truncated episode's final observation is available for bootstrapping.
* Generalised Advantage Estimation; clipped surrogate objective; value loss;
  entropy bonus; per-epoch KL early stopping; linear learning-rate decay.
* Running observation normalisation and return-based reward scaling (their
  statistics are saved in every checkpoint, so evaluation uses the same).
* Curriculum over *environment difficulty*: when the success rate of full
  episodes reaches ``advance_success`` the next stage is switched on in all
  workers.
* Base knowledge (``cfg.prior``): training can start from the behaviour-cloned
  policy (``init_from=pretrained.pt``). The critic is warmed up first, then an
  imitation term pulls the policy toward the scripted teacher's action on the
  states the policy itself visits (DAgger-style labels from
  ``info["teacher_action"]``), with a weight that decays to zero - after that,
  reward alone drives learning.
* TensorBoard logging, periodic deterministic evaluation at full difficulty,
  ``latest.pt`` / ``best.pt`` checkpoints, and ``--resume``.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter, deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..config import Config
from ..env.pick_place_env import PickPlaceEnv
from .network import ActorCritic
from .normalizer import ObsNormalizer, RewardScaler

SAME_STEP = gym.vector.AutoresetMode.SAME_STEP


# =====================================================================
# Environments
# =====================================================================
def _make_env_fn(cfg_dict: dict, stage: int, seed: int):
    def _thunk():
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        return PickPlaceEnv(Config.from_dict(cfg_dict), stage=stage, seed=seed)
    return _thunk


def make_vector_env(cfg: Config, stage, n_envs: int, seed: int, use_async: bool = True):
    """``stage`` is one curriculum stage for all workers, or a list with one per worker."""
    stages = list(stage) if isinstance(stage, (list, tuple)) else [stage] * n_envs
    fns = [_make_env_fn(cfg.to_dict(), stages[i], seed + 1000 * i) for i in range(n_envs)]
    if use_async and n_envs > 1:
        try:
            return gym.vector.AsyncVectorEnv(fns, context="spawn", autoreset_mode=SAME_STEP)
        except Exception as exc:                       # pragma: no cover - platform specific
            print(f"[WARN] parallel environments unavailable ({exc}); running them in-process.")
    return gym.vector.SyncVectorEnv(fns, autoreset_mode=SAME_STEP)


def finished_episodes(infos: dict) -> list[dict]:
    """Per-environment episode summaries from a vector step's ``final_info``."""
    fin = infos.get("final_info")
    if not fin or "episode" not in fin:
        return []
    ep, mask = fin["episode"], fin["_episode"]
    out = []
    for i in np.flatnonzero(mask):
        out.append({k: (v[i].item() if hasattr(v[i], "item") else v[i])
                    for k, v in ep.items() if not k.startswith("_")})
    return out


# =====================================================================
# Checkpoints
# =====================================================================
def save_checkpoint(path: Path, policy, optimizer, obs_norm, reward_scaler, cfg: Config,
                    global_step: int, stage: int, best_success: float, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "obs_norm": obs_norm.state_dict(),
        "reward_scaler": reward_scaler.state_dict() if reward_scaler is not None else None,
        "config": cfg.to_dict(),
        "global_step": global_step,
        "stage": stage,
        "best_success": best_success,
        "obs_dim": policy.obs_dim,
        "action_dim": policy.action_dim,
        **(extra or {}),
    }, tmp)
    os.replace(tmp, path)


def load_policy(path, device="cpu"):
    """(policy, obs_normalizer, config, raw checkpoint dict) from a checkpoint file."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    policy = ActorCritic(ckpt["obs_dim"], ckpt["action_dim"], cfg.ppo.hidden_sizes, cfg.ppo.init_log_std)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    obs_norm = ObsNormalizer(ckpt["obs_dim"], cfg.ppo.obs_clip)
    obs_norm.load_state_dict(ckpt["obs_norm"])
    return policy, obs_norm, cfg, ckpt


# =====================================================================
# Trainer
# =====================================================================
class PPOTrainer:
    def __init__(self, cfg: Config, run_dir: Path, resume: str | Path | None = None,
                 use_async: bool = True, log_fn=print, init_from: str | Path | None = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log = log_fn
        self.use_async = use_async
        p = cfg.ppo
        torch.manual_seed(cfg.run.seed)
        np.random.seed(cfg.run.seed)
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) - p.n_envs)))
        self.device = torch.device(p.device)

        self.global_step = 0
        self.stage = cfg.curriculum.start_stage
        self.best_success = -1.0
        probe = PickPlaceEnv(cfg, stage=self.stage, seed=cfg.run.seed)
        self.obs_dim = probe.observation_space.shape[0]
        self.action_dim = probe.action_space.shape[0]
        probe.close()

        self.policy = ActorCritic(self.obs_dim, self.action_dim, p.hidden_sizes, p.init_log_std).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=p.lr, eps=1e-5)
        self.obs_norm = ObsNormalizer(self.obs_dim, p.obs_clip)
        self.reward_scaler = RewardScaler(p.n_envs, p.gamma) if p.normalize_reward else None

        if resume:
            ckpt = torch.load(resume, map_location=self.device, weights_only=False)
            self.policy.load_state_dict(ckpt["policy"])
            if ckpt.get("optimizer"):
                self.optimizer.load_state_dict(ckpt["optimizer"])
            self.obs_norm.load_state_dict(ckpt["obs_norm"])
            if self.reward_scaler is not None and ckpt.get("reward_scaler"):
                self.reward_scaler.load_state_dict(ckpt["reward_scaler"])
            self.global_step = int(ckpt["global_step"])
            self.stage = int(ckpt["stage"])
            self.best_success = float(ckpt.get("best_success", -1.0))
            self.log(f"[resume] {resume}: step {self.global_step}, curriculum stage {self.stage}")
        elif init_from:
            ckpt = torch.load(init_from, map_location=self.device, weights_only=False)
            self.policy.load_state_dict(ckpt["policy"])
            self.obs_norm.load_state_dict(ckpt["obs_norm"])
            if self.reward_scaler is not None and ckpt.get("reward_scaler"):
                self.reward_scaler.load_state_dict(ckpt["reward_scaler"])
            self.log(f"[prior] starting from the imitation policy {init_from}")
        self.resumed = bool(resume)
        self.from_prior = bool(init_from) or bool(resume and cfg.prior.enabled)
        if self.from_prior:                                    # gentler fine-tuning of a working policy
            for g in self.optimizer.param_groups:
                g["lr"] = cfg.prior.finetune_lr

        (self.run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.run_dir / "tb"))
        except Exception as exc:                                # pragma: no cover
            self.log(f"[WARN] TensorBoard unavailable: {exc}")

    # -----------------------------------------------------------------
    def _scalar(self, tag, value):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, self.global_step)

    def _checkpoint(self, name: str, **extra):
        save_checkpoint(self.ckpt_dir / name, self.policy, self.optimizer, self.obs_norm, self.reward_scaler,
                        self.cfg, self.global_step, self.stage, self.best_success, extra)

    def bc_coef(self) -> float:
        """Weight of the teacher-imitation term: decays linearly to 0."""
        pr = self.cfg.prior
        if not pr.enabled or pr.bc_coef <= 0:
            return 0.0
        return float(pr.bc_coef * max(0.0, 1.0 - self.global_step / max(pr.bc_decay_steps, 1)))

    def _evaluate(self):
        from .evaluate import evaluate_policy
        e = self.cfg.eval
        final_stage = len(self.cfg.curriculum.stages) - 1
        res = evaluate_policy(self.policy, self.obs_norm, self.cfg, episodes=e.train_episodes,
                              stage=final_stage, seed=10_000 + self.global_step)
        self._scalar("eval/success_rate", res["success_rate"])
        self._scalar("eval/mean_return", res["mean_return"])
        self.log(f"  [eval] step {self.global_step:>9,d}  full-difficulty success "
                 f"{res['success_rate'] * 100:5.1f}%  return {res['mean_return']:6.2f}")
        if res["success_rate"] > self.best_success:
            self.best_success = res["success_rate"]
            self._checkpoint("best.pt", eval=res)
            self.log(f"  [ckpt] new best ({self.best_success * 100:.1f}%) -> {self.ckpt_dir / 'best.pt'}")
        return res

    # -----------------------------------------------------------------
    def train(self) -> dict:
        cfg, p, cur = self.cfg, self.cfg.ppo, self.cfg.curriculum
        N, T = p.n_envs, p.n_steps
        batch = N * T
        n_updates_total = max(1, p.total_timesteps // batch)
        start_update = self.global_step // batch

        self.log("=" * 72)
        self.log("PPO training - hybrid pick-and-place (scripted moves + learned GRASP/PLACE)"
                 + (" - fine-tuning the imitation policy" if self.from_prior else " - from scratch"))
        self.log(f"  envs {N} x {T} steps = batch {batch} | updates {n_updates_total} | "
                 f"target {p.total_timesteps:,} decisions | stage {self.stage} "
                 f"({cur.stages[self.stage].name})")
        self.log(f"  run dir: {self.run_dir}")
        self.log("=" * 72)

        envs = make_vector_env(cfg, self.stage, N, cfg.run.seed + self.global_step, self.use_async)
        obs_raw, infos = envs.reset(seed=cfg.run.seed + self.global_step)
        teacher = np.asarray(infos.get("teacher_action", np.zeros((N, self.action_dim))), np.float32)
        base_lr = cfg.prior.finetune_lr if self.from_prior else p.lr
        start_update_run = start_update

        buf_obs = np.zeros((T, N, self.obs_dim), np.float32)
        buf_act = np.zeros((T, N, self.action_dim), np.float32)
        buf_logp = np.zeros((T, N), np.float32)
        buf_rew = np.zeros((T, N), np.float32)
        buf_done = np.zeros((T, N), np.float32)
        buf_val = np.zeros((T, N), np.float32)
        buf_teach = np.zeros((T, N, self.action_dim), np.float32)

        recent = deque(maxlen=200)                     # all finished episodes
        stage_window = deque(maxlen=cur.window)        # full (GRASP-start) episodes at this stage
        next_eval = (self.global_step // cfg.eval.interval_steps + 1) * cfg.eval.interval_steps
        t0, steps0 = time.time(), self.global_step
        last = {}

        try:
            for update in range(start_update, n_updates_total):
                if p.anneal_lr:
                    frac = 1.0 - update / n_updates_total
                    for g in self.optimizer.param_groups:
                        g["lr"] = frac * base_lr

                # ---------------- rollout ----------------
                self.policy.eval()
                for t in range(T):
                    obs_n = self.obs_norm(obs_raw, update=True)
                    with torch.no_grad():
                        a, logp, _, v = self.policy.get_action_and_value(
                            torch.as_tensor(obs_n, device=self.device))
                    a_np = a.cpu().numpy()
                    next_raw, rew, term, trunc, infos = envs.step(a_np)
                    done = term | trunc
                    r = self.reward_scaler(rew, done) if self.reward_scaler is not None else rew.copy()
                    # time-limit truncation is not a real terminal state: bootstrap
                    boot = trunc & ~term
                    if boot.any() and "final_obs" in infos:
                        fo = np.stack([infos["final_obs"][i] for i in np.flatnonzero(boot)])
                        with torch.no_grad():
                            fv = self.policy.get_value(torch.as_tensor(self.obs_norm(fo), device=self.device))
                        r[boot] += p.gamma * fv.cpu().numpy()
                    buf_obs[t], buf_act[t], buf_logp[t] = obs_n, a_np, logp.cpu().numpy()
                    buf_rew[t], buf_done[t], buf_val[t] = r, done, v.cpu().numpy()
                    buf_teach[t] = teacher
                    obs_raw = next_raw
                    if "teacher_action" in infos:                 # label for the next observation
                        teacher = np.asarray(infos["teacher_action"], np.float32)
                    self.global_step += N

                    for ep in finished_episodes(infos):
                        recent.append(ep)
                        if ep.get("start_skill") == "grasp" and ep.get("stage") == self.stage:
                            stage_window.append(float(ep["success"]))

                # ---------------- GAE ----------------
                with torch.no_grad():
                    next_v = self.policy.get_value(
                        torch.as_tensor(self.obs_norm(obs_raw), device=self.device)).cpu().numpy()
                adv = np.zeros((T, N), np.float32)
                last_gae = np.zeros(N, np.float32)
                for t in reversed(range(T)):
                    nv = next_v if t == T - 1 else buf_val[t + 1]
                    nonterm = 1.0 - buf_done[t]
                    delta = buf_rew[t] + p.gamma * nv * nonterm - buf_val[t]
                    last_gae = delta + p.gamma * p.gae_lambda * nonterm * last_gae
                    adv[t] = last_gae
                ret = adv + buf_val

                # ---------------- PPO update ----------------
                warmup = (self.from_prior and not self.resumed
                          and update - start_update_run < cfg.prior.value_warmup_updates)
                stats = self._update(buf_obs.reshape(batch, -1), buf_act.reshape(batch, -1),
                                     buf_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1),
                                     teacher=buf_teach.reshape(batch, -1), bc_coef=self.bc_coef(),
                                     critic_only=warmup)

                # ---------------- curriculum ----------------
                if (self.stage < len(cur.stages) - 1 and len(stage_window) >= cur.min_episodes
                        and np.mean(stage_window) >= cur.advance_success):
                    self.stage += 1
                    envs.call("set_curriculum_stage", self.stage)
                    stage_window.clear()
                    self.log(f"  [curriculum] -> stage {self.stage} ({cur.stages[self.stage].name})")

                # ---------------- logging ----------------
                last = self._log_update(update, n_updates_total, recent, stage_window, stats,
                                        (self.global_step - steps0) / max(time.time() - t0, 1e-6))
                if (update + 1) % cfg.run.checkpoint_every_updates == 0:
                    self._checkpoint("latest.pt")
                if self.global_step >= next_eval:
                    next_eval += cfg.eval.interval_steps
                    last["eval"] = self._evaluate()
        except KeyboardInterrupt:
            self.log("\n[train] interrupted - saving latest checkpoint")
        finally:
            self._checkpoint("latest.pt")
            envs.close()

        if self.best_success < 0:                       # never evaluated (very short run)
            last["eval"] = self._evaluate()
        if self.writer is not None:
            self.writer.close()
        elapsed = time.time() - t0
        summary = {"global_step": self.global_step, "stage": self.stage, "best_success": self.best_success,
                   "minutes": elapsed / 60, "decisions_per_s": (self.global_step - steps0) / max(elapsed, 1e-6)}
        (self.run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
        self.log(f"[train] done: {self.global_step:,} decisions in {elapsed / 60:.1f} min, "
                 f"best full-difficulty success {max(self.best_success, 0) * 100:.1f}%")
        return summary

    # -----------------------------------------------------------------
    def _update(self, obs, act, old_logp, adv, ret, teacher=None, bc_coef: float = 0.0,
                critic_only: bool = False) -> dict:
        p = self.cfg.ppo
        target_kl = self.cfg.prior.finetune_target_kl if self.from_prior else p.target_kl
        self.policy.train()
        dev = self.device
        obs_t, act_t = torch.as_tensor(obs, device=dev), torch.as_tensor(act, device=dev)
        old_logp_t, ret_t = torch.as_tensor(old_logp, device=dev), torch.as_tensor(ret, device=dev)
        adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), device=dev)
        teach_t = torch.as_tensor(teacher, device=dev) if teacher is not None else None
        n = len(obs)
        sums = Counter()
        n_mb, epochs_run, early = 0, 0, False
        for epoch in range(p.n_epochs):
            perm = torch.randperm(n, device=dev)
            kls = []
            for start in range(0, n, p.minibatch_size):
                idx = perm[start:start + p.minibatch_size]
                _, logp, ent, v = self.policy.get_action_and_value(obs_t[idx], act_t[idx])
                log_ratio = logp - old_logp_t[idx]
                ratio = log_ratio.exp()
                pg1 = -adv_t[idx] * ratio
                pg2 = -adv_t[idx] * torch.clamp(ratio, 1 - p.clip_ratio, 1 + p.clip_ratio)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * ((v - ret_t[idx]) ** 2).mean()
                if teach_t is not None:
                    bc_loss = ((self.policy.actor(obs_t[idx]) - teach_t[idx]) ** 2).sum(-1).mean()
                else:
                    bc_loss = torch.zeros((), device=dev)
                if critic_only:                    # warm the critic up before moving the policy
                    loss = p.vf_coef * v_loss
                else:
                    loss = pg_loss + p.vf_coef * v_loss - p.entropy_coef * ent.mean() + bc_coef * bc_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), p.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean().item()
                    clipfrac = ((ratio - 1).abs() > p.clip_ratio).float().mean().item()
                kls.append(kl)
                sums.update({"policy_loss": pg_loss.item(), "value_loss": v_loss.item(),
                             "entropy": ent.mean().item(), "approx_kl": kl, "clipfrac": clipfrac,
                             "teacher_mse": bc_loss.item()})
                n_mb += 1
            epochs_run = epoch + 1
            if target_kl and not critic_only and np.mean(kls) > target_kl:
                early = True
                break
        out = {k: v / max(n_mb, 1) for k, v in sums.items()}
        out.update(epochs=epochs_run, kl_stop=early, lr=self.optimizer.param_groups[0]["lr"],
                   std=float(torch.exp(self.policy.log_std.detach()).mean()), bc_coef=bc_coef,
                   critic_only=critic_only)
        return out

    def _log_update(self, update, n_updates, recent, stage_window, stats, sps) -> dict:
        info = {"stage": self.stage}
        for k, v in stats.items():
            if k in ("bc_coef", "teacher_mse"):
                self._scalar(f"prior/{k}", float(v))
            else:
                self._scalar(f"losses/{k}", float(v))
        self._scalar("charts/decisions_per_s", sps)
        self._scalar("curriculum/stage", self.stage)
        if recent:
            eps = list(recent)
            succ = np.mean([e["success"] for e in eps])
            full = [e for e in eps if e.get("start_skill") == "grasp"]
            info.update(success=succ, ret=np.mean([e["return"] for e in eps]))
            self._scalar("charts/success_rate", succ)
            self._scalar("charts/episode_return", info["ret"])
            self._scalar("charts/episode_decisions", np.mean([e["length"] for e in eps]))
            self._scalar("perception/ball_error_mm", 1000 * np.mean([e["ball_err"] for e in eps]))
            if full:
                ms = np.array([e["max_stage"] for e in full])
                info["full_success"] = float(np.mean([e["success"] for e in full]))
                self._scalar("charts/full_episode_success", info["full_success"])
                for k, name in enumerate(["grasped", "lifted", "released", "success"], start=1):
                    self._scalar(f"progress/reached_{name}", float(np.mean(ms >= k)))
            if stage_window:
                self._scalar("curriculum/stage_success", float(np.mean(stage_window)))
            ends = Counter(e["end"] for e in eps)
            for k, v in ends.items():
                self._scalar(f"ends/{k}", v / len(eps))
        if update % 5 == 0 or update == n_updates - 1:
            fs = info.get("full_success")
            self.log(f"  upd {update + 1:>5d}/{n_updates} | step {self.global_step:>9,d} | "
                     f"stage {self.stage} | success {info.get('success', 0) * 100:5.1f}% "
                     f"(full {'-' if fs is None else f'{fs * 100:5.1f}%'}) | "
                     f"return {info.get('ret', 0):6.2f} | kl {stats['approx_kl']:.4f} | "
                     f"std {stats['std']:.3f} | teacher w {stats['bc_coef']:.2f}"
                     f"{' (critic warm-up)' if stats['critic_only'] else ''} | {sps:5.0f} dec/s")
        return info
