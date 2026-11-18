"""
Evaluation and demos of a trained policy.

* ``evaluate_policy`` - deterministic episodes in one environment; returns the
  success rate, mean return and the per-episode summaries.
* ``run_eval`` - evaluation of a checkpoint at full difficulty, with a
  breakdown of the success rate by what changed in the world (table height,
  ball size, basket size, distance, rolling ball, arm) and the failure causes;
  written to ``eval.json``.
* ``run_demo`` - a GIF (side view + head-camera view) of a few episodes, or a
  live MuJoCo viewer with ``view=True``.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..env.pick_place_env import PickPlaceEnv


def _act(policy, obs_norm, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
    with torch.no_grad():
        a = policy.act(torch.as_tensor(obs_norm(obs[None])), deterministic=deterministic)
    return a[0].cpu().numpy()


def evaluate_policy(policy, obs_norm, cfg: Config, episodes: int, stage: int, seed: int = 0,
                    start_skill: str = "grasp", env: PickPlaceEnv | None = None) -> dict:
    """Run deterministic episodes; returns aggregate numbers and episode summaries."""
    own = env is None
    env = env or PickPlaceEnv(cfg, stage=stage, seed=seed)
    env.set_curriculum_stage(stage)
    summaries = []
    for k in range(episodes):
        obs, _ = env.reset(seed=seed + k, options={"start_skill": start_skill, "hover_start": False})
        while True:
            obs, _, term, trunc, info = env.step(_act(policy, obs_norm, obs))
            if term or trunc:
                summaries.append(info["episode"])
                break
    if own:
        env.close()
    succ = [e["success"] for e in summaries]
    return {"episodes": len(summaries), "success_rate": float(np.mean(succ)) if succ else 0.0,
            "mean_return": float(np.mean([e["return"] for e in summaries])) if summaries else 0.0,
            "summaries": summaries}


def _breakdown(summaries, key, edges, fmt="{:.2f}"):
    rows = {}
    vals = np.array([e[key] for e in summaries], dtype=float)
    succ = np.array([e["success"] for e in summaries], dtype=float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (vals >= lo) & (vals < hi)
        if m.any():
            rows[f"{fmt.format(lo)}-{fmt.format(hi)}"] = {"episodes": int(m.sum()),
                                                          "success_rate": float(succ[m].mean())}
    return rows


def summarise(summaries: list[dict]) -> dict:
    succ = np.array([e["success"] for e in summaries], dtype=float)
    out = {
        "episodes": len(summaries),
        "success_rate": float(succ.mean()) if len(succ) else 0.0,
        "mean_return": float(np.mean([e["return"] for e in summaries])) if summaries else 0.0,
        "mean_decisions": float(np.mean([e["length"] for e in summaries])) if summaries else 0.0,
        "outcomes": dict(Counter(e["end"] for e in summaries)),
        "reached": {name: float(np.mean([e["max_stage"] >= k for e in summaries]))
                    for k, name in enumerate(["grasped", "lifted", "released", "success"], start=1)},
        "ball_perception_error_mm": float(1000 * np.mean([e["ball_err"] for e in summaries])),
        "by_table_height_m": _breakdown(summaries, "table_z", [0.17, 0.21, 0.25, 0.29, 0.331]),
        "by_ball_radius_mm": _breakdown([dict(e, r=1000 * e["ball_radius"]) for e in summaries],
                                        "r", [18.0, 20.5, 23.0, 25.5], "{:.1f}"),
        "by_basket_half_width_mm": _breakdown([dict(e, h=1000 * e["box_half"]) for e in summaries],
                                              "h", [38.0, 43.0, 48.0, 52.0], "{:.0f}"),
        "by_distance_m": _breakdown(summaries, "distance", [0.10, 0.20, 0.30, 0.50]),
        "by_rolling": {k: {"episodes": int(sum(e["rolling"] == v for e in summaries)),
                           "success_rate": float(np.mean([e["success"] for e in summaries if e["rolling"] == v]))}
                       for k, v in (("rolling", True), ("still", False))
                       if any(e["rolling"] == v for e in summaries)},
        "by_arm": {s: {"episodes": int(sum(e["side"] == s for e in summaries)),
                       "success_rate": float(np.mean([e["success"] for e in summaries if e["side"] == s]))}
                   for s in ("left", "right") if any(e["side"] == s for e in summaries)},
    }
    return out


def run_eval(checkpoint: Path, episodes: int, out_json: Path | None = None, seed: int = 123,
             log=print) -> dict:
    from .ppo import load_policy
    policy, obs_norm, cfg, ckpt = load_policy(checkpoint)
    final_stage = len(cfg.curriculum.stages) - 1
    log(f"[eval] {checkpoint} (trained {ckpt['global_step']:,} decisions, stage {ckpt['stage']}) - "
        f"{episodes} episodes at full difficulty")
    t0 = time.time()
    res = evaluate_policy(policy, obs_norm, cfg, episodes, final_stage, seed)
    report = summarise(res["summaries"])
    report["checkpoint"] = str(checkpoint)
    report["trained_decisions"] = int(ckpt["global_step"])
    report["minutes"] = (time.time() - t0) / 60
    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=2))
    log(f"[eval] success {report['success_rate'] * 100:.1f}%  |  reached: "
        + ", ".join(f"{k} {v * 100:.0f}%" for k, v in report["reached"].items()))
    log(f"[eval] outcomes: {report['outcomes']}")
    if out_json:
        log(f"[eval] report -> {out_json}")
    return report


def run_demo(checkpoint: Path, out_gif: Path | None, episodes: int = 3, view: bool = False,
             fps: int = 25, seed: int = 7, log=print):
    from .ppo import load_policy
    if view:
        run_viewer(checkpoint, episodes=episodes, seed=seed, log=log)
        return None
    policy, obs_norm, cfg, _ = load_policy(checkpoint)
    final_stage = len(cfg.curriculum.stages) - 1
    env = PickPlaceEnv(cfg, stage=final_stage, seed=seed)

    import imageio.v2 as imageio
    frames, tick = [], [0]

    def hook(e):
        tick[0] += 1
        if tick[0] % 2 == 0:                      # 12.5 fps of 25 Hz control
            frames.append(e.render())
    env.frame_hook = hook
    outcomes = []
    for k in range(episodes):
        obs, _ = env.reset(seed=seed + k, options={"start_skill": "grasp", "hover_start": False})
        frames.append(env.render())
        while True:
            obs, _, term, trunc, info = env.step(_act(policy, obs_norm, obs))
            if term or trunc:
                outcomes.append(info["episode"]["end"])
                for _ in range(6):
                    frames.append(frames[-1])
                break
    env.close()
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_gif, frames, duration=1000.0 / (fps / 2), loop=0)
    log(f"[demo] {len(frames)} frames, outcomes {outcomes} -> {out_gif}")
    return out_gif


def run_viewer(checkpoint: Path, episodes: int = 0, reload: bool = False, stage: int | None = None,
               seed: int = 7, log=print):
    """Watch the policy in the interactive MuJoCo viewer, in real time.

    ``episodes=0`` runs until the window is closed. With ``reload=True`` the
    checkpoint is re-read whenever the trainer saves a newer one, so you can
    watch the policy improve while training runs in another terminal. The
    whole pipeline is shown: camera-guided transit, the learned GRASP,
    transport, the learned PLACE and the retreat.
    """
    import os

    import mujoco.viewer

    from .ppo import load_policy

    checkpoint = Path(checkpoint)
    policy, obs_norm, cfg, ckpt = load_policy(checkpoint)
    mtime = os.path.getmtime(checkpoint)
    final_stage = len(cfg.curriculum.stages) - 1
    show_stage = (int(ckpt["stage"]) if reload else final_stage) if stage is None else stage
    env = PickPlaceEnv(cfg, stage=show_stage, seed=seed)
    log(f"[viewer] {checkpoint} - trained {ckpt['global_step']:,} decisions, showing curriculum stage "
        f"{show_stage}. Close the window (or Ctrl+C here) to stop.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        last = [time.perf_counter()]

        def hook(e):                                   # real-time playback, one frame per control tick
            viewer.sync()
            wait = e.dt_tick - (time.perf_counter() - last[0])
            if wait > 0:
                time.sleep(wait)
            last[0] = time.perf_counter()
            if not viewer.is_running():
                raise KeyboardInterrupt
        env.frame_hook = hook
        k, wins = 0, 0
        try:
            while viewer.is_running() and (episodes <= 0 or k < episodes):
                if reload and os.path.getmtime(checkpoint) != mtime:
                    try:                                # the trainer writes atomically (tmp + rename)
                        policy, obs_norm, _, ckpt = load_policy(checkpoint)
                        mtime = os.path.getmtime(checkpoint)
                        if stage is None:
                            env.set_curriculum_stage(int(ckpt["stage"]))
                        log(f"[viewer] reloaded: {ckpt['global_step']:,} decisions, stage {ckpt['stage']}")
                    except Exception as exc:            # pragma: no cover - mid-write race
                        log(f"[viewer] could not reload yet ({exc})")
                obs, _ = env.reset(seed=seed + k, options={"start_skill": "grasp", "hover_start": False})
                while True:
                    obs, _, term, trunc, info = env.step(_act(policy, obs_norm, obs))
                    if term or trunc:
                        k += 1
                        wins += info["episode"]["success"]
                        log(f"[viewer] episode {k}: {info['episode']['end']}  (success so far {wins}/{k})")
                        break
        except KeyboardInterrupt:
            pass
    env.close()


# ---------------------------------------------------------------------------
# The scripted teacher on its own (the base knowledge)
# ---------------------------------------------------------------------------
def evaluate_teacher(cfg: Config, episodes: int, stage: int, seed: int = 0,
                     start_skill: str = "grasp", env: PickPlaceEnv | None = None) -> list[dict]:
    """Full-pipeline episodes driven by the scripted teacher (camera + gripper sensors only)."""
    own = env is None
    env = env or PickPlaceEnv(cfg, stage=stage, seed=seed)
    env.set_curriculum_stage(stage)
    summaries = []
    for k in range(episodes):
        _, info = env.reset(seed=seed + k, options={"start_skill": start_skill, "hover_start": False})
        while True:
            _, _, term, trunc, info = env.step(info["teacher_action"])
            if term or trunc:
                summaries.append(info["episode"])
                break
    if own:
        env.close()
    return summaries


def run_teacher(cfg: Config, episodes: int, out_json: Path | None = None, view: bool = False,
                seed: int = 31, log=print) -> dict:
    """Success of the teacher at every curriculum stage, or watch it in the viewer."""
    if view:
        import mujoco.viewer
        stage = len(cfg.curriculum.stages) - 1
        env = PickPlaceEnv(cfg, stage=0, seed=seed)
        log("[teacher] watching the scripted teacher (base knowledge) - close the window to stop")
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            last = [time.perf_counter()]

            def hook(e):
                viewer.sync()
                wait = e.dt_tick - (time.perf_counter() - last[0])
                if wait > 0:
                    time.sleep(wait)
                last[0] = time.perf_counter()
                if not viewer.is_running():
                    raise KeyboardInterrupt
            env.frame_hook = hook
            k = 0
            try:
                while viewer.is_running() and (episodes <= 0 or k < episodes):
                    env.set_curriculum_stage(k % (stage + 1))
                    s = evaluate_teacher(cfg, 1, env.get_curriculum_stage(), seed + k, env=env)[0]
                    k += 1
                    log(f"[teacher] episode {k} (stage {s['stage']}): {s['end']}")
            except KeyboardInterrupt:
                pass
        env.close()
        return {}
    report = {}
    for st in range(len(cfg.curriculum.stages)):
        t0 = time.time()
        summ = evaluate_teacher(cfg, episodes, st, seed + 1000 * st)
        rep = summarise(summ)
        report[f"stage_{st}_{cfg.curriculum.stages[st].name}"] = rep
        log(f"[teacher] stage {st} ({cfg.curriculum.stages[st].name:8s}): success "
            f"{rep['success_rate'] * 100:5.1f}% over {episodes} episodes | outcomes {rep['outcomes']} "
            f"({(time.time() - t0) / 60:.1f} min)")
    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=2))
        log(f"[teacher] report -> {out_json}")
    return report
