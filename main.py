#!/usr/bin/env python3
"""
Dual-arm pick-and-place: one command for the whole pipeline.

    python main.py                 # everything: check -> teacher -> pretrain -> train -> evaluate -> demo
    python main.py all --quick     # the same, shrunk to a few minutes (smoke test)

Individual stages:

    python main.py check           # assets, reachability maps, camera accuracy, speed, parallel envs
    python main.py teacher [--view]  # the scripted teacher (base knowledge): success per stage, or watch it
    python main.py pretrain        # teacher demonstrations + behaviour cloning -> checkpoints/pretrained.pt
    python main.py train [--timesteps N] [--envs K] [--resume PATH] [--no-prior]
    python main.py eval  [--checkpoint PATH] [--episodes 100]
    python main.py demo  [--checkpoint PATH] [--view]
    python main.py watch           # live MuJoCo viewer that follows a training run (2nd terminal)
    python main.py test            # unit / integration tests (pytest)
    python main.py maps            # rebuild the reachability maps

Common options: --run-name NAME (outputs go to outputs/NAME), --seed S,
--set section.field=value (any config value, e.g. --set ppo.lr=1e-4).

What is scripted and what is learned: see README.md and dual_arm_rl/config.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from dual_arm_rl.config import ASSETS_DIR, MODEL_XML, OUTPUTS_DIR, PROJECT_ROOT, WORKSPACE_MAPS, Config, quick_config


# ---------------------------------------------------------------------------
# Logging to console and outputs/<run>/log.txt
# ---------------------------------------------------------------------------
class Log:
    def __init__(self, path: Path | None):
        self.file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str = ""):
        print(msg, flush=True)
        if self.file:
            self.file.write(msg + "\n")
            self.file.flush()


def banner(log, title):
    log("")
    log("#" * 72)
    log(f"#  {title}")
    log("#" * 72)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_maps(cfg: Config, log, force: bool = False) -> bool:
    if WORKSPACE_MAPS.exists() and not force:
        log(f"[maps] reachability maps present: {WORKSPACE_MAPS}")
        return True
    log("[maps] building reachability maps (one-off, several minutes) ...")
    from dual_arm_rl.tools.compute_workspace import build_maps
    build_maps()
    return WORKSPACE_MAPS.exists()


def stage_check(cfg: Config, log, quick: bool = False) -> bool:
    """Everything that should work before spending hours on training."""
    banner(log, "CHECK  assets, maps, camera perception, speed, parallel environments")
    ok = True
    meshes = list((ASSETS_DIR / "mj_meshes").glob("*.stl"))
    log(f"[check] model {MODEL_XML.name}: {'ok' if MODEL_XML.exists() else 'MISSING'}, {len(meshes)} meshes")
    ok &= MODEL_XML.exists() and len(meshes) > 0
    ok &= stage_maps(cfg, log)

    from dual_arm_rl.env.pick_place_env import PickPlaceEnv
    from dual_arm_rl.env.scene import Scene
    from dual_arm_rl.perception.camera import HeadCamera

    # Camera accuracy at each curriculum stage
    scene = Scene(cfg)
    cam = HeadCamera(scene.model, cfg.camera)
    rng = np.random.default_rng(cfg.run.seed)
    n = 30 if quick else 80
    log(f"[check] camera perception over {n} random worlds per stage (detection, 3-D error):")
    for k, st in enumerate(cfg.curriculum.stages):
        seen = side_ok = 0
        eb, ex, et = [], [], []
        for _ in range(n):
            world = scene.reset(rng, st)
            det = cam.capture(scene.data, rng, st.visual_level)
            if det.ball_pos is not None and det.box_pos is not None:
                seen += 1
                eb.append(np.linalg.norm(det.ball_pos - scene.ball_pos()))
                ex.append(np.linalg.norm(det.box_pos[:2] - scene.box_pos()[:2]))
                side_ok += (det.ball_pos[1] > 0) == (world["side"] == "left")
            if det.table_z is not None:
                et.append(abs(det.table_z - scene.sc.table_z))
        rate = seen / n
        mb, mx = 1000 * np.mean(eb or [1]), 1000 * np.mean(ex or [1])
        good = rate >= 0.95 and mb < 12 and mx < 12 and side_ok == seen
        ok &= good
        log(f"   stage {k} {st.name:9s} detected {rate * 100:5.1f}% | ball {mb:4.1f} mm | basket {mx:4.1f} mm | "
            f"table {1000 * np.mean(et or [1]):3.1f} mm | arm choice {side_ok}/{seen} correct "
            f"{'OK' if good else 'FAIL'}")
    cam.close()

    # Environment speed (single process)
    env = PickPlaceEnv(cfg, stage=0, seed=cfg.run.seed)
    obs, _ = env.reset()
    t0, steps = time.time(), 0
    while steps < (60 if quick else 150):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        steps += 1
        if term or trunc:
            env.reset()
    dps = steps / (time.time() - t0)
    finite = bool(np.isfinite(obs).all())
    ok &= finite
    log(f"[check] environment: obs {obs.shape[0]} values (finite: {finite}), "
        f"{dps:.0f} decisions/s in one process (random actions, incl. resets)")
    env.close()

    # Parallel environments (spawned worker processes)
    from dual_arm_rl.rl.ppo import make_vector_env
    try:
        envs = make_vector_env(cfg, 0, 2, cfg.run.seed, use_async=True)
        envs.reset(seed=0)
        for _ in range(3):
            envs.step(np.zeros((2, 4), np.float32))
        kind = type(envs).__name__
        envs.close()
        log(f"[check] parallel environments: {kind} OK")
    except Exception as exc:
        ok = False
        log(f"[check] parallel environments FAILED: {exc}")

    log(f"[check] {'PASSED' if ok else 'FAILED'}")
    return ok


def stage_train(cfg: Config, run_dir: Path, log, resume=None, use_async=True) -> dict:
    banner(log, "TRAIN  PPO on the learned skills (GRASP, PLACE) with curriculum")
    from dual_arm_rl.rl.ppo import PPOTrainer
    if resume is None and (run_dir / "checkpoints" / "latest.pt").exists():
        log(f"[train] note: {run_dir} already has a checkpoint; starting a fresh run would overwrite it. "
            f"Use --resume {run_dir / 'checkpoints' / 'latest.pt'} to continue it.")
    init_from = None
    if cfg.prior.enabled and resume is None:
        init_from = run_dir / "checkpoints" / "pretrained.pt"
        if not init_from.exists():
            log("[train] no imitation policy yet - running the pretrain stage first")
            stage_pretrain(cfg, run_dir, log, use_async)
    elif not cfg.prior.enabled:
        log("[train] --no-prior: learning from scratch (no teacher, no imitation)")
    trainer = PPOTrainer(cfg, run_dir, resume=resume, use_async=use_async, log_fn=log, init_from=init_from)
    return trainer.train()


def stage_teacher(cfg: Config, run_dir: Path, log, episodes=None, view=False) -> dict:
    banner(log, "TEACHER  scripted base knowledge (camera + gripper sensors), full pipeline")
    from dual_arm_rl.rl.evaluate import run_teacher
    n = episodes or (0 if view else max(8, cfg.eval.train_episodes))
    return run_teacher(cfg, n, None if view else run_dir / "teacher.json", view=view, log=log)


def stage_pretrain(cfg: Config, run_dir: Path, log, use_async=True) -> Path:
    banner(log, "PRETRAIN  teacher demonstrations -> behaviour cloning (the policy copies the teacher)")
    from dual_arm_rl.rl.pretrain import run_pretrain
    return run_pretrain(cfg, run_dir, log=log, use_async=use_async)


def find_checkpoint(run_dir: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for name in ("best.pt", "latest.pt"):
        p = run_dir / "checkpoints" / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No checkpoint in {run_dir / 'checkpoints'}; train first or pass --checkpoint")


def stage_eval(cfg: Config, run_dir: Path, log, checkpoint=None, episodes=None) -> dict:
    banner(log, "EVALUATE  full pipeline at full difficulty (deterministic policy)")
    from dual_arm_rl.rl.evaluate import run_eval
    ckpt = find_checkpoint(run_dir, checkpoint)
    return run_eval(ckpt, episodes or cfg.eval.final_episodes, run_dir / "eval.json", log=log)


def stage_demo(cfg: Config, run_dir: Path, log, checkpoint=None, episodes=None, view=False):
    banner(log, "DEMO  " + ("live viewer" if view else "GIF (side view + head camera)"))
    from dual_arm_rl.rl.evaluate import run_demo
    ckpt = find_checkpoint(run_dir, checkpoint)
    return run_demo(ckpt, None if view else run_dir / "demo.gif", episodes or cfg.eval.demo_episodes,
                    view=view, fps=cfg.eval.demo_fps, log=log)


def stage_watch(run_dir: Path, log, checkpoint=None, episodes=None):
    """Live viewer that reloads the run's latest checkpoint whenever training saves one."""
    from dual_arm_rl.rl.evaluate import run_viewer
    ckpt = Path(checkpoint) if checkpoint else run_dir / "checkpoints" / "latest.pt"
    if not ckpt.exists():
        log(f"[watch] waiting for {ckpt} (training saves it every few updates) ...")
        while not ckpt.exists():
            time.sleep(5)
    run_viewer(ckpt, episodes=episodes or 0, reload=True, log=log)


def stage_test(log, quick: bool = False) -> bool:
    banner(log, "TEST  pytest")
    cmd = [sys.executable, "-m", "pytest", str(PROJECT_ROOT / "tests"), "-q"]
    if quick:
        cmd += ["-m", "not slow"]
    log("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=PROJECT_ROOT) == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="all",
                    choices=["all", "check", "teacher", "pretrain", "train", "eval", "demo", "watch",
                             "test", "maps"])
    ap.add_argument("--quick", action="store_true", help="shrink everything to a few-minute smoke run")
    ap.add_argument("--run-name", default=None, help="outputs/<run-name> (default: main, or quick)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--timesteps", type=int, default=None, help="policy decisions to train for")
    ap.add_argument("--envs", type=int, default=None, help="parallel environments")
    ap.add_argument("--resume", default=None, help="checkpoint to continue training from")
    ap.add_argument("--checkpoint", default=None, help="checkpoint for eval/demo (default: best.pt)")
    ap.add_argument("--episodes", type=int, default=None, help="episodes for eval/demo")
    ap.add_argument("--view", action="store_true", help="demo in the interactive MuJoCo viewer")
    ap.add_argument("--no-async", action="store_true", help="run environments in-process (debugging)")
    ap.add_argument("--no-prior", action="store_true",
                    help="train from scratch: no teacher demonstrations, no imitation")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE",
                    help="override any config value (repeatable)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config()
    if args.resume:                                   # continue with the run's own configuration
        import torch
        cfg = Config.from_dict(torch.load(args.resume, map_location="cpu", weights_only=False)["config"])
    if args.quick:
        cfg = quick_config(cfg)
    cfg.apply_overrides(args.set)
    if args.run_name:
        cfg.run.run_name = args.run_name
    if args.seed is not None:
        cfg.run.seed = args.seed
    if args.timesteps:
        cfg.ppo.total_timesteps = args.timesteps
    if args.envs:
        cfg.ppo.n_envs = args.envs
    if args.no_prior:
        cfg.prior.enabled = False
    run_dir = OUTPUTS_DIR / cfg.run.run_name
    log = Log(run_dir / "log.txt")
    log(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  python main.py {' '.join(argv or sys.argv[1:])}")

    cmd = args.command
    if cmd == "maps":
        return 0 if stage_maps(cfg, log, force=True) else 1
    if cmd == "test":
        return 0 if stage_test(log, args.quick) else 1
    if cmd == "check":
        return 0 if stage_check(cfg, log, args.quick) else 1
    if cmd == "teacher":
        stage_teacher(cfg, run_dir, log, args.episodes, args.view)
        return 0
    if cmd == "pretrain":
        stage_pretrain(cfg, run_dir, log, not args.no_async)
        return 0
    if cmd == "train":
        stage_train(cfg, run_dir, log, args.resume, not args.no_async)
        return 0
    if cmd == "eval":
        stage_eval(cfg, run_dir, log, args.checkpoint, args.episodes)
        return 0
    if cmd == "demo":
        stage_demo(cfg, run_dir, log, args.checkpoint, args.episodes, args.view)
        return 0
    if cmd == "watch":
        stage_watch(run_dir, log, args.checkpoint, args.episodes)
        return 0

    # all: the full pipeline
    t0 = time.time()
    if not stage_check(cfg, log, args.quick):
        log("[all] checks failed - fix the problems above before training (see README.md).")
        return 1
    if cfg.prior.enabled and args.resume is None:
        stage_teacher(cfg, run_dir, log, None if not args.quick else 3)
        stage_pretrain(cfg, run_dir, log, not args.no_async)
    stage_train(cfg, run_dir, log, args.resume, not args.no_async)
    report = stage_eval(cfg, run_dir, log, None, args.episodes)
    gif = stage_demo(cfg, run_dir, log, None, None, args.view)
    banner(log, "DONE")
    log(f"  run directory : {run_dir}")
    log(f"  best model    : {find_checkpoint(run_dir, None)}")
    log(f"  evaluation    : {run_dir / 'eval.json'}  (success {report['success_rate'] * 100:.1f}%)")
    if gif:
        log(f"  demo          : {gif}")
    log(f"  TensorBoard   : tensorboard --logdir {OUTPUTS_DIR}")
    log(f"  total time    : {(time.time() - t0) / 60:.1f} min")
    (run_dir / "pipeline_summary.json").write_text(json.dumps(
        {"success_rate": report["success_rate"], "minutes": (time.time() - t0) / 60}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
