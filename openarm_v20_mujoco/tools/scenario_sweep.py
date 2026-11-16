#!/usr/bin/env python3
"""
Scenario sweep: vary one world parameter at a time and report how a policy copes.

For each parameter the tool evaluates a list of values (the nominal range plus
a couple of beyond-range extremes), several episodes each, with every other
parameter at its nominal value. The result shows *which parameter breaks the
policy and where* - something a single success-rate number cannot.

    python tools/scenario_sweep.py                       # scripted baseline, all parameters
    python tools/scenario_sweep.py --episodes 40 --params table_z ball_radius
    python tools/scenario_sweep.py --checkpoint checkpoints/best_model.pt
    python tools/scenario_sweep.py --level full --out sweep.csv   # also randomise physics

Outputs a markdown table to stdout and, with ``--out``, a CSV.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bimanual_env as be  # noqa: E402
from scripted_policy import run_episode, scripted_action  # noqa: E402

# parameter -> list of values. The first and last entries lie beyond the
# training range on purpose, to show how gracefully the policy degrades.
SWEEPS = {
    "table_z": [0.13, 0.17, 0.21, 0.25, 0.29, 0.33, 0.37],
    "table_dx": [-0.04, -0.02, 0.0, 0.02, 0.04],
    "table_dy": [-0.04, -0.02, 0.0, 0.02, 0.04],
    "ball_radius": [0.016, 0.019, 0.022, 0.025, 0.028, 0.031],
    "box_half": [0.034, 0.038, 0.045, 0.052, 0.058],
    "box_depth": [0.040, 0.050, 0.060, 0.075, 0.090],
    "table_friction_scale": [0.4, 0.7, 1.0, 1.3, 1.6],
    "box_friction_scale": [0.4, 0.7, 1.0, 1.3, 1.6],
    "ball_solref_scale": [0.4, 0.6, 1.0, 1.6, 2.4],
    "encoder_pos_noise": [0.0, 0.002, 0.005, 0.01, 0.02],
    "dropout_prob": [0.0, 0.02, 0.05, 0.10, 0.25],
    "push_prob": [0.0, 0.01, 0.03, 0.08],
    "n_distractors": [0, 1, 2, 3],
}


def load_policy(checkpoint: str | None, env):
    """Scripted baseline, or a trained PPO policy acting deterministically."""
    if checkpoint is None:
        return scripted_action, "scripted baseline"
    import torch
    from ppo_network import SkillConditionedActorCritic
    ck = torch.load(checkpoint, map_location="cpu")
    net = SkillConditionedActorCritic(obs_dim=env.obs_dim, action_dim=env.action_dim,
                                      hidden_sizes=(256, 256), actor_hidden=128, critic_hidden=128)
    net.load_state_dict(ck["policy_state_dict"])
    net.eval()

    def act(_env, obs):
        with torch.inference_mode():
            return net.get_deterministic_action(torch.as_tensor(obs, dtype=torch.float32)).numpy()
    return act, f"PPO checkpoint {Path(checkpoint).name}"


def evaluate(env, policy, scenario, episodes, seed0):
    rows = [run_episode(env, policy, seed=seed0 + i, scenario=scenario) for i in range(episodes)]
    ok = np.array([r["success"] for r in rows])
    fails = Counter(f'{r["furthest_phase"]}' for r in rows if not r["success"])
    return {
        "success": ok.mean(),
        "steps": float(np.mean([r["steps"] for r in rows])),
        "failed_in": ", ".join(f"{k}:{v}" for k, v in sorted(fails.items())) or "-",
        "unreachable": sum(r["unreachable"] for r in rows),
        "obstacle_hits": sum(r["obstacle_hit"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=20, help="episodes per value")
    ap.add_argument("--params", nargs="*", default=list(SWEEPS), choices=list(SWEEPS))
    ap.add_argument("--level", default="off", choices=list(be.LEVEL_SCALE),
                    help="randomisation level for everything NOT being swept")
    ap.add_argument("--checkpoint", default=None, help="PPO checkpoint (default: scripted baseline)")
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--out", default=None, help="write a CSV here")
    ap.add_argument("--threshold", type=float, default=0.8, help="success rate counted as 'working'")
    args = ap.parse_args()

    env = be.BimanualPickPlaceEnv(randomization_level=args.level, obs_noise_scale=0.0)
    policy, name = load_policy(args.checkpoint, env)
    print(f"# Scenario sweep - {name}, {args.episodes} episodes/value, other parameters at level '{args.level}'\n")

    # nominal reference
    base = evaluate(env, policy, None, args.episodes, args.seed)
    print(f"nominal world: {base['success']:.0%} success, {base['steps']:.0f} steps\n")

    records = []
    for param in args.params:
        print(f"## {param}\n")
        print("| value | success | steps | failures end in | out of reach | obstacle hits |")
        print("|---|---|---|---|---|---|")
        working = []
        for value in SWEEPS[param]:
            r = evaluate(env, policy, {param: value}, args.episodes, args.seed)
            records.append({"param": param, "value": value, **r})
            mark = "" if r["success"] >= args.threshold else " **<**"
            print(f"| {value:g} | {r['success']:.0%}{mark} | {r['steps']:.0f} | {r['failed_in']} "
                  f"| {r['unreachable']} | {r['obstacle_hits']} |", flush=True)
            if r["success"] >= args.threshold:
                working.append(value)
        if len(working) == len(SWEEPS[param]):
            print(f"\nholds across the whole tested range ({min(working):g} ... {max(working):g})\n")
        elif working:
            print(f"\nworks (>= {args.threshold:.0%}) from {min(working):g} to {max(working):g}\n")
        else:
            print("\nnever reaches the threshold\n")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
