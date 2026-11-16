"""
A hand-written policy for BimanualPickPlaceEnv, used as the reference
baseline by the tests and the scenario sweep.

It issues zero position residual (the environment's own IK base trajectory
does the reaching) and only decides the gripper: open while approaching,
closed from the moment the pinch point reaches the ball until the ball is
over the basket, then open. If this policy fails in a scenario, the scenario
itself (reachability, physics) is the problem, not learning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bimanual_env as be  # noqa: E402


def scripted_action(env: "be.BimanualPickPlaceEnv") -> np.ndarray:
    """Zero residual; close on the ball, carry it closed, open over the basket."""
    a = np.zeros(4, dtype=np.float32)
    if env._phase == be.PHASE_GRASP:
        a[3] = -1.0
    elif env._phase == be.PHASE_DROP and not env._ball_released:
        ee = env._get_ee_state()[0]
        box = env.data.xpos[env._box_body_id]
        over_box = (np.linalg.norm(ee[:2] - box[:2]) < 0.015
                    and ee[2] > env._sc.box_rim_z + 0.04)
        a[3] = 1.0 if over_box else -1.0
    else:
        a[3] = 1.0
    return a


def run_episode(env: "be.BimanualPickPlaceEnv", policy=scripted_action, seed=None, scenario=None):
    """Run one episode; returns a dict of outcome fields.

    ``scenario`` is a dict of Scenario fields to pin (see ``reset(options=...)``).
    """
    options = {"scenario": scenario} if scenario else None
    obs, info = env.reset(seed=seed, options=options)
    sc = info["scenario"]
    separation = float(np.linalg.norm(env._ball_pos[:2] - env._box_pos[:2]))   # at the start
    furthest = be.PHASE_APPROACH
    obstacle_hit = False
    steps = 0
    for steps in range(1, be.MAX_EPISODE_STEPS + 1):
        obs, _, terminated, truncated, info = env.step(policy(env) if policy is scripted_action else policy(env, obs))
        furthest = max(furthest, env._phase)
        obstacle_hit |= bool(info["arm_obstacle_contact"])
        if terminated or truncated:
            break
    return {
        "success": bool(info["is_success"]),
        "steps": steps,
        "furthest_phase": be.PHASE_NAMES[furthest],
        "end_phase": info["phase"],
        "arm": info["active_arm"],
        "unreachable": bool(info.get("ball_unreachable", False)),
        "obstacle_hit": obstacle_hit,
        "separation": separation,
        "scenario": sc,
    }
