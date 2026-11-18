"""A privileged, hand-written controller for the two learned skills - TEST ONLY.

It reads the simulator's true ball/basket positions, which the policy never
gets, and is used only to prove the environment is solvable (the scripted
moves, the safety limits and the friction grasp all work together). It is
never used in training: the PPO policy learns GRASP and PLACE from reward.
"""
import numpy as np

from dual_arm_rl.env.reward import GRASP

GAIN = 0.4


def oracle_action(env) -> np.ndarray:
    s, side = env.scene, env._side
    pinch, ball, box = s.pinch_pos(side), s.ball_pos(), s.box_pos()
    step = env.cfg.control.policy_max_step
    a = np.zeros(4, dtype=np.float32)
    if env._skill == GRASP:
        if env._held_sensor():
            a[2], a[3] = 1.0, -1.0                       # lift
        elif env._grip_cmd < 0.5:
            a[3] = -1.0                                  # keep closing
        else:
            err = ball - pinch
            if np.linalg.norm(err[:2]) > 0.006:
                err[2] = max(err[2], 0.0)                # align, then descend
            a[:3] = np.clip(GAIN * err / step, -1, 1)
            a[3] = 1.0 if np.linalg.norm(ball - pinch) > 0.006 else -1.0
    else:
        err = box - ball
        err[2] = 0.0
        a[:3] = np.clip(GAIN * err / step, -1, 1)
        a[3] = 1.0 if np.linalg.norm(err[:2]) < 0.008 else -1.0
    if side == "right":
        a[1] = -a[1]                                     # world -> canonical (mirrored) frame
    return a


def run_oracle(env, episodes: int, seed: int = 0, hover_start=None) -> list[dict]:
    out = []
    for k in range(episodes):
        opts = {"start_skill": "grasp"}
        if hover_start is not None:
            opts["hover_start"] = hover_start
        env.reset(seed=seed + k, options=opts)
        while True:
            _, _, term, trunc, info = env.step(oracle_action(env))
            if term or trunc:
                out.append(info["episode"])
                break
    return out
