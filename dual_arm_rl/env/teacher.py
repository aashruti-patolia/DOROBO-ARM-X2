"""
The scripted teacher: the robot's base knowledge of how to grip and drop.

It plays the two learned skills, GRASP and PLACE, the way a classical
controller would, using **only what the robot can sense**: the camera's
estimates of the ball, the basket and the table, the pinch point from forward
kinematics, the finger angle and the gripper force. It never reads the
simulator's ground truth.

Its output is the policy's own 4-D action (dx, dy, dz, grip in [-1, 1], in the
canonical mirrored frame), so its actions can be imitated directly:

* demonstrations are recorded from it and the policy is first trained to
  copy it (behaviour cloning), and
* during PPO fine-tuning it labels the states the *policy* visits, with a
  weight that fades out, so the policy keeps the basics while reward teaches
  it to do better.

The teacher decides from the current state only (no memory of its own past
actions), so its label is meaningful in any state the policy reaches.

GRASP:  holding -> lift | fingers closed on nothing -> open, back off |
        off in XY -> move over the ball | above the ball -> descend | -> close
PLACE:  ball not over the basket -> move it over | not low enough -> lower |
        -> open
"""
from __future__ import annotations

import numpy as np

from .reward import GRASP


class Teacher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.p = cfg.prior
        self.c = cfg.control
        self.dt = 0.002 * cfg.control.substeps        # control tick [s]

    def _move(self, err: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """P-D step toward ``err``: the damping on the hand's own velocity keeps the
        lagging servo-driven arm from overshooting."""
        n = len(err)
        cmd = self.p.teacher_gain * err - self.p.teacher_damping * vel[:n] * self.dt
        return np.clip(cmd / self.c.policy_max_step, -1.0, 1.0)

    def act(self, env) -> np.ndarray:
        """Teacher action for the env's current state (canonical frame)."""
        s, side, b, c, p = env.scene, env._side, env.perception.state, self.c, self.p
        pinch, vel = s.pinch_pos(side), s.pinch_vel(side)
        a = np.zeros(4, dtype=np.float32)
        if env._skill == GRASP:
            a[:] = self._grasp(env, s, side, b, pinch, vel, c, p)
        else:
            a[:] = self._place(env, b, pinch, vel, c, p)
        if side == "right":
            a[1] = -a[1]                                     # world -> canonical (mirrored) frame
        return a

    # -----------------------------------------------------------------
    def _grasp(self, env, s, side, b, pinch, vel, c, p):
        a = np.zeros(4)
        if env._held_sensor():                                # ball in the fingers: lift it
            a[2], a[3] = 1.0, -1.0
            return a
        closing = env._grip_cmd < 0.5
        if closing and s.finger_angle(side) < c.held_min_angle and s.finger_speed(side) < 0.3:
            a[2], a[3] = 0.5, 1.0                             # closed on nothing: open, back off
            return a
        err = b.ball_pos - pinch
        xy = float(np.linalg.norm(err[:2]))
        if closing and xy < 2 * p.align_tol and abs(err[2]) < 3 * p.height_tol:
            a[3] = -1.0                                       # closing around the ball: keep closing
            return a
        a[3] = 1.0
        if xy > p.coarse_align:                               # get over the ball first (rise allowed)
            err = err.copy()
            err[2] = max(err[2], 0.0)
            a[:3] = self._move(err, vel)
        elif err[2] < -p.height_tol:                          # descend while centring
            a[:3] = self._move(err, vel)
        elif xy > p.align_tol:                                # at ball height: final centring
            err = err.copy()
            err[2] = 0.0
            a[:3] = self._move(err, vel)
        else:                                                 # fingers around the ball: close
            a[3] = -1.0
        return a

    def _place(self, env, b, pinch, vel, c, p):
        a = np.zeros(4)
        # where the ball is: its camera estimate if seen near the hand, else the pinch point
        near = b.ball_visible and np.linalg.norm(b.ball_pos - pinch) < c.ball_near_hand
        ball = b.ball_pos if near else pinch
        err_xy = b.box_pos[:2] - ball[:2]
        lower_to = b.box_pos[2] + p.place_lower
        a[:2] = self._move(err_xy, vel)
        if np.linalg.norm(err_xy) < 2 * p.align_tol:
            a[2] = float(self._move(np.array([lower_to - pinch[2]]), vel[2:])[0])
        steady = np.linalg.norm(vel) < p.steady_speed
        aligned = np.linalg.norm(err_xy) < p.align_tol and pinch[2] < lower_to + 0.01
        a[3] = 1.0 if (aligned and steady) else -1.0
        return a
