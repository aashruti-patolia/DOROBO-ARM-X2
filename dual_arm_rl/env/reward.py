"""
Reward for the two learned skills: staged, potential-based shaping.

    r = event + gamma * Phi(s') - Phi(s) - penalties

* ``Phi`` measures task progress on one global scale. Every stage starts
  above the maximum of the stage before it:

  ===================  ==========================================  ==========
  stage                condition                                   Phi
  ===================  ==========================================  ==========
  GRASP, reaching      pinch point approaching the ball            0 ... 1
  GRASP, holding       both finger pads on the ball, lifting       2 ... 3
  PLACE                ball over the basket, aligning               3 ... 4
  terminal             episode over                                0
  ===================  ==========================================  ==========

* ``event`` is the outcome: +success when the released ball rests inside the
  upright basket, -miss when it lands outside, -timeout when a skill runs out
  of its step budget, -failure when the basket is tipped over or the ball
  leaves the table.

Shaping by a potential difference (Ng, Harada & Russell, 1999) rewards
*progress* and never rewards staying put: hovering in a well-paid state earns
(gamma - 1) * Phi < 0 per step, so the policy cannot farm shaping. The
shaped return of an episode adds up to the outcome minus Phi(s_0), so the
optimal policy is the one that completes the task. (The previous reward paid
+5 per step for touching the ball, which is exactly what the last run learned
to do: return 935, success 0 %.)

The reward is computed from the simulator's true state: it is the training
signal, never an input to the policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ControlConfig, RewardConfig

GRASP, PLACE = 0, 1
SKILL_NAMES = {GRASP: "grasp", PLACE: "place"}


@dataclass
class TaskSnapshot:
    """True state the reward needs (privileged: the policy never sees it)."""
    skill: int
    pinch: np.ndarray
    ball: np.ndarray
    box: np.ndarray            # basket base centre
    table_z: float
    ball_radius: float
    held: bool                 # both finger pads touching the ball


class RewardFunction:
    def __init__(self, cfg: RewardConfig, control: ControlConfig, gamma: float):
        self.cfg = cfg
        self.control = control
        self.gamma = gamma

    def potential(self, s: TaskSnapshot) -> float:
        c = self.cfg
        if s.skill == GRASP:
            if s.held:
                rest = s.table_z + s.ball_radius
                lift = np.clip((s.ball[2] - rest) / self.control.lift_success, 0.0, 1.0)
                return 2.0 + float(lift)
            return float(1.0 - np.tanh(c.reach_scale * np.linalg.norm(s.pinch - s.ball)))
        dxy = np.linalg.norm(s.ball[:2] - s.box[:2])
        return float(3.0 + (1.0 - np.tanh(c.align_scale * dxy)))

    @staticmethod
    def stage_of(s: TaskSnapshot) -> int:
        """0 reaching, 1 holding the ball, 2 placing (for progress statistics)."""
        if s.skill == PLACE:
            return 2
        return 1 if s.held else 0

    def __call__(self, phi: float, phi_next: float, terminal: bool, outcome: str | None,
                 penalties: dict) -> tuple[float, dict]:
        c = self.cfg
        event = {"success": c.success_bonus, "miss": -c.miss_penalty, "timeout": -c.timeout_penalty,
                 "failure": -c.failure_penalty}.get(outcome, 0.0)
        shaping = self.gamma * (0.0 if terminal else phi_next) - phi
        pen = float(sum(penalties.values()))
        total = event + shaping - pen
        info = {"r_event": event, "r_shaping": shaping, "r_penalty": -pen, "r_total": total}
        info.update({f"p_{k}": -v for k, v in penalties.items()})
        return total, info

    def penalties(self, *, box_contact: bool, distractor_contact: bool, safety_clip: float,
                  action: np.ndarray, prev_action: np.ndarray) -> dict:
        c = self.cfg
        da = np.asarray(action) - np.asarray(prev_action)
        return {
            "time": c.time_penalty,
            "box_contact": c.box_contact_penalty if box_contact else 0.0,
            "distractor": c.distractor_contact_penalty if distractor_contact else 0.0,
            "safety": c.safety_push_penalty if safety_clip > 1e-3 else 0.0,
            "action_rate": c.action_rate_penalty * float(da @ da),
        }
