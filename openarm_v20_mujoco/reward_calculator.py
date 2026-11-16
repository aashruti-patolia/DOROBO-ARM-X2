"""
Reward Calculator for Bimanual Pick-and-Place PPO Training
============================================================

Implements a dense, multi-stage reward function gated by the current
manipulation phase (skill_id):

    R = R_pick + R_grasp + R_drop + R_penalties

Each sub-reward is only active during its corresponding phase, enabling
smooth curriculum-driven training of the three sub-skills:
  0 = APPROACH & PICK
  1 = PRECISION GRASP & LIFT
  2 = COLLISION-FREE BASKET DROP

References:
  - Residual RL reward shaping (Johannink et al., 2019)
  - Dense reward engineering for dexterous manipulation (OpenAI, 2020)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    """Tunable coefficients for every reward/penalty term."""

    # -- Picking (Phase 0) --
    pick_distance_scale: float = 5.0       # tanh steepness for reach distance
    pick_distance_weight: float = 1.0      # weight of distance shaping
    pick_height_match_weight: float = 0.2  # bonus for matching ball Z

    # -- Grasping (Phase 1) --
    grasp_contact_bonus: float = 2.0       # per-finger contact reward
    grasp_both_fingers_bonus: float = 3.0  # bonus for both fingers touching
    grasp_slip_penalty: float = -1.5       # penalty for ball tangential velocity
    grasp_lift_weight: float = 2.0         # weight for lift height progress
    grasp_lift_target: float = 0.170       # target lift above table [m]

    # -- Dropping (Phase 2) --
    drop_xy_align_scale: float = 8.0       # tanh steepness for XY alignment
    drop_xy_align_weight: float = 1.5      # weight for XY alignment
    drop_height_bonus: float = 1.0         # bonus for being at safe release Z
    drop_safe_height_margin: float = 0.02  # epsilon above rim [m]
    drop_low_velocity_weight: float = 0.5  # bonus for near-zero XY velocity
    drop_release_bonus: float = 2.5        # paid ONCE per episode on a release over the basket
    drop_ball_in_box_bonus: float = 10.0   # paid ONCE per episode when the released ball lands inside

    # -- Basket collision penalty --
    collision_exp_lambda: float = 5.0      # exponential penalty magnitude at zero clearance
    collision_exp_k: float = 20.0          # exponential decay rate
    collision_margin: float = 0.03         # clearance [m] beyond which the soft penalty is zero
    collision_contact_penalty: float = -10.0  # hard penalty on actual contact (ALWAYS full strength)
    collision_curriculum_weight: float = 1.0  # curriculum weight of the soft proximity term only
    failure_penalty: float = -20.0            # once, when the episode fails: the gripper touched
                                              # the basket or the basket was tipped over

    # -- Action smoothness --
    smoothness_weight: float = 0.01        # ||da_t - da_{t-1}||^2 coefficient

    # -- General --
    time_penalty: float = -0.002           # small per-step penalty for speed
    success_bonus: float = 20.0            # paid ONCE per episode on a successful, released drop


class RewardCalculator:
    """Computes phase-gated dense rewards for the bimanual pick-and-place task.

    Usage:
        rc = RewardCalculator(config=RewardConfig())
        reward, info = rc.compute(
            phase=0,
            ee_pos=..., ball_pos=..., basket_pos=...,
            ball_vel=...,
            finger_contacts=(True, False),
            basket_clearance=0.05,
            arm_basket_contact=False,
            gripper_open_ratio=0.8,
            action=current_action,
            prev_action=previous_action,
            table_z=0.25,
            box_rim_z=0.31,
            ball_in_box=False,
            ball_released=False,
        )
    """

    def __init__(self, config: RewardConfig | None = None):
        self.cfg = config or RewardConfig()
        self.reset()

    def reset(self):
        """Clear per-episode state. Call at the start of every episode.

        The release, in-box and success bonuses are one-time payouts; this
        re-arms them. (Paying them every step would reward lingering in the
        rewarded state instead of reaching it.)
        """
        self._release_paid = False
        self._in_box_paid = False
        self._success_paid = False

    def compute(
        self,
        phase: int,
        ee_pos: np.ndarray,
        ball_pos: np.ndarray,
        basket_pos: np.ndarray,
        ball_vel: np.ndarray,
        finger_contacts: tuple[bool, bool],
        basket_clearance: float,
        arm_basket_contact: bool,
        gripper_open_ratio: float,
        action: np.ndarray,
        prev_action: np.ndarray,
        table_z: float,
        box_rim_z: float,
        ball_in_box: bool,
        ball_released: bool = True,
        task_failed: bool = False,
    ) -> tuple[float, dict]:
        """Compute the total reward for the current step.

        Parameters
        ----------
        phase : int
            Current manipulation phase (0=pick, 1=grasp, 2=drop).
        ee_pos : ndarray (3,)
            End-effector TCP position in world frame.
        ball_pos : ndarray (3,)
            Ball center position in world frame.
        basket_pos : ndarray (3,)
            Basket center (bottom) position in world frame.
        ball_vel : ndarray (6,) or (3,)
            Ball linear velocity (first 3 components used).
        finger_contacts : tuple(bool, bool)
            Whether (finger1, finger2) are in contact with the ball.
        basket_clearance : float
            Minimum distance from active arm geoms to basket geoms [m].
        arm_basket_contact : bool
            True if any arm/gripper geom is in contact with basket.
        gripper_open_ratio : float
            Current gripper opening (1=open, 0=closed).
        action : ndarray
            Current policy action output.
        prev_action : ndarray
            Previous step's policy action output.
        table_z : float
            Table surface height [m].
        box_rim_z : float
            Top of basket rim Z coordinate [m].
        ball_in_box : bool
            Whether the ball has landed inside the basket.
        ball_released : bool
            Whether the gripper has let go of the ball. The in-box and success
            bonuses require it, so holding the ball in the basket pays nothing.
        task_failed : bool
            The episode just failed (gripper touched the basket, or the basket
            was tipped). Pays ``failure_penalty`` and no success bonus.

        Returns
        -------
        total_reward : float
        info : dict
            Breakdown of individual reward components.
        """
        cfg = self.cfg
        info = {}

        # ---- Phase-gated sub-rewards ----
        r_pick = 0.0
        r_grasp = 0.0
        r_drop = 0.0

        if phase == 0:
            r_pick, pick_info = self._pick_reward(ee_pos, ball_pos)
            info.update(pick_info)
        elif phase == 1:
            r_grasp, grasp_info = self._grasp_reward(
                ee_pos, ball_pos, ball_vel, finger_contacts,
                gripper_open_ratio, table_z
            )
            info.update(grasp_info)
        elif phase == 2:
            r_drop, drop_info = self._drop_reward(
                ee_pos, ball_pos, basket_pos, ball_vel,
                gripper_open_ratio, box_rim_z, ball_in_box and ball_released
            )
            info.update(drop_info)

        # ---- Collision penalty (active in all phases) ----
        r_collision = self._collision_penalty(
            basket_clearance, arm_basket_contact
        )
        info["r_collision"] = r_collision

        # ---- Action smoothness penalty ----
        r_smooth = self._smoothness_penalty(action, prev_action)
        info["r_smooth"] = r_smooth

        # ---- Time penalty ----
        r_time = cfg.time_penalty
        info["r_time"] = r_time

        # ---- Success bonus ----
        r_success = 0.0
        if (ball_in_box and ball_released and phase == 2 and not task_failed
                and not self._success_paid):
            r_success = cfg.success_bonus
            self._success_paid = True
        info["r_success"] = r_success

        # ---- Failure penalty ----
        r_failure = cfg.failure_penalty if task_failed else 0.0
        info["r_failure"] = r_failure

        total = (r_pick + r_grasp + r_drop + r_collision + r_smooth + r_time
                 + r_success + r_failure)

        info["r_pick"] = r_pick
        info["r_grasp"] = r_grasp
        info["r_drop"] = r_drop
        info["r_total"] = total

        return total, info

    # -----------------------------------------------------------------
    # Phase 0: Approach & Pick
    # -----------------------------------------------------------------
    def _pick_reward(
        self,
        ee_pos: np.ndarray,
        ball_pos: np.ndarray,
    ) -> tuple[float, dict]:
        """Dense distance-shaped reward for reaching the ball."""
        cfg = self.cfg

        # Distance shaping: r = 1 - tanh(k * ||p_ee - p_ball||)
        dist = np.linalg.norm(ee_pos - ball_pos)
        r_dist = cfg.pick_distance_weight * (
            1.0 - np.tanh(cfg.pick_distance_scale * dist)
        )

        # Height matching: bonus for being at ball height
        height_err = abs(ee_pos[2] - ball_pos[2])
        r_height = cfg.pick_height_match_weight * max(0.0, 1.0 - 10.0 * height_err)

        total = r_dist + r_height
        return total, {
            "pick_dist": dist,
            "pick_r_dist": r_dist,
            "pick_r_height": r_height,
        }

    # -----------------------------------------------------------------
    # Phase 1: Precision Grasp & Lift
    # -----------------------------------------------------------------
    def _grasp_reward(
        self,
        ee_pos: np.ndarray,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        finger_contacts: tuple[bool, bool],
        gripper_open_ratio: float,
        table_z: float,
    ) -> tuple[float, dict]:
        """Reward for grasping and lifting the ball."""
        cfg = self.cfg

        # Contact reward
        n_contacts = int(finger_contacts[0]) + int(finger_contacts[1])
        r_contact = cfg.grasp_contact_bonus * (n_contacts / 2.0)
        if n_contacts == 2:
            r_contact += cfg.grasp_both_fingers_bonus

        # Slip penalty: penalize ball tangential velocity when gripper is closed
        ball_lin_vel = ball_vel[:3] if len(ball_vel) >= 3 else ball_vel
        ball_speed = np.linalg.norm(ball_lin_vel)
        r_slip = 0.0
        if gripper_open_ratio < 0.3 and n_contacts > 0:
            # Gripper is mostly closed and touching ball
            r_slip = cfg.grasp_slip_penalty * min(ball_speed, 1.0)

        # Lift height reward: proportional progress toward target lift
        lift_height = max(0.0, ball_pos[2] - table_z)
        lift_progress = min(1.0, lift_height / cfg.grasp_lift_target)
        r_lift = cfg.grasp_lift_weight * lift_progress

        total = r_contact + r_slip + r_lift
        return total, {
            "grasp_n_contacts": n_contacts,
            "grasp_r_contact": r_contact,
            "grasp_r_slip": r_slip,
            "grasp_ball_speed": ball_speed,
            "grasp_r_lift": r_lift,
            "grasp_lift_height": lift_height,
        }

    # -----------------------------------------------------------------
    # Phase 2: Collision-Free Basket Drop
    # -----------------------------------------------------------------
    def _drop_reward(
        self,
        ee_pos: np.ndarray,
        ball_pos: np.ndarray,
        basket_pos: np.ndarray,
        ball_vel: np.ndarray,
        gripper_open_ratio: float,
        box_rim_z: float,
        ball_in_box: bool,
    ) -> tuple[float, dict]:
        """Reward for aligning over basket and releasing cleanly."""
        cfg = self.cfg

        # XY alignment: ball (or EE if still held) over basket center
        # Use ball position when gripper is open, EE when closed
        ref_pos = ball_pos if gripper_open_ratio > 0.5 else ee_pos
        xy_dist = np.linalg.norm(ref_pos[:2] - basket_pos[:2])
        r_xy = cfg.drop_xy_align_weight * (
            1.0 - np.tanh(cfg.drop_xy_align_scale * xy_dist)
        )

        # Safe height bonus: EE should be above rim + margin
        above_rim = ee_pos[2] - (box_rim_z + cfg.drop_safe_height_margin)
        r_height = cfg.drop_height_bonus * (1.0 if above_rim > 0 else 0.0)

        # Low velocity bonus at release: reward near-zero XY velocity
        ball_lin_vel = ball_vel[:3] if len(ball_vel) >= 3 else ball_vel
        xy_speed = np.linalg.norm(ball_lin_vel[:2])
        r_low_vel = cfg.drop_low_velocity_weight * max(0.0, 1.0 - 5.0 * xy_speed)

        # Release bonus (once per episode): gripper open over the basket
        r_release = 0.0
        if (not self._release_paid
                and gripper_open_ratio > 0.7 and xy_dist < 0.03):
            r_release = cfg.drop_release_bonus
            self._release_paid = True

        # Ball landed in box bonus (once per episode; caller passes
        # ball_in_box only after the ball has been released)
        r_in_box = 0.0
        if ball_in_box and not self._in_box_paid:
            r_in_box = cfg.drop_ball_in_box_bonus
            self._in_box_paid = True

        total = r_xy + r_height + r_low_vel + r_release + r_in_box
        return total, {
            "drop_xy_dist": xy_dist,
            "drop_r_xy": r_xy,
            "drop_r_height": r_height,
            "drop_r_low_vel": r_low_vel,
            "drop_r_release": r_release,
            "drop_r_in_box": r_in_box,
        }

    # -----------------------------------------------------------------
    # Collision penalty (all phases)
    # -----------------------------------------------------------------
    def _collision_penalty(
        self, basket_clearance: float, arm_basket_contact: bool
    ) -> float:
        """Exponential penalty as clearance -> 0, hard penalty on contact.

        The soft term is shifted so it reaches exactly zero at
        ``collision_margin``: staying a safe distance from the basket (e.g.
        hovering above it before the drop) is free, and only intrusions
        inside the margin are penalised.
        """
        cfg = self.cfg

        # Soft exponential penalty, shifted/rescaled to vanish at the margin
        floor = np.exp(-cfg.collision_exp_k * cfg.collision_margin)
        raw = np.exp(-cfg.collision_exp_k * max(basket_clearance, 0.0))
        r_soft = -cfg.collision_exp_lambda * max(0.0, raw - floor) / (1.0 - floor)

        # Hard contact penalty
        r_hard = cfg.collision_contact_penalty if arm_basket_contact else 0.0

        # Only the soft "getting close" term is eased in by the curriculum.
        # Actually touching the basket is penalised in full from the first step.
        return cfg.collision_curriculum_weight * r_soft + r_hard

    # -----------------------------------------------------------------
    # Action smoothness penalty
    # -----------------------------------------------------------------
    def _smoothness_penalty(
        self, action: np.ndarray, prev_action: np.ndarray
    ) -> float:
        """Penalize jerky policy outputs."""
        delta = action - prev_action
        return -self.cfg.smoothness_weight * float(np.dot(delta, delta))

    # -----------------------------------------------------------------
    # Curriculum schedule
    # -----------------------------------------------------------------
    def set_curriculum(self, collision_weight: float):
        """Update the collision penalty curriculum weight.

        Parameters
        ----------
        collision_weight : float
            Ramp from 0.0 (early training) to 1.0 (full penalty).
        """
        self.cfg.collision_curriculum_weight = np.clip(collision_weight, 0.0, 2.0)
