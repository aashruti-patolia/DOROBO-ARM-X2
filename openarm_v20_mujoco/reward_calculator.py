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
    pick_orient_weight: float = 0.3        # bonus for grasp-approach alignment
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
    drop_release_bonus: float = 5.0        # one-time bonus for successful release
    drop_ball_in_box_bonus: float = 10.0   # big bonus when ball lands inside

    # -- Basket collision penalty --
    collision_exp_lambda: float = 5.0      # exponential penalty magnitude
    collision_exp_k: float = 20.0          # exponential decay rate
    collision_contact_penalty: float = -10.0  # hard penalty on actual contact
    collision_curriculum_weight: float = 1.0  # multiplied by curriculum schedule

    # -- Action smoothness --
    smoothness_weight: float = 0.01        # ||da_t - da_{t-1}||^2 coefficient

    # -- General --
    time_penalty: float = -0.002           # small per-step penalty for speed
    success_bonus: float = 20.0            # episode-level success bonus


class RewardCalculator:
    """Computes phase-gated dense rewards for the bimanual pick-and-place task.

    Usage:
        rc = RewardCalculator(config=RewardConfig())
        reward, info = rc.compute(
            phase=0,
            ee_pos=..., ball_pos=..., basket_pos=...,
            ee_quat=..., ball_vel=...,
            finger_contacts=(True, False),
            basket_clearance=0.05,
            arm_basket_contact=False,
            gripper_open_ratio=0.8,
            action=current_action,
            prev_action=previous_action,
            table_z=0.25,
            box_rim_z=0.31,
            ball_in_box=False,
        )
    """

    def __init__(self, config: RewardConfig | None = None):
        self.cfg = config or RewardConfig()

    def compute(
        self,
        phase: int,
        ee_pos: np.ndarray,
        ball_pos: np.ndarray,
        basket_pos: np.ndarray,
        ee_quat: np.ndarray,
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
        ee_quat : ndarray (4,)
            End-effector quaternion (w, x, y, z).
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
            r_pick, pick_info = self._pick_reward(ee_pos, ball_pos, ee_quat, table_z)
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
                gripper_open_ratio, box_rim_z, ball_in_box
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
        r_success = cfg.success_bonus if ball_in_box and phase == 2 else 0.0
        info["r_success"] = r_success

        total = r_pick + r_grasp + r_drop + r_collision + r_smooth + r_time + r_success

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
        ee_quat: np.ndarray,
        table_z: float,
    ) -> tuple[float, dict]:
        """Dense distance-shaped reward for reaching the ball."""
        cfg = self.cfg

        # Distance shaping: r = 1 - tanh(k * ||p_ee - p_ball||)
        dist = np.linalg.norm(ee_pos - ball_pos)
        r_dist = cfg.pick_distance_weight * (
            1.0 - np.tanh(cfg.pick_distance_scale * dist)
        )

        # Orientation alignment bonus:
        # Reward the EE Z-axis pointing generally downward (toward table)
        # ee_quat -> rotation matrix -> z-axis column
        ee_z_axis = _quat_to_z_axis(ee_quat)
        # Ideal: z-axis should have a negative z-component (pointing down)
        # Dot with [0, 0, -1]
        align = -ee_z_axis[2]  # Higher is better (max 1.0 when pointing down)
        r_orient = cfg.pick_orient_weight * max(0.0, align)

        # Height matching: bonus for being at ball height
        height_err = abs(ee_pos[2] - ball_pos[2])
        r_height = cfg.pick_height_match_weight * max(0.0, 1.0 - 10.0 * height_err)

        total = r_dist + r_orient + r_height
        return total, {
            "pick_dist": dist,
            "pick_r_dist": r_dist,
            "pick_r_orient": r_orient,
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

        # Release bonus: gripper open while positioned over basket
        r_release = 0.0
        if gripper_open_ratio > 0.7 and xy_dist < 0.03:
            r_release = cfg.drop_release_bonus * 0.5  # Partial

        # Ball landed in box bonus
        r_in_box = cfg.drop_ball_in_box_bonus if ball_in_box else 0.0

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
        """Exponential penalty as clearance -> 0, hard penalty on contact."""
        cfg = self.cfg

        # Soft exponential penalty
        r_soft = -cfg.collision_exp_lambda * np.exp(
            -cfg.collision_exp_k * max(basket_clearance, 0.0)
        )

        # Hard contact penalty
        r_hard = cfg.collision_contact_penalty if arm_basket_contact else 0.0

        return cfg.collision_curriculum_weight * (r_soft + r_hard)

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


# =====================================================================
# Helper
# =====================================================================
def _quat_to_z_axis(quat: np.ndarray) -> np.ndarray:
    """Extract the Z-axis (third column) of the rotation matrix from quaternion (w,x,y,z)."""
    w, x, y, z = quat
    return np.array([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ])

