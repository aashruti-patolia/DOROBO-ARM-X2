"""
BimanualPickPlaceEnv -- Gymnasium Environment for PPO Training
================================================================

Residual RL environment for the OpenArm v2.0 bimanual robot.
The IK solver provides base trajectory targets; the PPO policy outputs
small residual corrections (delta_pos, delta_orient, gripper_aperture).

Observation space (~44 dims):
  skill_id (3)  +  arm_id (2)  +  joint_pos (7)  +  joint_vel (7)  +
  ee_pos (3)  +  ee_quat (4)  +  gripper_state (1)  +
  ball_ee_vec (3)  +  ball_vel (3)  +  ball_basket_vec (3)  +
  basket_clearance (1)  +  prev_action (7)

Action space (7 continuous in [-1, 1]):
  delta_pos (3)  +  delta_orient (3)  +  gripper (1)

Phase transitions:
  APPROACH(0) -> GRASP(1) -> DROP(2) -> DONE

After each episode (or successful drop), the active arm alternates and
objects are re-randomized with domain randomization.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# Ensure local imports work
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inverse_kinematics import (
    LEFT_ARM_HOME,
    RIGHT_ARM_HOME,
    LEFT_EE_SITE,
    RIGHT_EE_SITE,
    OpenArmIK,
)
from reward_calculator import RewardCalculator, RewardConfig


# =====================================================================
# Constants
# =====================================================================
PHASE_APPROACH = 0  # Approach & Pick
PHASE_GRASP = 1     # Precision Grasp & Lift
PHASE_DROP = 2      # Collision-Free Basket Drop
PHASE_NAMES = {0: "APPROACH", 1: "GRASP", 2: "DROP"}

TABLE_Z = 0.25
BALL_RADIUS = 0.022
BOX_DEPTH = 0.060
BOX_HALF_SIZE = 0.045
BOX_RIM_Z = TABLE_Z + BOX_DEPTH  # 0.310

# Reachable workspace bounds (same as pick_and_place.py)
WORKSPACE = {
    "left":  {"x": (0.16, 0.29), "y": (0.06, 0.24)},
    "right": {"x": (0.16, 0.29), "y": (-0.24, -0.06)},
}

# Residual action scaling
POS_RESIDUAL_SCALE = 0.015      # Max 15 mm position correction per step
ORIENT_RESIDUAL_SCALE = 0.10    # Max 0.1 rad orientation correction per step
GRIPPER_OPEN_MAX = 0.65         # Gripper open angle [rad]
GRIPPER_CLOSED = 0.05           # Gripper closed angle [rad]

# Phase transition thresholds
APPROACH_DIST_THRESH = 0.025     # Switch to GRASP when EE is within 25mm of ball
GRASP_LIFT_THRESH = 0.12         # Switch to DROP when ball is lifted 120mm above table
DROP_RELEASE_THRESH = 0.025      # Ball is over basket within 25mm XY

# Timing
PHYSICS_SUBSTEPS = 5             # MuJoCo substeps per env step
MAX_EPISODE_STEPS = 500          # Steps before truncation
MIN_SEPARATION = 0.125           # Minimum ball-box separation [m]
MAX_SEPARATION = 0.220           # Maximum ball-box separation [m]


class BimanualPickPlaceEnv(gym.Env):
    """Gymnasium environment for PPO training of bimanual pick-and-place.

    Implements the Residual RL formulation: the IK solver computes a
    base Cartesian target for the current phase, and the PPO policy
    outputs a small delta correction on top of it.

    Parameters
    ----------
    model_path : str
        Path to the MuJoCo XML file (openarm_pick_and_place.xml).
    reward_config : RewardConfig, optional
        Reward function coefficients.
    domain_randomization : bool
        Whether to apply domain randomization on reset.
    obs_noise_scale : float
        Standard deviation of additive Gaussian noise on observations.
    render_mode : str or None
        Gymnasium render mode ('human', 'rgb_array', or None).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: str = "openarm_pick_and_place.xml",
        reward_config: RewardConfig | None = None,
        domain_randomization: bool = True,
        obs_noise_scale: float = 0.002,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode

        # ---- Load MuJoCo model ----
        xml_path = Path(model_path)
        if not xml_path.is_absolute():
            xml_path = SCRIPT_DIR / xml_path
        xml_path = xml_path.resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"Model not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # ---- IK solver (uses its own internal data, never corrupts sim) ----
        self.ik = OpenArmIK(self.model, data=self.data)

        # ---- Reward calculator ----
        self.reward_calc = RewardCalculator(reward_config)

        # ---- Config ----
        self.domain_randomization = domain_randomization
        self.obs_noise_scale = obs_noise_scale

        # ---- Resolve joint/actuator/body/geom indices ----
        self._setup_indices()

        # ---- Observation & Action spaces ----
        # obs: skill(3) + arm(2) + q(7) + qdot(7) + ee_pos(3) + ee_quat(4) +
        #      gripper(1) + ball_ee(3) + ball_vel(3) + ball_basket(3) +
        #      clearance(1) + prev_action(7) = 44
        self.obs_dim = 44
        self.action_dim = 7  # delta_pos(3) + delta_orient(3) + gripper(1)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.action_dim,), dtype=np.float32
        )

        # ---- State variables ----
        self._active_arm: str = "left"
        self._phase: int = PHASE_APPROACH
        self._step_count: int = 0
        self._episode_count: int = 0
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._ball_pos = np.zeros(3)
        self._box_pos = np.zeros(3)
        self._gripper_open_ratio: float = 1.0
        self._ball_grasped: bool = False
        self._ball_released: bool = False

        # ---- Default random generator ----
        self._rng = np.random.default_rng()

        # ---- Renderer for rgb_array mode ----
        self._renderer = None

    def _setup_indices(self):
        """Resolve all MuJoCo name-to-id lookups once at init."""
        m = self.model

        # Arm joint qpos addresses (reuse from IK)
        self._left_qpos_adr = self.ik._left_qpos_adr
        self._right_qpos_adr = self.ik._right_qpos_adr

        # Finger qpos addresses
        self._left_finger_qpos = m.jnt_qposadr[
            m.joint("openarm_left_finger_joint1").id
        ]
        self._right_finger_qpos = m.jnt_qposadr[
            m.joint("openarm_right_finger_joint1").id
        ]

        # Actuator IDs for arm joints
        self._left_actuators = np.array([
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                              f"openarm_left_joint{i+1}")
            for i in range(7)
        ])
        self._right_actuators = np.array([
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                              f"openarm_right_joint{i+1}")
            for i in range(7)
        ])

        # Finger actuator IDs
        self._left_finger_act = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_ACTUATOR, "openarm_left_finger_joint1")
        self._right_finger_act = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_ACTUATOR, "openarm_right_finger_joint1")

        # EE site IDs
        self._left_site_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, LEFT_EE_SITE)
        self._right_site_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, RIGHT_EE_SITE)

        # Ball and box body/freejoint addresses
        ball_bid = m.body("ball").id
        self._ball_qpos_adr = m.jnt_qposadr[m.body_jntadr[ball_bid]]
        self._ball_dof_adr = m.jnt_dofadr[m.body_jntadr[ball_bid]]
        self._ball_body_id = ball_bid

        box_bid = m.body("box").id
        self._box_qpos_adr = m.jnt_qposadr[m.body_jntadr[box_bid]]
        self._box_dof_adr = m.jnt_dofadr[m.body_jntadr[box_bid]]

        # Ball geom ID
        self._ball_geom_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")

        # Geom ID sets for collision detection
        self._box_geom_ids = set()
        self._arm_geom_ids = set()
        for i in range(m.ngeom):
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
            if gname:
                if "box" in gname:
                    self._box_geom_ids.add(i)
                if any(k in gname for k in ["openarm", "finger", "ee_"]):
                    self._arm_geom_ids.add(i)

        # Active-arm specific geom IDs (for computing clearance per arm)
        self._left_arm_geom_ids = set()
        self._right_arm_geom_ids = set()
        for i in range(m.ngeom):
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
            if gname:
                if "left" in gname and any(k in gname for k in ["openarm", "finger", "ee_"]):
                    self._left_arm_geom_ids.add(i)
                elif "right" in gname and any(k in gname for k in ["openarm", "finger", "ee_"]):
                    self._right_arm_geom_ids.add(i)

        # Boolean per-geom lookup tables so contact tests are vectorised
        # (one pass over data.contact.geom instead of a Python loop).
        self._box_geom_mask = np.zeros(m.ngeom, dtype=bool)
        self._box_geom_mask[list(self._box_geom_ids)] = True
        self._arm_geom_mask = {}
        for side, ids in (("left", self._left_arm_geom_ids),
                          ("right", self._right_arm_geom_ids)):
            mask = np.zeros(m.ngeom, dtype=bool)
            mask[list(ids)] = True
            self._arm_geom_mask[side] = mask

        # Finger collision geoms per arm (resolved once, not every step)
        self._finger_geom_ids = {
            side: tuple(
                mujoco.mj_name2id(
                    m, mujoco.mjtObj.mjOBJ_GEOM,
                    f"openarm_{side}_ee_link{k}_collision")
                for k in (1, 2)
            )
            for side in ("left", "right")
        }

        # Box body id (ball id is stored above) for direct xpos indexing
        self._box_body_id = box_bid

        # DoF addresses for velocity readout
        self._left_dof_adr = self.ik._left_dof_adr
        self._right_dof_adr = self.ik._right_dof_adr

        # Weld constraint IDs for grasping
        self._left_weld_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_EQUALITY, "left_grasp_weld")
        self._right_weld_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_EQUALITY, "right_grasp_weld")

    # =================================================================
    # Gymnasium API
    # =================================================================

    def reset(self, *, seed=None, options=None):
        """Reset the environment for a new episode.

        Alternates the active arm each episode, re-randomizes object
        positions, and applies domain randomization if enabled.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Alternate arms
        if self._episode_count > 0:
            self._active_arm = "right" if self._active_arm == "left" else "left"
        self._episode_count += 1

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Arms to home pose
        self.data.qpos[self._left_qpos_adr] = LEFT_ARM_HOME
        self.data.qpos[self._right_qpos_adr] = RIGHT_ARM_HOME
        self.data.ctrl[self._left_actuators] = LEFT_ARM_HOME
        self.data.ctrl[self._right_actuators] = RIGHT_ARM_HOME

        # Open grippers
        self._set_gripper("left", 1.0)
        self._set_gripper("right", 1.0)
        self._gripper_open_ratio = 1.0

        # Randomize ball and box positions
        self._ball_pos, self._box_pos = self._sample_positions(self._active_arm)
        self._relocate_objects(self._ball_pos, self._box_pos)

        # Domain randomization
        if self.domain_randomization:
            self._apply_domain_randomization()

        # Settle physics
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # Sync IK solver
        self.ik.sync_from(self.data)

        # Reset state
        self._phase = PHASE_APPROACH
        self._step_count = 0
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._ball_grasped = False
        self._ball_released = False

        # Disable weld constraints
        self.model.eq_active0[self._left_weld_id] = 0
        self.model.eq_active0[self._right_weld_id] = 0

        obs = self._get_obs()
        info = {
            "active_arm": self._active_arm,
            "phase": PHASE_NAMES[self._phase],
            "ball_pos": self._ball_pos.copy(),
            "box_pos": self._box_pos.copy(),
        }
        return obs, info

    def step(self, action: np.ndarray):
        """Execute one environment step.

        1. Compute IK base target for current phase
        2. Apply PPO residual delta
        3. Execute via PD servos + MuJoCo physics
        4. Detect phase transitions
        5. Compute reward
        6. Check termination
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._step_count += 1

        # ---- 1. IK base target for current phase ----
        ik_target_pos, ik_target_quat = self._get_phase_target()

        # ---- 2. Apply residual corrections ----
        delta_pos = action[:3] * POS_RESIDUAL_SCALE
        delta_orient = action[3:6] * ORIENT_RESIDUAL_SCALE
        gripper_cmd = action[6]  # [-1, 1] -> [closed, open]

        # Corrected Cartesian target
        corrected_pos = ik_target_pos + delta_pos

        # For orientation: apply residual as small rotation to base quaternion
        corrected_quat = None
        if ik_target_quat is not None:
            corrected_quat = _apply_rotation_residual(ik_target_quat, delta_orient)

        # ---- 3. Solve IK and execute ----
        # (the solver recomputes kinematics itself, so skip it in the sync)
        self.ik.sync_from(self.data, forward=False)
        if self._active_arm == "left":
            ik_result = self.ik.solve_left(
                target_pos=corrected_pos, target_quat=corrected_quat,
                q_init=self.data.qpos[self._left_qpos_adr].copy()
            )
        else:
            ik_result = self.ik.solve_right(
                target_pos=corrected_pos, target_quat=corrected_quat,
                q_init=self.data.qpos[self._right_qpos_adr].copy()
            )

        # Apply joint commands
        qpos_adr = self._left_qpos_adr if self._active_arm == "left" else self._right_qpos_adr
        actuators = self._left_actuators if self._active_arm == "left" else self._right_actuators

        self.data.qpos[qpos_adr] = ik_result.q_arm
        self.data.ctrl[actuators] = ik_result.q_arm

        # Apply gripper command
        self._gripper_open_ratio = 0.5 * (gripper_cmd + 1.0)  # Map [-1,1] -> [0,1]
        self._set_gripper(self._active_arm, self._gripper_open_ratio)

        # Step physics
        for _ in range(PHYSICS_SUBSTEPS):
            # If ball is grasped, carry it with the gripper
            if self._ball_grasped and not self._ball_released:
                self._carry_ball()
            mujoco.mj_step(self.model, self.data)

        # ---- 4. Read state after physics ----
        ee_pos, ee_quat = self._get_ee_state()
        ball_pos = self.data.xpos[self._ball_body_id].copy()
        ball_vel = self.data.qvel[self._ball_dof_adr:self._ball_dof_adr + 6].copy()
        box_pos = self.data.xpos[self._box_body_id].copy()

        # Check contacts (each computed once per step and reused below)
        finger_contacts = self._check_finger_ball_contacts()
        arm_basket_contact = self._check_arm_box_contact()
        basket_clearance = self._compute_basket_clearance(
            ee_pos=ee_pos, arm_box_contact=arm_basket_contact)

        # Ball in box check
        ball_in_box = self._check_ball_in_box(ball_pos, box_pos)

        # ---- 5. Phase transitions ----
        self._update_phase(ee_pos, ball_pos, box_pos, finger_contacts, ball_in_box)

        # Handle grasp state
        if self._phase == PHASE_GRASP and not self._ball_grasped:
            if finger_contacts[0] and finger_contacts[1] and self._gripper_open_ratio < 0.3:
                self._ball_grasped = True

        if self._phase == PHASE_DROP and self._ball_grasped and not self._ball_released:
            if self._gripper_open_ratio > 0.7:
                self._ball_grasped = False
                self._ball_released = True

        # ---- 6. Compute reward ----
        reward, reward_info = self.reward_calc.compute(
            phase=self._phase,
            ee_pos=ee_pos,
            ball_pos=ball_pos,
            basket_pos=box_pos,
            ee_quat=ee_quat,
            ball_vel=ball_vel,
            finger_contacts=finger_contacts,
            basket_clearance=basket_clearance,
            arm_basket_contact=arm_basket_contact,
            gripper_open_ratio=self._gripper_open_ratio,
            action=action,
            prev_action=self._prev_action,
            table_z=TABLE_Z,
            box_rim_z=BOX_RIM_Z,
            ball_in_box=ball_in_box,
        )

        self._prev_action = action.copy()

        # ---- 7. Termination ----
        terminated = False
        truncated = False

        # Terminate on success
        if ball_in_box and self._ball_released:
            terminated = True

        # Terminate on catastrophic basket collision (optional hard fail)
        if arm_basket_contact and self._phase == PHASE_DROP:
            # Don't terminate, just penalize heavily (already done in reward)
            pass

        # Truncate on timeout
        if self._step_count >= MAX_EPISODE_STEPS:
            truncated = True

        # Truncate if ball fell off table
        if ball_pos[2] < TABLE_Z - 0.05:
            truncated = True

        # Nothing has touched the simulation since the readout above, so the
        # observation reuses it instead of recomputing EE state and contacts.
        obs = self._get_obs(ee_state=(ee_pos, ee_quat),
                            basket_clearance=basket_clearance)
        info = {
            "active_arm": self._active_arm,
            "phase": PHASE_NAMES[self._phase],
            "step": self._step_count,
            "ball_pos": ball_pos.copy(),
            "box_pos": box_pos.copy(),
            "ee_pos": ee_pos.copy(),
            "ball_grasped": self._ball_grasped,
            "ball_released": self._ball_released,
            "ball_in_box": ball_in_box,
            "arm_basket_contact": arm_basket_contact,
            "basket_clearance": basket_clearance,
            "ik_converged": ik_result.converged,
            "ik_pos_error": ik_result.pos_error,
        }
        info.update(reward_info)

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Render the current frame (for 'rgb_array' mode)."""
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # =================================================================
    # Observation construction
    # =================================================================

    def _get_obs(self, ee_state=None, basket_clearance=None) -> np.ndarray:
        """Build the full observation vector.

        ``ee_state`` / ``basket_clearance`` may be passed in when the caller
        has already computed them for the current simulation state.
        """
        # Skill one-hot
        skill_id = np.zeros(3, dtype=np.float32)
        skill_id[self._phase] = 1.0

        # Arm one-hot
        arm_id = np.array(
            [1.0, 0.0] if self._active_arm == "left" else [0.0, 1.0],
            dtype=np.float32
        )

        # Proprioception (in active arm's local frame)
        if self._active_arm == "left":
            joint_pos = self.data.qpos[self._left_qpos_adr].astype(np.float32)
            joint_vel = self.data.qvel[self._left_dof_adr].astype(np.float32)
        else:
            joint_pos = self.data.qpos[self._right_qpos_adr].astype(np.float32)
            joint_vel = self.data.qvel[self._right_dof_adr].astype(np.float32)

        ee_pos, ee_quat = ee_state if ee_state is not None else self._get_ee_state()
        gripper_state = np.array([self._gripper_open_ratio], dtype=np.float32)

        # Exteroception
        ball_pos = self.data.xpos[self._ball_body_id].copy()
        ball_vel = self.data.qvel[self._ball_dof_adr:self._ball_dof_adr + 3].copy()
        box_pos = self.data.xpos[self._box_body_id]

        ball_ee_vec = (ball_pos - ee_pos).astype(np.float32)
        ball_basket_vec = (box_pos - ball_pos).astype(np.float32)
        if basket_clearance is None:
            basket_clearance = self._compute_basket_clearance(ee_pos=ee_pos)
        basket_clearance = np.array([basket_clearance], dtype=np.float32)

        # Mirror observations for right arm (flip Y axis so policy sees
        # a canonical "left arm" view, doubling sample efficiency)
        if self._active_arm == "right":
            ee_pos, ee_quat, ball_ee_vec, ball_basket_vec = self._mirror_obs(
                ee_pos, ee_quat, ball_ee_vec, ball_basket_vec
            )
            ball_vel[1] = -ball_vel[1]  # Mirror Y velocity too

        obs = np.concatenate([
            skill_id,           # (3,)
            arm_id,             # (2,)
            joint_pos,          # (7,)
            joint_vel,          # (7,)
            ee_pos,             # (3,)
            ee_quat,            # (4,)
            gripper_state,      # (1,)
            ball_ee_vec,        # (3,)
            ball_vel,           # (3,)
            ball_basket_vec,    # (3,)
            basket_clearance,   # (1,)
            self._prev_action,  # (7,)
        ]).astype(np.float32)

        # Add observation noise
        if self.obs_noise_scale > 0:
            noise = self._rng.normal(0, self.obs_noise_scale, size=obs.shape)
            # Don't add noise to one-hot encodings
            noise[:5] = 0.0
            obs += noise.astype(np.float32)

        return obs

    def _mirror_obs(self, ee_pos, ee_quat, ball_ee_vec, ball_basket_vec):
        """Mirror Y-axis for right arm so the policy sees canonical left-arm frame."""
        ee_pos = ee_pos.copy()
        ee_quat = ee_quat.copy()
        ball_ee_vec = ball_ee_vec.copy()
        ball_basket_vec = ball_basket_vec.copy()

        ee_pos[1] = -ee_pos[1]
        ball_ee_vec[1] = -ball_ee_vec[1]
        ball_basket_vec[1] = -ball_basket_vec[1]

        # Mirror quaternion: negate y and z components for Y-reflection
        ee_quat[2] = -ee_quat[2]  # y component
        ee_quat[3] = -ee_quat[3]  # z component

        return ee_pos, ee_quat, ball_ee_vec, ball_basket_vec

    # =================================================================
    # Phase target generation (IK base trajectory)
    # =================================================================

    def _get_phase_target(self):
        """Compute the IK base Cartesian target for the current phase.

        Returns (target_pos, target_quat) where target_quat may be None.
        """
        ball_pos = self.data.xpos[self._ball_body_id].copy()
        box_pos = self.data.xpos[self._box_body_id].copy()

        if self._phase == PHASE_APPROACH:
            # Target: just above the ball, descending toward it
            dist = np.linalg.norm(self._get_ee_state()[0][:2] - ball_pos[:2])
            if dist > 0.05:
                # Far: aim above ball
                target = ball_pos.copy()
                target[2] += 0.10  # 100mm above
            else:
                # Close: descend to ball equator
                target = ball_pos.copy()
            return target, None

        elif self._phase == PHASE_GRASP:
            # Target: at ball position (grasping), then lift
            if not self._ball_grasped:
                target = ball_pos.copy()
            else:
                # Lift phase
                target = ball_pos.copy()
                target[2] = TABLE_Z + 0.170  # Lift height
            return target, None

        elif self._phase == PHASE_DROP:
            # Target: above box with safe clearance
            target = box_pos.copy()
            target[2] = BOX_RIM_Z + 0.080  # 80mm above rim
            return target, None

        return ball_pos.copy(), None

    # =================================================================
    # Phase transition logic
    # =================================================================

    def _update_phase(self, ee_pos, ball_pos, box_pos, finger_contacts, ball_in_box):
        """Check and execute phase transitions."""
        if self._phase == PHASE_APPROACH:
            # Transition to GRASP when EE is close to ball
            dist = np.linalg.norm(ee_pos - ball_pos)
            if dist < APPROACH_DIST_THRESH:
                self._phase = PHASE_GRASP

        elif self._phase == PHASE_GRASP:
            # Transition to DROP when ball is lifted high enough
            if self._ball_grasped:
                lift_height = ball_pos[2] - TABLE_Z
                if lift_height > GRASP_LIFT_THRESH:
                    self._phase = PHASE_DROP

        # DROP phase has no automatic transition — episode ends when
        # ball lands in box (terminated) or timeout (truncated)

    # =================================================================
    # Gripper control
    # =================================================================

    def _set_gripper(self, side: str, open_ratio: float):
        """Set gripper finger opening."""
        open_ratio = np.clip(open_ratio, 0.0, 1.0)
        if side == "left":
            val = GRIPPER_CLOSED + open_ratio * (GRIPPER_OPEN_MAX - GRIPPER_CLOSED)
            self.data.ctrl[self._left_finger_act] = val
            self.data.qpos[self._left_finger_qpos] = val
        else:
            val = -(GRIPPER_CLOSED + open_ratio * (GRIPPER_OPEN_MAX - GRIPPER_CLOSED))
            self.data.ctrl[self._right_finger_act] = val
            self.data.qpos[self._right_finger_qpos] = val

    # =================================================================
    # Ball carrying (kinematic attachment during grasp)
    # =================================================================

    def _carry_ball(self):
        """Kinematically attach ball to end-effector during grasp."""
        ee_pos, _ = self._get_ee_state()
        self.data.qpos[self._ball_qpos_adr:self._ball_qpos_adr + 3] = ee_pos
        self.data.qvel[self._ball_dof_adr:self._ball_dof_adr + 6] = 0.0

    # =================================================================
    # State queries
    # =================================================================

    def _get_ee_state(self):
        """Get end-effector position and quaternion for active arm."""
        if self._active_arm == "left":
            pos = self.data.site_xpos[self._left_site_id].copy()
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, self.data.site_xmat[self._left_site_id])
        else:
            pos = self.data.site_xpos[self._right_site_id].copy()
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, self.data.site_xmat[self._right_site_id])
        return pos.astype(np.float32), quat.astype(np.float32)

    def _contact_geoms(self) -> np.ndarray:
        """Geom-id pairs of the active contacts, shape (ncon, 2)."""
        return self.data.contact.geom[:self.data.ncon]

    def _check_finger_ball_contacts(self) -> tuple[bool, bool]:
        """Check if each finger is in contact with the ball."""
        if self.data.ncon == 0:
            return (False, False)
        geoms = self._contact_geoms()
        f1_id, f2_id = self._finger_geom_ids[self._active_arm]
        with_ball = geoms[(geoms == self._ball_geom_id).any(axis=1)]
        return (bool((with_ball == f1_id).any()),
                bool((with_ball == f2_id).any()))

    def _check_arm_box_contact(self) -> bool:
        """Check if any arm/gripper geom is in contact with box."""
        if self.data.ncon == 0:
            return False
        geoms = self._contact_geoms()
        g1, g2 = geoms[:, 0], geoms[:, 1]
        arm = self._arm_geom_mask[self._active_arm]
        box = self._box_geom_mask
        return bool(((arm[g1] & box[g2]) | (arm[g2] & box[g1])).any())

    def _compute_basket_clearance(self, ee_pos=None, arm_box_contact=None) -> float:
        """Estimate minimum distance from active arm to basket.

        Uses end-effector-to-box-center distance as a proxy.
        For higher fidelity, one would use mj_distanceGeoms or iterate
        contact.dist, but this is fast and sufficient for reward shaping.

        ``ee_pos`` / ``arm_box_contact`` may be passed in when already known
        for the current simulation state, to avoid recomputing them.
        """
        if ee_pos is None:
            ee_pos, _ = self._get_ee_state()
        if arm_box_contact is None:
            arm_box_contact = self._check_arm_box_contact()

        # Actual contact means zero clearance
        if arm_box_contact:
            return 0.0

        box_pos = self.data.xpos[self._box_body_id]
        # Distance from EE to nearest box wall edge (approximate)
        dx = max(0.0, abs(ee_pos[0] - box_pos[0]) - BOX_HALF_SIZE)
        dy = max(0.0, abs(ee_pos[1] - box_pos[1]) - BOX_HALF_SIZE)
        dz = max(0.0, ee_pos[2] - (box_pos[2] + BOX_DEPTH))
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))

    def _check_ball_in_box(self, ball_pos, box_pos) -> bool:
        """Check if ball is inside the box aperture."""
        h_dist = np.hypot(ball_pos[0] - box_pos[0], ball_pos[1] - box_pos[1])
        v_dist = ball_pos[2] - box_pos[2]
        in_aperture = h_dist < (BOX_HALF_SIZE * 0.9)
        on_floor = 0.01 < v_dist < (BOX_DEPTH + 0.035)
        return in_aperture and on_floor

    # =================================================================
    # Object placement & randomization
    # =================================================================

    def _sample_positions(self, arm: str):
        """Sample random non-overlapping ball and box positions."""
        ws = WORKSPACE[arm]
        x_range, y_range = ws["x"], ws["y"]

        for _ in range(300):
            target_dist = self._rng.uniform(MIN_SEPARATION, MAX_SEPARATION)
            ball_x = self._rng.uniform(*x_range)
            ball_y = self._rng.uniform(*y_range)
            theta = self._rng.uniform(0, 2 * np.pi)
            box_x = ball_x + target_dist * np.cos(theta)
            box_y = ball_y + target_dist * np.sin(theta)

            if (x_range[0] <= box_x <= x_range[1]) and \
               (y_range[0] <= box_y <= y_range[1]):
                ball_pos = np.array([ball_x, ball_y, TABLE_Z + BALL_RADIUS])
                box_pos = np.array([box_x, box_y, TABLE_Z])
                return ball_pos, box_pos

        # Fallback: direct rejection sampling
        for _ in range(300):
            ball_x = self._rng.uniform(*x_range)
            ball_y = self._rng.uniform(*y_range)
            box_x = self._rng.uniform(*x_range)
            box_y = self._rng.uniform(*y_range)

            dist = np.hypot(ball_x - box_x, ball_y - box_y)
            if dist >= MIN_SEPARATION:
                ball_pos = np.array([ball_x, ball_y, TABLE_Z + BALL_RADIUS])
                box_pos = np.array([box_x, box_y, TABLE_Z])
                return ball_pos, box_pos

        raise RuntimeError("Failed to sample valid positions after 600 attempts.")

    def _relocate_objects(self, ball_pos, box_pos):
        """Place ball and box at the specified positions."""
        d = self.data
        d.qpos[self._ball_qpos_adr:self._ball_qpos_adr + 3] = ball_pos
        d.qpos[self._ball_qpos_adr + 3:self._ball_qpos_adr + 7] = [1, 0, 0, 0]
        d.qvel[self._ball_dof_adr:self._ball_dof_adr + 6] = 0.0

        d.qpos[self._box_qpos_adr:self._box_qpos_adr + 3] = box_pos
        d.qpos[self._box_qpos_adr + 3:self._box_qpos_adr + 7] = [1, 0, 0, 0]
        d.qvel[self._box_dof_adr:self._box_dof_adr + 6] = 0.0

        mujoco.mj_forward(self.model, d)

    def _apply_domain_randomization(self):
        """Randomize physical properties for sim-to-real transfer."""
        m = self.model

        # Ball mass randomization (+/- 30%)
        ball_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        base_mass = 0.03
        m.body_mass[self._ball_body_id] = base_mass * self._rng.uniform(0.7, 1.3)

        # Ball friction randomization
        base_friction = np.array([1.8, 0.05, 0.001])
        friction_scale = self._rng.uniform(0.8, 1.2, size=3)
        m.geom_friction[ball_geom] = base_friction * friction_scale

        # Box position jitter (+/- 5mm)
        box_jitter = self._rng.uniform(-0.005, 0.005, size=2)
        self.data.qpos[self._box_qpos_adr:self._box_qpos_adr + 2] += box_jitter

        mujoco.mj_forward(m, self.data)


# =====================================================================
# Helper: Apply small rotation residual to a quaternion
# =====================================================================
def _apply_rotation_residual(base_quat: np.ndarray, delta_rot: np.ndarray) -> np.ndarray:
    """Apply a small rotation vector delta to a base quaternion.

    Parameters
    ----------
    base_quat : ndarray (4,)  -- (w, x, y, z)
    delta_rot : ndarray (3,)  -- rotation vector (axis * angle)

    Returns
    -------
    result_quat : ndarray (4,) -- (w, x, y, z)
    """
    angle = np.linalg.norm(delta_rot)
    if angle < 1e-8:
        return base_quat.copy()

    axis = delta_rot / angle
    half_angle = angle / 2.0
    dq = np.array([
        np.cos(half_angle),
        axis[0] * np.sin(half_angle),
        axis[1] * np.sin(half_angle),
        axis[2] * np.sin(half_angle),
    ])

    # Quaternion multiplication: dq * base_quat
    w1, x1, y1, z1 = dq
    w2, x2, y2, z2 = base_quat
    result = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

    # Normalize
    result /= np.linalg.norm(result) + 1e-16
    return result


# =====================================================================
# Registration (optional)
# =====================================================================
def register_env():
    """Register this environment with Gymnasium."""
    gym.register(
        id="BimanualPickPlace-v0",
        entry_point="bimanual_env:BimanualPickPlaceEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
    )
