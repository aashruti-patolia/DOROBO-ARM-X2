#!/usr/bin/env python3
"""
Inverse Kinematics for the OpenArm v2.0 Bimanual Robot
=======================================================
Damped Least-Squares (Levenberg-Marquardt) IK solver using MuJoCo,
adapted from the OpenArm project methodology for this dual-arm setup.

Supports:
  - Single-arm IK (position only, position + orientation, or 6-DoF)
  - Bimanual / coordinated dual-arm IK (both arms solved simultaneously)
  - Gripper opening control (decoupled from arm IK)
  - Joint-limit clamping with configurable margin
  - Nullspace posture bias to keep the arm in ergonomic configurations
  - Collision-aware self-collision avoidance via joint-limit padding
  - Cartesian waypoint trajectory generation

References:
  - OpenArm v2.0 IK approach (7-DoF redundant arm with QDD actuators)
  - MuJoCo Jacobian-based IK  (mj_jacSite / mj_jacBody)
  - S. R. Buss, "Introduction to Inverse Kinematics with Jacobian
    Transpose, Pseudoinverse and Damped Least-Squares methods", 2004

Usage:
    from inverse_kinematics import OpenArmIK
    ik = OpenArmIK("openarm_v20_mujoco/openarm_v20.xml")

    # Solve left arm to reach a target position
    q_solution = ik.solve_left(target_pos=[0.25, 0.15, 0.35])

    # Solve both arms with full 6-DoF targets
    q_left, q_right = ik.solve_bimanual(
        left_pos=[0.20, 0.20, 0.30], left_quat=[1, 0, 0, 0],
        right_pos=[0.20, -0.20, 0.30], right_quat=[1, 0, 0, 0],
    )
"""
from __future__ import annotations

import numpy as np
import mujoco
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# =====================================================================
# Configuration
# =====================================================================
@dataclass
class IKConfig:
    """Solver hyper-parameters."""
    max_iterations: int = 100        # Max Newton-Raphson iterations
    pos_tolerance: float = 1e-4      # Position convergence [m]
    rot_tolerance: float = 1e-3      # Orientation convergence [rad]
    step_size: float = 0.5           # Line-search scaling (0 < α ≤ 1)
    damping: float = 1e-3            # Levenberg-Marquardt regularisation λ
    joint_limit_margin: float = 0.05 # Safety margin from joint limits [rad]
    nullspace_weight: float = 0.5    # Weight for nullspace posture bias
    max_linear_step: float = 0.05    # Max Cartesian position step [m]
    max_angular_step: float = 0.25   # Max Cartesian orientation step [rad]


# =====================================================================
# Kinematic chain definition — matches openarm_v20.xml
# =====================================================================
# Left arm: joints 0–6 (qpos 0–6),  finger: joint 7–8 (qpos 7–8)
# Right arm: joints 9–15 (qpos 9–15), finger: joint 16–17 (qpos 16–17)

LEFT_ARM_JOINT_NAMES = [
    "openarm_left_joint1",   # Shoulder Pan      (Y axis)
    "openarm_left_joint2",   # Shoulder Lift      (-X axis)
    "openarm_left_joint3",   # Shoulder Rotation  (-Z axis)
    "openarm_left_joint4",   # Elbow Flex         (-Y axis)
    "openarm_left_joint5",   # Wrist Roll         (-Z axis)
    "openarm_left_joint6",   # Wrist Pitch        (-Y axis)
    "openarm_left_joint7",   # Wrist Rotation     (X axis)
]

RIGHT_ARM_JOINT_NAMES = [
    "openarm_right_joint1",  # Shoulder Pan      (-Y axis)
    "openarm_right_joint2",  # Shoulder Lift      (-X axis)
    "openarm_right_joint3",  # Shoulder Rotation  (-Z axis)
    "openarm_right_joint4",  # Elbow Flex         (-Y axis)
    "openarm_right_joint5",  # Wrist Roll         (-Z axis)
    "openarm_right_joint6",  # Wrist Pitch        (Y axis)
    "openarm_right_joint7",  # Wrist Rotation     (X axis)
]

LEFT_EE_SITE  = "left_ee_site"
RIGHT_EE_SITE = "right_ee_site"

# Ergonomic nullspace target: roughly a natural resting pose
LEFT_ARM_HOME  = np.array([0.0, -0.5, 0.0, 1.2, 0.0, 0.0, 0.0])
RIGHT_ARM_HOME = np.array([0.0, 0.5, 0.0, 1.2, 0.0, 0.0, 0.0])

N_ARM_JOINTS = 7


# =====================================================================
# IK Result
# =====================================================================
@dataclass
class IKResult:
    """Return value from solve_*() methods."""
    q_arm: np.ndarray               # Joint angles [7] in radians
    converged: bool                  # Whether the solver converged
    iterations: int                  # Iterations taken
    pos_error: float                 # Final position error [m]
    rot_error: float                 # Final orientation error [rad]
    ee_pos: np.ndarray              # Achieved end-effector position [3]
    ee_quat: np.ndarray             # Achieved end-effector quaternion [4]


@dataclass
class BimanualIKResult:
    """Return value from solve_bimanual()."""
    left: IKResult
    right: IKResult
    converged: bool                  # True only if both arms converged


# =====================================================================
# Helper functions
# =====================================================================
def _quat_error(quat_target: np.ndarray, quat_current: np.ndarray) -> np.ndarray:
    """Compute orientation error as a 3D rotation vector (axis × angle).

    Uses the quaternion error: e = 2 * (q_target ⊗ q_current^{-1}).xyz
    Sign-corrected so the shortest path is taken.
    """
    # Ensure unit quaternions (w, x, y, z format in MuJoCo)
    qt = quat_target / (np.linalg.norm(quat_target) + 1e-16)
    qc = quat_current / (np.linalg.norm(quat_current) + 1e-16)

    # Quaternion conjugate of current
    qc_conj = np.array([qc[0], -qc[1], -qc[2], -qc[3]])

    # Error quaternion: q_err = q_target * q_current_conj
    # Hamilton product: (a1, b1, c1, d1) * (a2, b2, c2, d2)
    q_err = np.zeros(4)
    q_err[0] = qt[0]*qc_conj[0] - qt[1]*qc_conj[1] - qt[2]*qc_conj[2] - qt[3]*qc_conj[3]
    q_err[1] = qt[0]*qc_conj[1] + qt[1]*qc_conj[0] + qt[2]*qc_conj[3] - qt[3]*qc_conj[2]
    q_err[2] = qt[0]*qc_conj[2] - qt[1]*qc_conj[3] + qt[2]*qc_conj[0] + qt[3]*qc_conj[1]
    q_err[3] = qt[0]*qc_conj[3] + qt[1]*qc_conj[2] - qt[2]*qc_conj[1] + qt[3]*qc_conj[0]

    # Ensure shortest path (if w < 0, negate)
    if q_err[0] < 0:
        q_err = -q_err

    # Small-angle approximation: rotation vector ≈ 2 * (x, y, z)
    return 2.0 * q_err[1:4]


def _clamp_joint_angles(q: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                        limited: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Clamp joint angles within [lower + margin, upper - margin] for limited joints.

    For unlimited joints (limited=False), bounds are [-inf, inf], preventing
    unlimited joints from being frozen at 0.
    """
    lo = np.where(limited, lower + margin, -np.inf)
    hi = np.where(limited, upper - margin, np.inf)
    return np.clip(q, lo, hi)


def _limit_step(dx: np.ndarray, max_pos: float, max_rot: float) -> np.ndarray:
    """Limit the Cartesian displacement to prevent overshooting."""
    pos_part = dx[:3]
    rot_part = dx[3:6] if len(dx) > 3 else np.zeros(3)

    pos_norm = np.linalg.norm(pos_part)
    rot_norm = np.linalg.norm(rot_part)

    if pos_norm > max_pos:
        pos_part = pos_part * (max_pos / pos_norm)
    if rot_norm > max_rot:
        rot_part = rot_part * (max_rot / rot_norm)

    if len(dx) > 3:
        return np.concatenate([pos_part, rot_part])
    return pos_part


# =====================================================================
# Main IK Solver Class
# =====================================================================
class OpenArmIK:
    """Inverse Kinematics solver for the OpenArm v2.0 bimanual robot.

    Uses the Damped Least-Squares (Levenberg-Marquardt) method with
    MuJoCo's analytical Jacobian computation. Each arm is a 7-DoF
    redundant chain, with the extra degree of freedom absorbed by
    nullspace posture optimisation.

    Parameters
    ----------
    model_or_path : str, Path, or mujoco.MjModel
        Path to the MuJoCo XML model file (openarm_v20.xml) or an existing MjModel.
    data : mujoco.MjData, optional
        Optional MjData to sync initial joint angles from. An internal, dedicated
        MjData instance is always used so IK never corrupts live simulation state.
    config : IKConfig, optional
        Solver configuration. Uses defaults if not provided.
    """

    def __init__(
        self,
        model_or_path: str | Path | mujoco.MjModel,
        data: mujoco.MjData | None = None,
        config: IKConfig | None = None,
    ):
        if isinstance(model_or_path, mujoco.MjModel):
            self.model = model_or_path
            self.model_path = None
        else:
            self.model_path = Path(model_or_path).resolve()
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))

        self.cfg = config or IKConfig()

        # Dedicated private MjData workspace for IK kinematics.
        # Guarantees IK never overwrites or corrupts live simulation data.
        self.data = mujoco.MjData(self.model)
        if data is not None:
            self.data.qpos[:] = data.qpos[:]
            mujoco.mj_forward(self.model, self.data)

        # Resolve site IDs for end-effector TCP points
        self._left_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, LEFT_EE_SITE)
        self._right_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, RIGHT_EE_SITE)

        if self._left_site_id < 0 or self._right_site_id < 0:
            raise ValueError(
                f"EE sites not found in model. Expected '{LEFT_EE_SITE}' "
                f"and '{RIGHT_EE_SITE}'. Available sites: "
                + str([mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
                       for i in range(self.model.nsite)]))

        # Resolve joint IDs and build index maps
        self._left_jnt_ids = self._resolve_joints(LEFT_ARM_JOINT_NAMES)
        self._right_jnt_ids = self._resolve_joints(RIGHT_ARM_JOINT_NAMES)

        # Qpos addresses for each arm's joints
        self._left_qpos_adr = np.array(
            [self.model.jnt_qposadr[j] for j in self._left_jnt_ids])
        self._right_qpos_adr = np.array(
            [self.model.jnt_qposadr[j] for j in self._right_jnt_ids])

        # DoF addresses for Jacobian column extraction
        self._left_dof_adr = np.array(
            [self.model.jnt_dofadr[j] for j in self._left_jnt_ids])
        self._right_dof_adr = np.array(
            [self.model.jnt_dofadr[j] for j in self._right_jnt_ids])

        # Joint limits and limited flags [7] for each arm
        self._left_jnt_lower = np.array(
            [self.model.jnt_range[j, 0] for j in self._left_jnt_ids])
        self._left_jnt_upper = np.array(
            [self.model.jnt_range[j, 1] for j in self._left_jnt_ids])
        self._left_jnt_limited = np.array(
            [bool(self.model.jnt_limited[j]) for j in self._left_jnt_ids], dtype=bool)

        self._right_jnt_lower = np.array(
            [self.model.jnt_range[j, 0] for j in self._right_jnt_ids])
        self._right_jnt_upper = np.array(
            [self.model.jnt_range[j, 1] for j in self._right_jnt_ids])
        self._right_jnt_limited = np.array(
            [bool(self.model.jnt_limited[j]) for j in self._right_jnt_ids], dtype=bool)

        # Pre-allocate full Jacobian buffers (3 × nv)
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    def _fk(self):
        """Update the kinematic quantities IK needs in the scratch data.

        mj_kinematics gives site poses; mj_comPos gives the subtree CoMs and
        cdof that mj_jacSite reads. This is all the solver uses, and it is
        far cheaper than mj_forward (no collision, dynamics or actuation).
        """
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def sync_from(self, sim_data: mujoco.MjData, forward: bool = True):
        """Synchronize joint positions from a live simulation data context.

        Pass ``forward=False`` when a solve_*() call follows immediately:
        the solver recomputes kinematics itself, so the update here would
        be thrown away.
        """
        self.data.qpos[:] = sim_data.qpos[:]
        if forward:
            self._fk()

    def sync_to(self, sim_data: mujoco.MjData):
        """Copy solved joint positions back to a live simulation data context."""
        sim_data.qpos[:] = self.data.qpos[:]
        mujoco.mj_forward(self.model, sim_data)

    def _resolve_joints(self, names: list[str]) -> list[int]:
        """Resolve joint names to MuJoCo IDs, raising on missing."""
        ids = []
        for name in names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in model.")
            ids.append(jid)
        return ids

    # -----------------------------------------------------------------
    # Jacobian extraction
    # -----------------------------------------------------------------
    def _get_arm_jacobian(self, site_id: int, dof_adr: np.ndarray,
                          position_only: bool = False) -> np.ndarray:
        """Compute the end-effector Jacobian for the specified arm joints.

        Returns shape (3, 7) if position_only, else (6, 7).
        """
        self._jacp[:] = 0
        self._jacr[:] = 0
        mujoco.mj_jacSite(
            self.model, self.data, self._jacp, self._jacr, site_id)

        # Extract only the columns corresponding to this arm's DoFs
        Jp = self._jacp[:, dof_adr]  # (3, 7) position Jacobian
        if position_only:
            return Jp

        Jr = self._jacr[:, dof_adr]  # (3, 7) rotation Jacobian
        return np.vstack([Jp, Jr])   # (6, 7)

    # -----------------------------------------------------------------
    # Forward kinematics query
    # -----------------------------------------------------------------
    
    def forward_kinematics(self, q_left: np.ndarray | None = None,
                           q_right: np.ndarray | None = None
                           ) -> dict[str, dict[str, np.ndarray]]:
        """Compute the end-effector position and orientation for given joint angles.

        Parameters
        ----------
        q_left : array-like of shape (7,), optional
            Left arm joint angles. If None, leaves left arm in current internal state.
        q_right : array-like of shape (7,), optional
            Right arm joint angles. If None, leaves right arm in current internal state.

        Returns
        -------
        dict with keys 'left' and/or 'right', each containing
        'pos' (3,) and 'quat' (4,).
        """
        if q_left is not None:
            q_left = np.asarray(q_left, dtype=np.float64)
            self.data.qpos[self._left_qpos_adr] = q_left
        if q_right is not None:
            q_right = np.asarray(q_right, dtype=np.float64)
            self.data.qpos[self._right_qpos_adr] = q_right

        self._fk()

        result = {}
        if q_left is not None:
            result["left"] = {
                "pos": self.data.site_xpos[self._left_site_id].copy(),
                "quat": np.zeros(4),
            }
            mujoco.mju_mat2Quat(
                result["left"]["quat"],
                self.data.site_xmat[self._left_site_id])

        if q_right is not None:
            result["right"] = {
                "pos": self.data.site_xpos[self._right_site_id].copy(),
                "quat": np.zeros(4),
            }
            mujoco.mju_mat2Quat(
                result["right"]["quat"],
                self.data.site_xmat[self._right_site_id])

        return result

    # -----------------------------------------------------------------
    # Core single-arm IK solver
    # -----------------------------------------------------------------
    def _solve_arm(
        self,
        site_id: int,
        qpos_adr: np.ndarray,
        dof_adr: np.ndarray,
        jnt_lower: np.ndarray,
        jnt_upper: np.ndarray,
        jnt_limited: np.ndarray,
        home_pose: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
    ) -> IKResult:
        """Damped Least-Squares IK for a single 7-DoF arm.

        Parameters
        ----------
        site_id : int
            MuJoCo site ID for the end-effector TCP.
        qpos_adr : ndarray (7,)
            Indices into data.qpos for this arm's joints.
        dof_adr : ndarray (7,)
            Indices into Jacobian columns for this arm's DoFs.
        jnt_lower, jnt_upper : ndarray (7,)
            Joint angle limits.
        jnt_limited : ndarray (7,) of bool
            Whether each joint has limits enabled.
        home_pose : ndarray (7,)
            Nullspace target (ergonomic home configuration).
        target_pos : ndarray (3,)
            Desired end-effector position in world frame.
        target_quat : ndarray (4,) or None
            Desired orientation as quaternion [w, x, y, z].
            If None, orientation is unconstrained (position-only IK).
        q_init : ndarray (7,) or None
            Initial guess. If None, uses home_pose.

        Returns
        -------
        IKResult
        """
        position_only = target_quat is None
        target_pos = np.asarray(target_pos, dtype=np.float64)
        if target_quat is not None:
            target_quat = np.asarray(target_quat, dtype=np.float64)
            target_quat /= np.linalg.norm(target_quat) + 1e-16

        cfg = self.cfg

        # Initialise joint configuration for this arm only.
        # Other arm and torso/finger joints remain undisturbed.
        # Clamp bounds are constant for the whole solve; build them once.
        q_lo = np.where(jnt_limited, jnt_lower + cfg.joint_limit_margin, -np.inf)
        q_hi = np.where(jnt_limited, jnt_upper - cfg.joint_limit_margin, np.inf)

        q_start = (np.asarray(q_init, dtype=np.float64)
                   if q_init is not None else home_pose)

        # Views into the scratch MjData, fetched once: each attribute access
        # on MjData builds a fresh array wrapper, which adds up inside the
        # loop. The underlying buffers do not move, so the views stay live.
        qpos = self.data.qpos
        site_pos = self.data.site_xpos[site_id]

        q = np.clip(q_start, q_lo, q_hi)
        qpos[qpos_adr] = q

        # Zero out velocities in scratch data
        self.data.qvel[:] = 0.0

        self._fk()

        lam2 = cfg.damping ** 2
        w = cfg.nullspace_weight

        converged = False
        iterations = 0
        pos_err_norm = np.inf
        rot_err_norm = 0.0

        for it in range(cfg.max_iterations):
            iterations = it + 1

            # Compute task-space error
            pos_err = target_pos - site_pos
            pos_err_norm = float(np.sqrt(pos_err.dot(pos_err)))

            if position_only:
                rot_err_norm = 0.0

                # Check convergence
                if pos_err_norm < cfg.pos_tolerance:
                    converged = True
                    break

                # Limit step size
                dx = pos_err
                if pos_err_norm > cfg.max_linear_step:
                    dx = pos_err * (cfg.max_linear_step / pos_err_norm)

                # Get position-only Jacobian (3 × 7)
                J = self._get_arm_jacobian(site_id, dof_adr, position_only=True)
            else:
                ee_quat = np.zeros(4)
                mujoco.mju_mat2Quat(ee_quat, self.data.site_xmat[site_id])
                rot_err = _quat_error(target_quat, ee_quat)
                rot_err_norm = np.linalg.norm(rot_err)
                dx = np.concatenate([pos_err, rot_err])

                # Check convergence
                if pos_err_norm < cfg.pos_tolerance and rot_err_norm < cfg.rot_tolerance:
                    converged = True
                    break

                # Limit step size
                dx = _limit_step(dx, cfg.max_linear_step, cfg.max_angular_step)

                # Get full Jacobian (6 × 7)
                J = self._get_arm_jacobian(site_id, dof_adr, position_only=False)

            # Damped Least-Squares: dq = J^T (J J^T + λ² I)^{-1} dx
            # plus a nullspace posture bias w * N p, with N = I - J^+ J and
            # J^+ = J^T (JJ^T + λ²I)^{-1}.  Expanding N p = p - J^+ (J p)
            # lets both terms share one solve of the damped system with two
            # right-hand sides, instead of forming J^+ and N explicitly:
            #   dq = J^T (JJ^T + λ²I)^{-1} (dx - w J p) + w p
            JJT = J @ J.T
            JJT.flat[::JJT.shape[0] + 1] += lam2
            posture_err = home_pose - q
            dq = J.T @ np.linalg.solve(JJT, dx - w * (J @ posture_err))
            dq += w * posture_err

            # Apply step with line-search scaling, clamp to joint limits
            # (unlimited joints have infinite bounds), write back, and
            # update kinematics.
            q = np.clip(q + cfg.step_size * dq, q_lo, q_hi)
            qpos[qpos_adr] = q
            self._fk()

        # Final state
        ee_pos_final = site_pos.copy()
        ee_quat_final = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat_final, self.data.site_xmat[site_id])

        return IKResult(
            q_arm=q.copy(),
            converged=converged,
            iterations=iterations,
            pos_error=pos_err_norm,
            rot_error=rot_err_norm,
            ee_pos=ee_pos_final,
            ee_quat=ee_quat_final,
        )

    # -----------------------------------------------------------------
    # Public API: Single Arm
    # -----------------------------------------------------------------
    def solve_left(
        self,
        target_pos: np.ndarray | list[float],
        target_quat: np.ndarray | list[float] | None = None,
        q_init: np.ndarray | list[float] | None = None,
    ) -> IKResult:
        """Solve IK for the left arm.

        Parameters
        ----------
        target_pos : (3,) desired TCP position [x, y, z] in world frame.
        target_quat : (4,) desired orientation [w, x, y, z], or None for
                      position-only IK (orientation unconstrained).
        q_init : (7,) initial joint angle guess, or None for home pose.

        Returns
        -------
        IKResult with q_arm, converged flag, errors, and achieved pose.
        """
        return self._solve_arm(
            site_id=self._left_site_id,
            qpos_adr=self._left_qpos_adr,
            dof_adr=self._left_dof_adr,
            jnt_lower=self._left_jnt_lower,
            jnt_upper=self._left_jnt_upper,
            jnt_limited=self._left_jnt_limited,
            home_pose=LEFT_ARM_HOME,
            target_pos=np.asarray(target_pos, dtype=np.float64),
            target_quat=(np.asarray(target_quat, dtype=np.float64)
                         if target_quat is not None else None),
            q_init=(np.asarray(q_init, dtype=np.float64)
                    if q_init is not None else None),
        )

    def solve_right(
        self,
        target_pos: np.ndarray | list[float],
        target_quat: np.ndarray | list[float] | None = None,
        q_init: np.ndarray | list[float] | None = None,
    ) -> IKResult:
        """Solve IK for the right arm.

        Parameters
        ----------
        target_pos : (3,) desired TCP position [x, y, z] in world frame.
        target_quat : (4,) desired orientation [w, x, y, z], or None.
        q_init : (7,) initial joint angle guess, or None for home pose.

        Returns
        -------
        IKResult
        """
        return self._solve_arm(
            site_id=self._right_site_id,
            qpos_adr=self._right_qpos_adr,
            dof_adr=self._right_dof_adr,
            jnt_lower=self._right_jnt_lower,
            jnt_upper=self._right_jnt_upper,
            jnt_limited=self._right_jnt_limited,
            home_pose=RIGHT_ARM_HOME,
            target_pos=np.asarray(target_pos, dtype=np.float64),
            target_quat=(np.asarray(target_quat, dtype=np.float64)
                         if target_quat is not None else None),
            q_init=(np.asarray(q_init, dtype=np.float64)
                    if q_init is not None else None),
        )

    # -----------------------------------------------------------------
    # Public API: Bimanual (Coordinated Dual-Arm)
    # -----------------------------------------------------------------
    def solve_bimanual(
        self,
        left_pos: np.ndarray | list[float],
        right_pos: np.ndarray | list[float],
        left_quat: np.ndarray | list[float] | None = None,
        right_quat: np.ndarray | list[float] | None = None,
        q_left_init: np.ndarray | list[float] | None = None,
        q_right_init: np.ndarray | list[float] | None = None,
    ) -> BimanualIKResult:
        """Solve IK for both arms simultaneously.

        Each arm is solved in the shared internal kinematic context.
        Because _solve_arm does not reset other joints, the left arm's
        solved pose is preserved when the right arm is solved.

        Parameters
        ----------
        left_pos, right_pos : (3,) desired TCP positions.
        left_quat, right_quat : (4,) desired orientations, or None.
        q_left_init, q_right_init : (7,) initial guesses, or None.

        Returns
        -------
        BimanualIKResult containing IKResult for each arm.
        """
        # Solve left arm
        left_result = self.solve_left(
            target_pos=left_pos, target_quat=left_quat, q_init=q_left_init)

        # Left solution remains in self.data.qpos[self._left_qpos_adr].
        # Solve right arm (without wiping left arm solution)
        right_result = self.solve_right(
            target_pos=right_pos, target_quat=right_quat, q_init=q_right_init)

        # Update both arm solutions in self.data and compute forward kinematics
        self.data.qpos[self._left_qpos_adr] = left_result.q_arm
        self.data.qpos[self._right_qpos_adr] = right_result.q_arm
        self._fk()

        return BimanualIKResult(
            left=left_result,
            right=right_result,
            converged=left_result.converged and right_result.converged,
        )

    # -----------------------------------------------------------------
    # Cartesian Trajectory Generation
    # -----------------------------------------------------------------
    def cartesian_trajectory(
        self,
        side: str,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        start_quat: np.ndarray | None = None,
        end_quat: np.ndarray | None = None,
        n_waypoints: int = 20,
        q_init: np.ndarray | None = None,
    ) -> list[IKResult]:
        """Generate a Cartesian straight-line trajectory with IK at each waypoint.

        Linearly interpolates position and uses SLERP for orientation.
        Each waypoint IK is seeded from the previous solution for continuity.

        Parameters
        ----------
        side : 'left' or 'right'
        start_pos, end_pos : (3,) start and end TCP positions
        start_quat, end_quat : (4,) start and end orientations, or None
        n_waypoints : number of interpolation steps
        q_init : initial joint guess for the first waypoint

        Returns
        -------
        List of IKResult, one per waypoint.
        """
        solver = self.solve_left if side == "left" else self.solve_right

        start_pos = np.asarray(start_pos, dtype=np.float64)
        end_pos = np.asarray(end_pos, dtype=np.float64)

        use_orientation = (start_quat is not None and end_quat is not None)
        if use_orientation:
            start_quat = np.asarray(start_quat, dtype=np.float64)
            start_quat /= np.linalg.norm(start_quat)
            end_quat = np.asarray(end_quat, dtype=np.float64)
            end_quat /= np.linalg.norm(end_quat)

        results = []
        q_prev = q_init

        for i in range(n_waypoints):
            t = i / max(n_waypoints - 1, 1)

            # Linear position interpolation
            pos = (1.0 - t) * start_pos + t * end_pos

            # SLERP for orientation
            quat = None
            if use_orientation:
                quat = self._slerp(start_quat, end_quat, t)

            result = solver(target_pos=pos, target_quat=quat, q_init=q_prev)
            results.append(result)

            # Seed next waypoint from this solution
            q_prev = result.q_arm

        return results

    @staticmethod
    def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Spherical Linear Interpolation between two quaternions."""
        dot = np.dot(q0, q1)

        # Shortest path
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        # Clamp for numerical stability
        dot = min(dot, 1.0)

        if dot > 0.9995:
            # Near-linear interpolation
            result = (1.0 - t) * q0 + t * q1
        else:
            theta = np.arccos(dot)
            sin_theta = np.sin(theta)
            result = (np.sin((1.0 - t) * theta) / sin_theta * q0
                      + np.sin(t * theta) / sin_theta * q1)

        return result / (np.linalg.norm(result) + 1e-16)

    # -----------------------------------------------------------------
    # Utility: workspace reachability check
    # -----------------------------------------------------------------
    def is_reachable(self, side: str,
                     target_pos: np.ndarray,
                     target_quat: np.ndarray | None = None) -> bool:
        """Quick check whether a target is reachable by the specified arm."""
        solver = self.solve_left if side == "left" else self.solve_right
        result = solver(target_pos=target_pos, target_quat=target_quat)
        return result.converged

    # -----------------------------------------------------------------
    # Utility: print kinematic info
    # -----------------------------------------------------------------
    def print_arm_info(self):
        """Print kinematic chain details for debugging."""
        print("=" * 65)
        print("OpenArm v2.0 Bimanual -- Kinematic Chain Info")
        print("=" * 65)

        for label, jnt_ids, lower, upper, home in [
            ("LEFT ARM",  self._left_jnt_ids,
             self._left_jnt_lower, self._left_jnt_upper, LEFT_ARM_HOME),
            ("RIGHT ARM", self._right_jnt_ids,
             self._right_jnt_lower, self._right_jnt_upper, RIGHT_ARM_HOME),
        ]:
            print(f"\n--- {label} ---")
            for i, jid in enumerate(jnt_ids):
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                axis = self.model.jnt_axis[jid]
                print(f"  J{i+1}: {name}")
                print(f"       axis={axis}  "
                      f"range=[{np.degrees(lower[i]):+7.1f} deg, "
                      f"{np.degrees(upper[i]):+7.1f} deg]  "
                      f"home={np.degrees(home[i]):+6.1f} deg")

        # FK at home position
        fk = self.forward_kinematics(q_left=LEFT_ARM_HOME, q_right=RIGHT_ARM_HOME)
        print("\n--- Home Position FK ---")
        for side, data in fk.items():
            print(f"  {side} TCP pos:  [{data['pos'][0]:+.4f}, "
                  f"{data['pos'][1]:+.4f}, {data['pos'][2]:+.4f}]")
            print(f"  {side} TCP quat: [{data['quat'][0]:+.4f}, "
                  f"{data['quat'][1]:+.4f}, {data['quat'][2]:+.4f}, "
                  f"{data['quat'][3]:+.4f}]")
        print("=" * 65)


# =====================================================================
# Interactive Demo & Self-Test
# =====================================================================
if __name__ == "__main__":
    import time

    MODEL_PATH = "openarm_v20_mujoco/openarm_v20.xml"

    print("Loading OpenArm v2.0 model...")
    ik = OpenArmIK(MODEL_PATH)
    ik.print_arm_info()

    # ---- Test 1: Position-only IK for left arm ----
    print("\n\n" + "=" * 65)
    print("TEST 1: Left Arm Position-Only IK")
    print("=" * 65)
    target = [0.20, 0.20, 0.45]
    t0 = time.perf_counter()
    result = ik.solve_left(target_pos=target)
    dt = time.perf_counter() - t0
    print(f"  Target:      {target}")
    print(f"  Achieved:    [{result.ee_pos[0]:.5f}, "
          f"{result.ee_pos[1]:.5f}, {result.ee_pos[2]:.5f}]")
    print(f"  Converged:   {result.converged}")
    print(f"  Pos error:   {result.pos_error * 1000:.3f} mm")
    print(f"  Iterations:  {result.iterations}")
    print(f"  Solve time:  {dt * 1000:.2f} ms")
    print(f"  Joint angles (deg): {np.degrees(result.q_arm)}")

    # ---- Test 2: 6-DoF IK for right arm ----
    print("\n\n" + "=" * 65)
    print("TEST 2: Right Arm 6-DoF IK")
    print("=" * 65)
    target_pos = [0.25, -0.20, 0.40]
    # Use the home FK orientation as a realistic target
    fk_home = ik.forward_kinematics(q_right=RIGHT_ARM_HOME)
    target_quat = fk_home["right"]["quat"].tolist()
    t0 = time.perf_counter()
    result = ik.solve_right(target_pos=target_pos, target_quat=target_quat)
    dt = time.perf_counter() - t0
    print(f"  Target pos:  {target_pos}")
    print(f"  Target quat: {target_quat}")
    print(f"  Achieved pos:  [{result.ee_pos[0]:.5f}, "
          f"{result.ee_pos[1]:.5f}, {result.ee_pos[2]:.5f}]")
    print(f"  Achieved quat: [{result.ee_quat[0]:.5f}, "
          f"{result.ee_quat[1]:.5f}, {result.ee_quat[2]:.5f}, "
          f"{result.ee_quat[3]:.5f}]")
    print(f"  Converged:   {result.converged}")
    print(f"  Pos error:   {result.pos_error * 1000:.3f} mm")
    print(f"  Rot error:   {np.degrees(result.rot_error):.3f} deg")
    print(f"  Iterations:  {result.iterations}")
    print(f"  Solve time:  {dt * 1000:.2f} ms")
    print(f"  Joint angles (deg): {np.degrees(result.q_arm)}")

    # ---- Test 3: Bimanual coordinated IK ----
    print("\n\n" + "=" * 65)
    print("TEST 3: Bimanual Coordinated IK")
    print("=" * 65)
    t0 = time.perf_counter()
    bi_result = ik.solve_bimanual(
        left_pos=[0.20, 0.15, 0.35],
        right_pos=[0.20, -0.15, 0.35],
    )
    dt = time.perf_counter() - t0
    print(f"  Both converged: {bi_result.converged}")
    print(f"  Left  pos error: {bi_result.left.pos_error * 1000:.3f} mm  "
          f"({bi_result.left.iterations} iters)")
    print(f"  Right pos error: {bi_result.right.pos_error * 1000:.3f} mm  "
          f"({bi_result.right.iterations} iters)")
    print(f"  Total solve time: {dt * 1000:.2f} ms")

    # ---- Test 4: Cartesian trajectory ----
    print("\n\n" + "=" * 65)
    print("TEST 4: Left Arm Cartesian Trajectory (10 waypoints)")
    print("=" * 65)
    t0 = time.perf_counter()
    trajectory = ik.cartesian_trajectory(
        side="left",
        start_pos=[0.15, 0.15, 0.45],
        end_pos=[0.25, 0.25, 0.30],
        n_waypoints=10,
    )
    dt = time.perf_counter() - t0
    all_converged = all(r.converged for r in trajectory)
    max_err = max(r.pos_error for r in trajectory) * 1000
    print(f"  All converged: {all_converged}")
    print(f"  Max pos error: {max_err:.3f} mm")
    print(f"  Total time:    {dt * 1000:.2f} ms")
    for i, r in enumerate(trajectory):
        print(f"    WP{i:02d}: pos=[{r.ee_pos[0]:+.4f}, {r.ee_pos[1]:+.4f}, "
              f"{r.ee_pos[2]:+.4f}]  err={r.pos_error*1000:.2f}mm  "
              f"{'OK' if r.converged else 'FAIL'}")

    # ---- Test 5: Reachability check ----
    print("\n\n" + "=" * 65)
    print("TEST 5: Reachability Checks")
    print("=" * 65)
    test_points = [
        ("Close reach", [0.15, 0.15, 0.40]),
        ("Far reach",   [0.50, 0.10, 0.30]),
        ("Behind",      [-0.30, 0.10, 0.40]),
        ("Very low",    [0.15, 0.15, 0.05]),
    ]
    for label, pt in test_points:
        reachable = ik.is_reachable("left", pt)
        print(f"  {label:15s} {pt} -> {'REACHABLE' if reachable else 'NOT REACHABLE'}")

    print("\n\nAll tests complete!")
