"""
Low-level control for the OpenArm v2.0 arms (scripted, never learned).

* ``OpenArmIK`` - damped least-squares (Levenberg-Marquardt) IK on a private
  MjData, with a nullspace posture bias for the redundant 7th DoF. Besides
  position-only and full-pose targets it supports a "fingers down, yaw free"
  orientation mode: only the two tilt components of the hand orientation are
  constrained, the rotation about the vertical is left to the solver.
* ``CartesianController`` - turns a Cartesian target (or a small delta, as the
  policy commands) into joint-servo targets. It clamps the target to a safety
  box and never lets it run ahead of the real hand (anti-windup).

Nothing here decides *where* to go: the scripted supervisor or the policy does.

Reference: S. R. Buss, "Introduction to Inverse Kinematics with Jacobian
Transpose, Pseudoinverse and Damped Least-Squares methods", 2004.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class IKConfig:
    """Solver hyper-parameters."""
    max_iterations: int = 100        # Max Newton-Raphson iterations
    pos_tolerance: float = 1e-4      # Position convergence [m]
    rot_tolerance: float = 1e-3      # Orientation convergence [rad]
    step_size: float = 0.5           # Line-search scaling (0 < a <= 1)
    damping: float = 1e-3            # Levenberg-Marquardt regularisation lambda
    joint_limit_margin: float = 0.05 # Safety margin from joint limits [rad]
    nullspace_weight: float = 0.5    # Weight for nullspace posture bias
    max_linear_step: float = 0.05    # Max Cartesian position step [m]
    max_angular_step: float = 0.25   # Max Cartesian orientation step [rad]
    orientation_weight: float = 1.0  # Weight of orientation rows relative to position
    max_joint_step: float = 0.25     # Max joint change per iteration [rad]


# Kinematic chains (left joints 1-7, right joints 1-7)
LEFT_ARM_JOINT_NAMES = [f"openarm_left_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINT_NAMES = [f"openarm_right_joint{i}" for i in range(1, 8)]

# The environment's end effector is the gripper's pinch point (where a gripped
# ball's centre sits), 13 mm above the fingertips.
LEFT_PINCH_SITE = "left_pinch_site"
RIGHT_PINCH_SITE = "right_pinch_site"

# Ergonomic resting pose; also the nullspace posture target
LEFT_ARM_HOME = np.array([0.0, -0.5, 0.0, 1.2, 0.0, 0.0, 0.0])
RIGHT_ARM_HOME = np.array([0.0, 0.5, 0.0, 1.2, 0.0, 0.0, 0.0])
HOME = {"left": LEFT_ARM_HOME, "right": RIGHT_ARM_HOME}

# Ready pose both arms start every episode in (and the idle arm parks in):
# hands raised beside the head camera with the fingers already pointing down -
# pinch point at (0.20, +/-0.30, 0.50), hand yaw -/+90 deg - so they are out
# of the camera's view, above any table/basket/ball the scenarios produce, and
# a short, natural joint move away from any pre-grasp pose.
LEFT_ARM_READY = np.array([0.2636, -1.6788, 0.0068, 1.9165, -0.1147, -0.3435, -1.2616])

# Mirroring the left arm across the robot's XZ plane gives the right arm with
# these joint-angle signs (derived from the joint axes, checked by a test):
# q_right = MIRROR_JOINT_SIGNS * q_left.
MIRROR_JOINT_SIGNS = np.array([-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
RIGHT_ARM_READY = MIRROR_JOINT_SIGNS * LEFT_ARM_READY
READY = {"left": LEFT_ARM_READY, "right": RIGHT_ARM_READY}

FINGERS_DOWN = "down"            # orientation mode: fingers straight down, yaw free (weighted with position)
FINGERS_DOWN_SOFT = "down_soft"  # position first; fingers as close to down as the remaining DoF allow


@dataclass
class IKResult:
    q_arm: np.ndarray        # joint angles [7]
    converged: bool
    iterations: int
    pos_error: float         # [m]
    rot_error: float         # [rad]
    ee_pos: np.ndarray       # achieved site position
    ee_quat: np.ndarray      # achieved site orientation (w, x, y, z)


def quat_error(quat_target: np.ndarray, quat_current: np.ndarray) -> np.ndarray:
    """Orientation error as a rotation vector, 2 * (q_target * q_current^-1).xyz."""
    qt = quat_target / (np.linalg.norm(quat_target) + 1e-16)
    qc = quat_current / (np.linalg.norm(quat_current) + 1e-16)
    qc_conj = np.array([qc[0], -qc[1], -qc[2], -qc[3]])
    q_err = np.zeros(4)
    mujoco.mju_mulQuat(q_err, qt, qc_conj)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


def _cross(a, b) -> np.ndarray:
    """3-vector cross product (np.cross is slow for single small vectors)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


_UP = np.array([0.0, 0.0, 1.0])


def tilt_from_vertical_deg(site_xmat: np.ndarray) -> float:
    """Angle between the hand's fingers and straight down [deg]."""
    return float(np.degrees(np.arccos(np.clip(site_xmat[8], -1.0, 1.0))))


def yaw_quat(yaw_deg: float) -> np.ndarray:
    """Fingers-down orientation with the given hand yaw (w, x, y, z)."""
    half = np.radians(yaw_deg) / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


class OpenArmIK:
    """Damped least-squares IK for both 7-DoF arms on a private MjData."""

    def __init__(self, model: mujoco.MjModel, config: IKConfig | None = None,
                 left_site: str = LEFT_PINCH_SITE, right_site: str = RIGHT_PINCH_SITE):
        self.model = model
        self.cfg = config or IKConfig()
        # Private data: IK never touches the live simulation state.
        self.data = mujoco.MjData(model)

        self.site_id = {
            "left": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, left_site),
            "right": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, right_site),
        }
        if min(self.site_id.values()) < 0:
            raise ValueError(f"Sites {left_site!r}/{right_site!r} not found in model")

        self.qpos_adr, self.dof_adr, self.lower, self.upper, self.limited = {}, {}, {}, {}, {}
        for side, names in (("left", LEFT_ARM_JOINT_NAMES), ("right", RIGHT_ARM_JOINT_NAMES)):
            ids = []
            for name in names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    raise ValueError(f"Joint {name!r} not found in model")
                ids.append(jid)
            self.qpos_adr[side] = np.array([model.jnt_qposadr[j] for j in ids])
            self.dof_adr[side] = np.array([model.jnt_dofadr[j] for j in ids])
            self.lower[side] = np.array([model.jnt_range[j, 0] for j in ids])
            self.upper[side] = np.array([model.jnt_range[j, 1] for j in ids])
            self.limited[side] = np.array([bool(model.jnt_limited[j]) for j in ids])

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # -- kinematics ---------------------------------------------------------
    def _fk(self):
        """Site poses + what mj_jacSite needs; far cheaper than mj_forward."""
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def sync_from(self, sim_data: mujoco.MjData, forward: bool = True):
        """Copy joint positions from the live simulation."""
        self.data.qpos[:] = sim_data.qpos[:]
        if forward:
            self._fk()

    def site_pose(self, side: str, q_arm: np.ndarray | None = None):
        """(position, rotation matrix) of the arm's site, optionally at ``q_arm``."""
        if q_arm is not None:
            self.data.qpos[self.qpos_adr[side]] = q_arm
            self._fk()
        sid = self.site_id[side]
        return self.data.site_xpos[sid].copy(), self.data.site_xmat[sid].reshape(3, 3).copy()

    # -- solver ---------------------------------------------------------------
    def solve(self, side: str, target_pos, target_quat=None, q_init=None,
              posture=None, max_iterations: int | None = None,
              damping: float | None = None) -> IKResult:
        """Solve for ``side``'s joints (see ``_solve``). In ``FINGERS_DOWN_SOFT``
        mode position has priority: if the tilt task cost position accuracy,
        the position-only solution is returned instead."""
        res = self._solve(side, target_pos, target_quat, q_init, posture, max_iterations, damping)
        if isinstance(target_quat, str) and target_quat == FINGERS_DOWN_SOFT and res.pos_error > 0.003:
            alt = self._solve(side, target_pos, None, q_init,
                              q_init if posture is None and q_init is not None else posture,
                              max_iterations, damping)
            if alt.pos_error < res.pos_error:
                return alt
        return res

    def _solve(self, side: str, target_pos, target_quat=None, q_init=None,
               posture=None, max_iterations: int | None = None,
               damping: float | None = None) -> IKResult:
        """Solve for ``side``'s joints.

        ``target_quat`` is a (w, x, y, z) quaternion, ``None`` for position
        only, or ``FINGERS_DOWN`` to constrain only the hand's tilt (fingers
        pointing straight down) and leave its yaw free.
        """
        cfg = self.cfg
        if target_quat is None:
            mode = "pos"
        elif isinstance(target_quat, str):
            mode = "soft" if target_quat == FINGERS_DOWN_SOFT else "down"
        else:
            mode = "quat"
        target_pos = np.asarray(target_pos, dtype=np.float64)
        if mode == "quat":
            target_quat = np.asarray(target_quat, dtype=np.float64)
            target_quat = target_quat / (np.linalg.norm(target_quat) + 1e-16)

        qpos_adr, dof_adr = self.qpos_adr[side], self.dof_adr[side]
        home = HOME[side]
        q_lo = np.where(self.limited[side], self.lower[side] + cfg.joint_limit_margin, -np.inf)
        q_hi = np.where(self.limited[side], self.upper[side] - cfg.joint_limit_margin, np.inf)
        posture_target = home if posture is None else np.asarray(posture, dtype=np.float64)

        sid = self.site_id[side]
        qpos = self.data.qpos
        site_pos = self.data.site_xpos[sid]
        site_mat = self.data.site_xmat[sid]

        q = np.clip(np.asarray(q_init, dtype=np.float64) if q_init is not None else home, q_lo, q_hi)
        qpos[qpos_adr] = q
        self.data.qvel[:] = 0.0
        self._fk()

        lam2 = (cfg.damping if damping is None else damping) ** 2
        w_null = cfg.nullspace_weight
        dq_cap = cfg.max_joint_step
        w_rot = cfg.orientation_weight
        converged, iterations = False, 0
        pos_err_norm, rot_err_norm = np.inf, 0.0

        for it in range(max_iterations or cfg.max_iterations):
            iterations = it + 1
            pos_err = target_pos - site_pos
            pos_err_norm = float(np.sqrt(pos_err @ pos_err))
            if pos_err_norm > cfg.max_linear_step:
                pos_err_step = pos_err * (cfg.max_linear_step / pos_err_norm)
            else:
                pos_err_step = pos_err

            self._jacp[:] = 0.0
            self._jacr[:] = 0.0
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, sid)
            Jp = self._jacp[:, dof_adr]

            if mode == "soft":
                # Task priority: position exactly, tilt in the position nullspace.
                Jr = self._jacr[:, dof_adr]
                z = site_mat[[2, 5, 8]]
                rot_err = _cross(z, _UP)
                helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                b1 = _cross(z, helper)
                b1 /= np.linalg.norm(b1)
                B = np.stack([b1, _cross(z, b1)])
                Jt, et = B @ Jr, B @ rot_err
                rot_err_norm = float(np.linalg.norm(rot_err))
                if pos_err_norm < cfg.pos_tolerance and rot_err_norm < 0.02:
                    converged = True
                    break
                A = Jp @ Jp.T
                A.flat[::4] += max(lam2, 0.02 ** 2)     # robust near the edge of reach
                Jp_pinv = Jp.T @ np.linalg.inv(A)
                dq1 = Jp_pinv @ pos_err_step
                Np = np.eye(len(q)) - Jp_pinv @ Jp
                Jt_bar = Jt @ Np
                C = Jt_bar @ Jt_bar.T
                C.flat[::3] += 2.5e-3
                rn = np.linalg.norm(et)
                if rn > cfg.max_angular_step:
                    et = et * (cfg.max_angular_step / rn)
                dq = dq1 + Np @ (Jt_bar.T @ np.linalg.solve(C, et - Jt @ dq1))
                dq = np.clip(dq, -dq_cap, dq_cap)
                q = np.clip(q + cfg.step_size * dq, q_lo, q_hi)
                qpos[qpos_adr] = q
                self._fk()
                continue

            if mode == "pos":
                rot_err_norm = 0.0
                if pos_err_norm < cfg.pos_tolerance:
                    converged = True
                    break
                J, dx = Jp, pos_err_step
            else:
                Jr = self._jacr[:, dof_adr]
                if mode == "quat":
                    cur = np.zeros(4)
                    mujoco.mju_mat2Quat(cur, site_mat)
                    rot_err = quat_error(target_quat, cur)
                    J_rot, e_rot = Jr, rot_err
                else:
                    # Tilt only: rotate the hand's z axis onto world +z (the
                    # fingers point along -z of the site). The two rows
                    # perpendicular to the current z axis constrain tilt; the
                    # row along it (yaw) is dropped, so yaw stays free.
                    z = site_mat[[2, 5, 8]]
                    rot_err = _cross(z, _UP)
                    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                    b1 = _cross(z, helper)
                    b1 /= np.linalg.norm(b1)
                    b2 = _cross(z, b1)
                    B = np.stack([b1, b2])
                    J_rot, e_rot = B @ Jr, B @ rot_err
                rot_err_norm = float(np.linalg.norm(rot_err))
                if pos_err_norm < cfg.pos_tolerance and rot_err_norm < cfg.rot_tolerance:
                    converged = True
                    break
                rn = np.linalg.norm(e_rot)
                if rn > cfg.max_angular_step:
                    e_rot = e_rot * (cfg.max_angular_step / rn)
                J = np.vstack([Jp, w_rot * J_rot])
                dx = np.concatenate([pos_err_step, w_rot * e_rot])

            # dq = J^T (J J^T + l^2 I)^-1 (dx - w J p) + w p   (nullspace posture bias)
            JJT = J @ J.T
            JJT.flat[::JJT.shape[0] + 1] += lam2
            posture_err = posture_target - q
            dq = J.T @ np.linalg.solve(JJT, dx - w_null * (J @ posture_err)) + w_null * posture_err
            dq = np.clip(dq, -dq_cap, dq_cap)
            q = np.clip(q + cfg.step_size * dq, q_lo, q_hi)
            qpos[qpos_adr] = q
            self._fk()

        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, site_mat)
        return IKResult(q_arm=q.copy(), converged=converged, iterations=iterations,
                        pos_error=pos_err_norm, rot_error=rot_err_norm,
                        ee_pos=site_pos.copy(), ee_quat=quat)

    def solve_closest(self, side: str, target_pos, target_quat, q_now,
                      pos_tol: float = 0.006, rot_tol: float = 0.12) -> IKResult | None:
        """IK from several starting guesses; the valid solution closest to the
        current joints (no elbow/wrist flips). None if no seed converges."""
        best, best_d = None, np.inf
        q_now = np.asarray(q_now, dtype=np.float64)
        # (seed, posture bias): keeping the current posture suits short moves; the home
        # posture and other seeds find a solution when the target is far away
        attempts = [(q_now, q_now), (q_now, None), (HOME[side], None), (READY[side], None)]
        for k, (seed, posture) in enumerate(attempts):
            r = self.solve(side, target_pos, target_quat, q_init=seed, posture=posture)
            ok = r.pos_error < pos_tol and (target_quat is None or isinstance(target_quat, str)
                                            or r.rot_error < rot_tol)
            if not ok:
                continue
            if k == 0:                     # found from the current joints, current posture: nearest
                return r
            d = float(np.abs(r.q_arm - q_now).sum())
            if d < best_d:
                best, best_d = r, d
        return best

    def solve_left(self, target_pos, target_quat=None, q_init=None, posture=None) -> IKResult:
        return self.solve("left", target_pos, target_quat, q_init, posture)

    def solve_right(self, target_pos, target_quat=None, q_init=None, posture=None) -> IKResult:
        return self.solve("right", target_pos, target_quat, q_init, posture)


class JointMove:
    """Smooth joint-space move along IK waypoints (as old ``pick_and_place.py``
    moved the arm, but through the servos and the physics, never teleported).

    ``q_path`` is the list [q_now, q_1, ..., q_goal]. Progress along the
    piecewise-linear joint path follows a cosine S-curve in time, so the arm
    starts and stops smoothly and never reverses direction. Each control tick
    ``next_setpoint`` gives the servo targets.
    """

    def __init__(self, q_path, dt_tick: float, joint_speed: float, min_time: float = 0.3):
        self.path = [np.asarray(q, dtype=np.float64).copy() for q in q_path]
        if len(self.path) == 1:
            self.path.append(self.path[0].copy())
        seg = [float(np.abs(b - a).max()) for a, b in zip(self.path[:-1], self.path[1:])]
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(self.cum[-1])
        # the peak speed of a cosine profile is (pi/2) * span / T
        duration = max(min_time, 0.5 * np.pi * total / max(joint_speed, 1e-6))
        self.n_ticks = max(1, int(np.ceil(duration / dt_tick)))
        self.k = 0

    @property
    def q_goal(self) -> np.ndarray:
        return self.path[-1]

    @property
    def done(self) -> bool:
        return self.k >= self.n_ticks

    def next_setpoint(self) -> np.ndarray:
        self.k = min(self.k + 1, self.n_ticks)
        s = 0.5 * (1.0 - np.cos(np.pi * self.k / self.n_ticks))
        if self.cum[-1] <= 1e-12:
            return self.path[-1].copy()
        d = s * self.cum[-1]
        i = min(int(np.searchsorted(self.cum, d, side="right")) - 1, len(self.path) - 2)
        span = self.cum[i + 1] - self.cum[i]
        u = 0.0 if span <= 1e-12 else (d - self.cum[i]) / span
        return (1.0 - u) * self.path[i] + u * self.path[i + 1]


@dataclass
class SafetyBox:
    """Axis-aligned box the commanded pinch point must stay inside [m]."""
    lo: np.ndarray
    hi: np.ndarray

    @property
    def centre(self) -> np.ndarray:
        return 0.5 * (self.lo + self.hi)

    @property
    def half(self) -> np.ndarray:
        return 0.5 * (self.hi - self.lo)

    def clamp(self, p: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(p, self.lo), self.hi)


class CartesianController:
    """Cartesian position control of one arm's pinch point, fingers kept down.

    Each control tick: move the target (by a delta or to a point), clamp it to
    the safety box and to within ``lead_max`` of the actual hand, solve IK from
    the current joints and write the result to the joint position servos.
    """

    def __init__(self, ik: OpenArmIK, actuator_ids: dict, lead_max: float = 0.03,
                 iterations: int = 15, orientation_weight: float = 0.6,
                 max_joint_step: float = 0.08, max_rot_step: float = 0.08, damping: float = 0.02,
                 windup: float = 0.06):
        self.ik = ik
        self.actuators = actuator_ids          # side -> array of 7 actuator ids
        self.lead_max = lead_max
        self.max_joint_step = max_joint_step   # safety: joint command change per tick [rad]
        self.max_rot_step = max_rot_step       # safety: orientation goal change per tick [rad]
        self.damping = damping                 # DLS damping for online control (robust near singularities)
        self.windup = windup                   # incremental commands: target never this far from the hand
        self.iterations = iterations
        self.orientation_weight = orientation_weight
        self.side = "left"
        self.target = np.zeros(3)
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_result: IKResult | None = None

    def reset(self, side: str, data: mujoco.MjData):
        self.side = side
        self.target = data.site_xpos[self.ik.site_id[side]].copy()
        self.target_quat = self._current_quat(data)

    def _current_quat(self, data: mujoco.MjData) -> np.ndarray:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, data.site_xmat[self.ik.site_id[self.side]])
        return q

    def orientation_error(self, data: mujoco.MjData, quat) -> float:
        """Angle between the hand and ``quat`` [rad]."""
        return float(np.linalg.norm(quat_error(np.asarray(quat, dtype=np.float64), self._current_quat(data))))

    def pinch(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xpos[self.ik.site_id[self.side]].copy()

    def command(self, data: mujoco.MjData, *, delta=None, target=None,
                box: SafetyBox | None = None, orientation=FINGERS_DOWN,
                hold_posture: bool = False) -> float:
        """Update the target and set the servo commands.

        ``orientation`` is a (w, x, y, z) quaternion, ``FINGERS_DOWN`` (tilt
        only, yaw free) or ``None`` (position only). ``hold_posture`` biases the
        redundant DoF toward the current joints instead of the home pose, which
        keeps the wrist (and a held ball) from swinging while carrying.

        Returns how far (in metres) the safety box pulled the requested target
        back, so the caller can tell when a command pushed against it.
        """
        actual = self.pinch(data)
        if target is not None:
            wanted = np.asarray(target, dtype=np.float64)
        else:
            wanted = self.target + np.asarray(delta, dtype=np.float64)
        intended = box.clamp(wanted) if box is not None else wanted.copy()
        clipped = float(np.linalg.norm(intended - wanted))
        if target is None:
            # anti-windup for incremental commands: the intended target may not
            # run away from a hand that is blocked (e.g. pressing on the table)
            off = intended - actual
            n = np.linalg.norm(off)
            if n > self.windup:
                intended = actual + off * (self.windup / n)
        self.target = intended
        # The IK goal never leads the real hand by more than lead_max, so the
        # servos are never asked for a huge jump; the path target is kept.
        goal = intended
        off = intended - actual
        n = np.linalg.norm(off)
        if n > self.lead_max:
            goal = actual + off * (self.lead_max / n)

        side = self.side
        q_now = data.qpos[self.ik.qpos_adr[side]].copy()
        if orientation is not None and not isinstance(orientation, str):
            orientation = self._advance_orientation(np.asarray(orientation, dtype=np.float64))
        else:
            self.target_quat = self._current_quat(data)
        self.ik.sync_from(data, forward=False)
        old_w = self.ik.cfg.orientation_weight
        self.ik.cfg.orientation_weight = self.orientation_weight
        try:
            res = self.ik.solve(side, goal, orientation, q_init=q_now,
                                posture=q_now if hold_posture else None,
                                max_iterations=self.iterations, damping=self.damping)
        finally:
            self.ik.cfg.orientation_weight = old_w
        self.last_result = res
        # safety: bounded joint command change per tick
        step = np.clip(res.q_arm - q_now, -self.max_joint_step, self.max_joint_step)
        data.ctrl[self.actuators[side]] = q_now + step
        return clipped

    def _advance_orientation(self, quat: np.ndarray) -> np.ndarray:
        """Move the commanded orientation toward ``quat`` by at most max_rot_step
        per tick. Like the position target it is kept independently of the
        (lagging) real hand, so slow servos do not stall the rotation."""
        err = quat_error(quat, self.target_quat)          # world-frame rotation vector
        ang = float(np.linalg.norm(err))
        if ang <= self.max_rot_step:
            self.target_quat = quat / np.linalg.norm(quat)
        else:
            dq, out = np.zeros(4), np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, err / ang, self.max_rot_step)
            mujoco.mju_mulQuat(out, dq, self.target_quat)
            self.target_quat = out / np.linalg.norm(out)
        return self.target_quat
