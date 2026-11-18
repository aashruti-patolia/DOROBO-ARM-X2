"""
PickPlaceEnv - hybrid scripted + learned pick-and-place (Gymnasium API).

One episode, as the robot experiences it::

    LOCATE      (scripted)  head camera finds ball + basket; the arm on the
                            ball's side is chosen, the other stays parked
    TRANSIT     (scripted)  free-space move: up to a safe height, over, and
                            down to hover above the ball (tracks a moving ball)
    GRASP       (PPO)       final approach, alignment, closing, lifting
    TRANSPORT   (scripted)  carry the ball at a safe height to hover above
                            the basket (a dropped ball -> back to TRANSIT)
    PLACE       (PPO)       fine positioning over the basket, steady, release
    RETREAT     (scripted)  lift the open hand clear, wait for the ball to settle
    outcome                 success / miss / failure

Scripted segments run *inside* ``step()``: PPO only sees the steps where it
decides, so one ``step()`` can span many control ticks. The scripted parts
use robot-available information only: camera estimates, joint encoders,
the finger angle and the gripper motor force. The supervisor also knows the
arm's workspace (the reachability maps).

Policy interface
----------------
* Action (4, in [-1, 1]): pinch-point displacement dx, dy, dz (up to
  ``policy_max_step`` per step) and the gripper (< 0 close, > 0 open). It
  is expressed in a canonical "left arm" frame: for the right arm Y is
  mirrored, so one policy serves both arms. Moves are clamped to a per-skill
  safety box.
* Observation (45): see ``OBS_LAYOUT``. Proprioception plus camera-based
  estimates only; no simulator ground truth.

Control runs at 25 Hz (20 x 2 ms physics substeps). The camera runs every
second tick.
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ..config import Config
from ..perception.camera import Perception
from .controller import (FINGERS_DOWN_SOFT, MIRROR_JOINT_SIGNS, CartesianController, JointMove, SafetyBox,
                         yaw_quat)
from .reward import GRASP, PLACE, SKILL_NAMES, RewardFunction, TaskSnapshot
from .scene import BALL_RADIUS, BOX_HALF_SIZE, Scene
from .teacher import Teacher

# Observation layout (name, size) - all vectors in the canonical (mirrored) frame
OBS_LAYOUT = [
    ("skill", 2),            # one-hot: GRASP, PLACE
    ("joint_pos", 7),
    ("joint_vel", 7),
    ("finger_open", 1),      # measured finger opening, 0 closed ... 1 open
    ("gripper_force", 1),    # gripper motor force (how a real gripper "feels" a grasp)
    ("pinch_in_box", 3),     # pinch point relative to the skill's safety box, in half-widths
    ("pinch_quat", 4),
    ("ball_rel", 3),         # camera: ball - pinch
    ("ball_vel", 3),         # camera: ball velocity estimate
    ("ball_visible", 1),
    ("ball_age", 1),         # time since the ball was last seen (0 ... 1)
    ("ball_radius", 1),      # camera estimate, relative to nominal
    ("box_rel", 3),          # camera: basket rim centre - pinch
    ("box_half", 1),         # camera estimate, relative to nominal
    ("box_visible", 1),
    ("height", 1),           # pinch height above the camera-estimated table
    ("skill_time", 1),       # fraction of the skill's step budget used
    ("prev_action", 4),
]
OBS_DIM = sum(n for _, n in OBS_LAYOUT)
OBS_SLICES = {}
_i = 0
for _name, _n in OBS_LAYOUT:
    OBS_SLICES[_name] = slice(_i, _i + _n)
    _i += _n

STAGE_NAMES = ["reach", "grasped", "lifted", "placed_attempt", "success"]


class PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, cfg: Config | None = None, stage: int | None = None,
                 render_mode: str | None = None, seed: int | None = None):
        super().__init__()
        self.cfg = cfg or Config()
        self.render_mode = render_mode
        self.scene = Scene(self.cfg)
        self.model, self.data = self.scene.model, self.scene.data
        c = self.cfg.control
        self.dt_tick = self.model.opt.timestep * c.substeps
        self.perception = Perception(self.model, self.cfg.camera, self.dt_tick)
        self.controller = CartesianController(self.scene.ik, self.scene.actuators,
                                              lead_max=c.target_lead_max, iterations=c.ik_iterations,
                                              orientation_weight=c.ik_orientation_weight,
                                              max_joint_step=c.max_joint_step, max_rot_step=c.max_rot_step,
                                              windup=c.policy_windup)
        self.reward_fn = RewardFunction(self.cfg.reward, c, self.cfg.ppo.gamma)
        self.teacher = Teacher(self.cfg)    # scripted base knowledge (labels for imitation)
        self.stage_idx = self.cfg.curriculum.start_stage if stage is None else int(stage)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._renderer = None
        self._scripted = False          # True while a scripted move runs (lower camera rate)
        self.frame_hook = None          # callable(env) run every control tick (demo recording)
        self._side = "left"
        self._skill = GRASP

    # =====================================================================
    # Curriculum
    # =====================================================================
    def set_curriculum_stage(self, k: int):
        self.stage_idx = int(np.clip(k, 0, len(self.cfg.curriculum.stages) - 1))

    def get_curriculum_stage(self) -> int:
        return self.stage_idx

    @property
    def stage(self):
        return self.cfg.curriculum.stages[self.stage_idx]

    # =====================================================================
    # Gymnasium API
    # =====================================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        if "stage" in options:
            self.set_curriculum_stage(options["stage"])
        stage = self.stage
        start = options.get("start_skill")
        if start is None:
            start = "place" if self._rng.random() < stage.place_start_prob else "grasp"
        hover_start = options.get("hover_start")
        if hover_start is None:
            hover_start = self._rng.random() < self.cfg.curriculum.hover_start_prob

        for attempt in range(10):
            world = self.scene.reset(self._rng, stage, options.get("scenario"))
            sc = self.scene.sc
            belief = self.perception.reset(self.data, self._rng, sc.visual_level, sc.sensor_level)
            if not (belief.ball_ever_seen and belief.box_ever_seen):
                continue                      # objects must start in the camera's view
            # The camera picks the arm: the one on the ball's side.
            self._side = "left" if belief.ball_pos[1] > 0 else "right"
            self._begin_episode(world)
            if start == "place" and self._start_with_ball_in_hand():
                break
            if start == "place":
                continue                      # could not set up the held ball: new world
            if (hover_start and self._start_at_hover()) or self._transit_to_ball():
                self._begin_skill(GRASP)
                break
        else:
            raise RuntimeError("Could not generate a valid episode in 10 attempts")

        self._episode["start_skill"] = SKILL_NAMES[self._skill]
        self._episode["hover_start"] = bool(hover_start and self._skill == GRASP)
        self._phi = self.reward_fn.potential(self._snapshot())
        obs = self._obs()
        info = self._info()
        info["scenario"] = self._world["scenario"]
        info["teacher_action"] = self.teacher.act(self)
        return obs, info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        c = self.cfg.control
        self._steps += 1
        self._skill_steps += 1
        self._queue.append(action)
        executed = self._queue.popleft()          # command latency (0 unless randomised)

        # ---- the policy's move, inside the safety box ----
        delta = executed[:3] * c.policy_max_step
        if self._side == "right":
            delta[1] = -delta[1]                  # canonical frame -> world
        # binary gripper, like a real parallel gripper: close fully (squeezing) or open fully
        grip = 1.0 if float(executed[3]) > 0.0 else 0.0
        if self._skill == GRASP:
            clip = self.controller.command(self.data, delta=delta, box=self._box,
                                           orientation=self._grasp_quat)
        else:
            clip = self.controller.command(self.data, delta=delta, box=self._box,
                                           orientation=FINGERS_DOWN_SOFT, hold_posture=True)
        self._grip_cmd = grip
        self.scene.set_gripper(self._side, grip)
        if self._skill == GRASP and not self._held_sensor():
            self.scene.maybe_push_ball(self._rng)
        self._tick()

        box_contact = self.scene.arm_box_contact(self._side)
        distractor_contact = self.scene.arm_distractor_contact(self._side)
        self._episode["box_contacts"] += int(box_contact)
        penalties = self.reward_fn.penalties(box_contact=box_contact, distractor_contact=distractor_contact,
                                             safety_clip=clip, action=action, prev_action=self._prev_action)
        self._prev_action = action.copy()
        self._update_progress()

        # ---- supervisor: skill completion, scripted segments, outcomes ----
        outcome = self._check_failure()
        if outcome is None:
            if self._skill == GRASP:
                outcome = self._after_grasp_step()
            else:
                outcome = self._after_place_step()

        if outcome is None:
            # A skill that runs out of its step budget has failed its attempt. The budget
            # is part of the observation (skill_time), so the timeout is predictable and is
            # a real terminal state: no bootstrapping, a penalty like a miss. (Bootstrapping
            # it let the policy "wait out the clock" instead of risking a release.)
            limit = c.grasp_max_steps if self._skill == GRASP else c.place_max_steps
            if self._skill_steps >= limit or self._steps >= c.episode_max_steps:
                outcome = "timeout"
                self._episode["end_reason"] = f"timeout_{SKILL_NAMES[self._skill]}"
        terminated = outcome is not None
        truncated = False

        phi_next = 0.0 if terminated else self.reward_fn.potential(self._snapshot())
        reward, rinfo = self.reward_fn(self._phi, phi_next, terminated, outcome, penalties)
        self._phi = phi_next
        self._episode["return"] += reward
        if outcome is not None:
            self._episode["end"] = self._episode.get("end_reason") or outcome

        obs = self._obs()
        info = self._info()
        info.update({k: rinfo[k] for k in ("r_event", "r_shaping", "r_penalty")})
        info["is_success"] = outcome == "success"
        info["teacher_action"] = self.teacher.act(self)
        if terminated or truncated:
            info["episode"] = self._episode_summary(outcome == "success")
        return obs, float(reward), terminated, truncated, info

    # =====================================================================
    # Episode bookkeeping
    # =====================================================================
    def _begin_episode(self, world):
        sc = self.scene.sc
        self._world = world
        self.controller.reset(self._side, self.data)
        self._queue = deque(np.zeros(4, dtype=np.float32) for _ in range(world["latency"]))
        self._steps = 0
        self._skill_steps = 0
        self._retries = 0
        self._prev_action = np.array([0, 0, 0, 1.0], dtype=np.float32)
        self._grip_cmd = 1.0
        self._grasp_quat = None
        self._box = None
        self._episode = {
            "return": 0.0, "max_stage": 0, "box_contacts": 0, "retries": 0,
            "ball_err_sum": 0.0, "ball_err_n": 0, "end": "", "end_reason": None,
            "side": self._side, "table_z": sc.table_z, "ball_radius": sc.ball_radius,
            "box_half": sc.box_half, "box_depth": sc.box_depth, "distance": world["distance"],
            "rolling": bool(sc.rolling), "stage": self.stage_idx,
        }

    def _begin_skill(self, skill: int):
        c, b = self.cfg.control, self.perception.state
        self._skill = skill
        self._skill_steps = 0
        pinch = self.scene.pinch_pos(self._side)
        self.controller.target = pinch.copy()
        if skill == GRASP:
            anchor = pinch.copy()
            z_lo = b.table_z + c.fingertip_below_pinch + c.table_margin
            self._box = SafetyBox(
                lo=np.array([anchor[0] - c.grasp_box_half_xy, anchor[1] - c.grasp_box_half_xy, z_lo]),
                hi=np.array([anchor[0] + c.grasp_box_half_xy, anchor[1] + c.grasp_box_half_xy,
                             anchor[2] + c.grasp_box_up]))
            self._prev_action[3] = 1.0
        else:
            rim = b.box_pos
            self._box = SafetyBox(
                lo=np.array([rim[0] - c.place_box_half_xy, rim[1] - c.place_box_half_xy,
                             rim[2] + c.place_box_min_above_rim]),
                hi=np.array([rim[0] + c.place_box_half_xy, rim[1] + c.place_box_half_xy,
                             rim[2] + c.place_box_max_above_rim]))
            self._prev_action[3] = -1.0
            self._grip_cmd = 0.0
        self._episode["max_stage"] = max(self._episode["max_stage"], 2 if skill == PLACE else 0)

    def _update_progress(self):
        snap = self._snapshot()
        stage = 1 if (snap.skill == GRASP and snap.held) else 0
        self._episode["max_stage"] = max(self._episode["max_stage"], stage)
        b = self.perception.state
        if b.ball_visible:
            self._episode["ball_err_sum"] += float(np.linalg.norm(b.ball_pos - self.scene.ball_pos()))
            self._episode["ball_err_n"] += 1

    def _episode_summary(self, success: bool) -> dict:
        e = dict(self._episode)
        if success:
            e["max_stage"] = 4
        e["success"] = bool(success)
        e["length"] = self._steps
        e["ball_err"] = e["ball_err_sum"] / max(e["ball_err_n"], 1)
        for k in ("ball_err_sum", "ball_err_n", "end_reason"):
            e.pop(k, None)
        return e

    # =====================================================================
    # Supervisor logic (scripted)
    # =====================================================================
    def _held_sensor(self) -> bool:
        """Robot-side grasp detection: fingers commanded closed but stalled open
        (something between them) and the motor squeezing."""
        c, s = self.cfg.control, self.scene
        return (self._grip_cmd < c.grasp_close_cmd
                and s.finger_angle(self._side) > c.held_min_angle
                and s.finger_speed(self._side) < c.held_max_finger_speed
                and s.gripper_force(self._side) > c.held_min_force)

    def _check_failure(self):
        if self.scene.box_tilt_deg() > self.cfg.reward.box_tip_deg:
            self._episode["end_reason"] = "box_tipped"
            return "failure"
        if self.scene.ball_off_table():
            self._episode["end_reason"] = "ball_off_table"
            return "failure"
        return None

    def _after_grasp_step(self):
        c, b = self.cfg.control, self.perception.state
        held = self._held_sensor()
        if held:
            pinch = self.scene.pinch_pos(self._side)
            lifted = pinch[2] - b.table_z >= b.ball_radius + c.lift_success
            near = (not b.ball_visible) or np.linalg.norm(b.ball_pos - pinch) < c.ball_near_hand
            if lifted and near:
                self._episode["max_stage"] = max(self._episode["max_stage"], 2)
                return self._transport()
            return None
        # A ball that rolled/was knocked out of the grasp box: re-approach (scripted).
        if b.ball_ever_seen:
            out = np.abs(b.ball_pos[:2] - self._box.centre[:2]) > self._box.half[:2] + 0.01
            if out.any() and self._grip_cmd > 0.5:
                return self._retry_grasp()
        return None

    def _retry_grasp(self):
        self._retries += 1
        self._episode["retries"] = self._retries
        if self._retries > 3:
            self._episode["end_reason"] = "too_many_retries"
            return "miss"
        if self._transit_to_ball():
            self._begin_skill(GRASP)
            return None
        self._episode["end_reason"] = "ball_unreachable"
        return "miss"

    def _transport(self):
        """Scripted: lift, carry the held ball at a safe height over the basket
        (joint-space path through IK waypoints), then start PLACE."""
        c, b = self.cfg.control, self.perception.state
        held = lambda: not self._held_sensor()                        # noqa: E731  (abort condition)
        pinch = self.scene.pinch_pos(self._side)
        safe_z = max(b.table_z + c.safe_height, b.box_pos[2] + c.preplace_hover, pinch[2] + 0.04)
        ok = self._joint_goto([pinch + [0.0, 0.0, 0.04]], (self._grasp_quat if self._grasp_quat is not None else FINGERS_DOWN_SOFT),
                              grip=0.0, speed=c.carry_joint_speed, abort=held)
        if ok:
            start = self.scene.pinch_pos(self._side)
            above = np.array([b.box_pos[0], b.box_pos[1], safe_z])
            line = [start + (above - start) * k / c.carry_waypoints for k in range(1, c.carry_waypoints + 1)]
            ok = self._joint_goto(line, FINGERS_DOWN_SOFT, grip=0.0, speed=c.carry_joint_speed, abort=held)
        if ok:
            hover = np.array([b.box_pos[0], b.box_pos[1], b.box_pos[2] + c.preplace_hover])
            ok = self._joint_goto([hover], FINGERS_DOWN_SOFT, grip=0.0, speed=c.carry_joint_speed, abort=held)
        if not self._held_sensor():
            # dropped on the way: go and pick it up again, if it is still reachable
            for _ in range(int(0.6 / self.dt_tick)):
                self._tick()
            if self._check_failure() is not None:
                return "failure"
            return self._retry_grasp()
        if not ok:
            self._episode["end_reason"] = "transport_failed"
            return "miss"
        self._begin_skill(PLACE)
        return None

    def _after_place_step(self):
        c = self.cfg.control
        if self._grip_cmd > c.release_open_cmd or not self._held_sensor():
            self._episode["max_stage"] = max(self._episode["max_stage"], 3)
            return self._release_and_settle()
        return None

    def _release_and_settle(self):
        """Scripted: open (if not already), lift clear, wait for the ball to settle."""
        c, r = self.cfg.control, self.cfg.reward
        self._grip_cmd = 1.0
        self.scene.set_gripper(self._side, 1.0)
        hold = self.data.qpos[self.scene.qpos_adr[self._side]].copy()
        self.data.ctrl[self.scene.actuators[self._side]] = hold
        for _ in range(4):                                  # let the fingers open in place
            self._tick()
        up = self.scene.pinch_pos(self._side) + [0.0, 0.0, c.retreat_height]
        self._joint_goto([up], FINGERS_DOWN_SOFT, grip=1.0, speed=c.scripted_joint_speed)
        settled = 0
        for _ in range(c.settle_max_ticks):
            if self.scene.box_tilt_deg() > r.box_tip_deg:
                self._episode["end_reason"] = "box_tipped"
                return "failure"
            if self.scene.ball_in_box(r.box_upright_deg) and np.linalg.norm(self.scene.ball_vel()) < r.settled_speed:
                settled += 1
                if settled >= r.settled_ticks:
                    return "success"
            else:
                settled = 0
                if (not self.scene.ball_in_box(r.box_upright_deg)
                        and np.linalg.norm(self.scene.ball_vel()) < 0.02
                        and self.scene.ball_pos()[2] < self.scene.sc.table_z + self.scene.sc.ball_radius + 0.01):
                    break                                  # resting on the table, outside
            self._tick()                                   # the arm holds its last setpoint
        if self.scene.ball_off_table():
            self._episode["end_reason"] = "ball_off_table"
            return "failure"
        self._episode["end_reason"] = "landed_outside"
        return "miss"

    def _transit_to_ball(self) -> bool:
        """Scripted: from wherever the hand is, over the (tracked) ball at a safe
        height and down to the pre-grasp hover, as one smooth joint-space move.

        The hover pose is solved first (it matters most) and the waypoint above
        it is solved from there, so both lie on the same arm configuration and
        the final descent is short and straight. A ball that moves meanwhile is
        followed by re-planning.
        """
        c, b, s, side = self.cfg.control, self.perception.state, self.scene, self._side
        yaw = s.grasp_yaw_at(side, b.ball_pos)
        if yaw is None:
            return False
        self._grasp_quat = yaw_quat(yaw)
        safe_z = b.table_z + c.safe_height
        for _ in range(4):
            hover = b.ball_pos + np.array([0.0, 0.0, c.pregrasp_hover])
            above = np.array([hover[0], hover[1], max(safe_z, hover[2])])
            q_now = self.data.qpos[s.qpos_adr[side]].copy()
            s.ik.sync_from(self.data)
            sol_h = s.ik.solve_closest(side, hover, self._grasp_quat, q_now)
            if sol_h is None:
                return False
            path = [q_now]
            pinch = s.pinch_pos(side)
            if np.linalg.norm(pinch[:2] - hover[:2]) > 0.03 or pinch[2] < safe_z - 0.02:
                # the waypoint above must stay on the hover's arm configuration
                qh = sol_h.q_arm
                sol_a = s.ik.solve(side, above, self._grasp_quat, q_init=qh, posture=qh)
                if not (sol_a.pos_error < 0.006 and sol_a.rot_error < 0.12):
                    sol_a = s.ik.solve(side, above, FINGERS_DOWN_SOFT, q_init=qh, posture=qh)
                if sol_a.pos_error < 0.006 and np.abs(sol_a.q_arm - qh).sum() < 2.0:
                    path.append(sol_a.q_arm)
            path.append(sol_h.q_arm)
            self._grip_cmd = 1.0
            s.set_gripper(side, 1.0)
            if not self._run_joint_path(path, hover, c.scripted_joint_speed):
                return False
            moved = np.linalg.norm(b.ball_pos + np.array([0.0, 0.0, c.pregrasp_hover]) - hover)
            if moved < c.replan_distance:
                break
        self.controller.reset(side, self.data)
        yaw = s.grasp_yaw_at(side, b.ball_pos)                      # the ball may have rolled
        if yaw is None:
            return False
        self._grasp_quat = yaw_quat(yaw)
        return True

    def _run_joint_path(self, path, goal_pos, speed, abort=None) -> bool:
        """Follow joint waypoints with one cosine profile, then settle on the goal."""
        c, s, side = self.cfg.control, self.scene, self._side
        adr, acts = s.qpos_adr[side], s.actuators[side]
        move = JointMove(path, self.dt_tick, speed)
        self._scripted = True
        try:
            return self._follow(move, goal_pos, adr, acts, abort)
        finally:
            self._scripted = False

    def _follow(self, move, goal_pos, adr, acts, abort) -> bool:
        c, s, side = self.cfg.control, self.scene, self._side
        while not move.done:
            self.data.ctrl[acts] = move.next_setpoint()
            self._tick()
            if abort is not None and abort():
                return False
            if self._check_failure() is not None:
                return False
        for _ in range(15):                             # servo lag: settle on the goal
            if (np.abs(self.data.qpos[adr] - move.q_goal).max() < 0.03
                    or np.linalg.norm(s.pinch_pos(side) - goal_pos) < c.scripted_tolerance):
                break
            self._tick()
            if abort is not None and abort():
                return False
        return True

    def _joint_goto(self, targets, orientation, *, grip, speed, abort=None) -> bool:
        """Scripted joint-space move through Cartesian waypoints (list of pinch-point
        positions): each solved by IK closest to the previous one, then followed
        with one smooth cosine profile."""
        s, side = self.scene, self._side
        self._grip_cmd = grip
        s.set_gripper(side, grip)
        q_now = self.data.qpos[s.qpos_adr[side]].copy()
        s.ik.sync_from(self.data)
        path, q_prev = [q_now], q_now
        for g in targets:
            sol = s.ik.solve_closest(side, g, orientation, q_prev)
            if sol is None:
                return False
            path.append(sol.q_arm)
            q_prev = sol.q_arm
        ok = self._run_joint_path(path, targets[-1], speed, abort)
        self.controller.reset(side, self.data)
        return ok

    def _start_at_hover(self) -> bool:
        """Training shortcut: put the hand straight at the pre-grasp hover (where the
        scripted transit would have taken it). Initial state only."""
        c, b, s = self.cfg.control, self.perception.state, self.scene
        yaw = s.grasp_yaw_at(self._side, b.ball_pos)
        if yaw is None:
            return False
        quat = yaw_quat(yaw)
        target = b.ball_pos + np.array([0.0, 0.0, c.pregrasp_hover])
        res = s.ik.solve(self._side, target, quat)
        if res.pos_error > 0.003 or res.rot_error > 0.05:
            return False
        d = self.data
        d.qpos[s.qpos_adr[self._side]] = res.q_arm
        d.ctrl[s.actuators[self._side]] = res.q_arm
        d.qvel[s.dof_adr[self._side]] = 0.0
        mujoco.mj_forward(self.model, d)
        if s.arm_box_contact(self._side) or s.arm_distractor_contact(self._side) or any(s.finger_ball_contacts(self._side)):
            return False
        self._grasp_quat = quat
        self.controller.reset(self._side, d)
        for _ in range(3):
            self.controller.command(d, target=target, orientation=quat)
            self._tick()
        return True

    def _start_with_ball_in_hand(self) -> bool:
        """PLACE-start episode: the hand already holds the ball above the basket.
        An initial-state choice (lets PLACE be practised early); no actions are shown."""
        c, b = self.cfg.control, self.perception.state
        target = np.array([b.box_pos[0], b.box_pos[1], b.box_pos[2] + c.preplace_hover])
        if not self.scene.place_ball_in_hand(self._side, target, self._rng):
            return False
        self.perception.tracker.state.ball_visible = False   # the ball jumped: no velocity spike
        self.controller.reset(self._side, self.data)
        self._grip_cmd = 0.0
        self.scene.set_gripper(self._side, 0.0)
        for _ in range(3):
            self.controller.command(self.data, target=self.controller.target, orientation=FINGERS_DOWN_SOFT,
                                    hold_posture=True)
            self._tick()
        if not self._held_sensor():
            return False
        self._begin_skill(PLACE)
        return True

    # =====================================================================
    # Simulation tick
    # =====================================================================
    def _tick(self):
        """One 40 ms control tick: physics substeps, idle arm parked, camera."""
        sc = self.scene.sc
        other = "right" if self._side == "left" else "left"
        self.data.ctrl[self.scene.actuators[other]] = self.scene.q_start[other]
        self.scene.step_physics(self.cfg.control.substeps)
        cam = self.cfg.camera
        self.perception.tick(self.data, self._rng, sc.visual_level, sc.sensor_level,
                             every=cam.scripted_every_n_ticks if self._scripted else cam.every_n_ticks)
        if self.frame_hook is not None:
            self.frame_hook(self)

    # =====================================================================
    # Observation / info
    # =====================================================================
    def _snapshot(self) -> TaskSnapshot:
        s = self.scene
        return TaskSnapshot(skill=self._skill, pinch=s.pinch_pos(self._side), ball=s.ball_pos(),
                            box=s.box_pos(), table_z=s.sc.table_z, ball_radius=s.sc.ball_radius,
                            held=all(s.finger_ball_contacts(self._side)))

    def _obs(self) -> np.ndarray:
        s, c, b, sc = self.scene, self.cfg.control, self.perception.state, self.scene.sc
        side = self._side
        mir = np.array([1.0, -1.0, 1.0]) if side == "right" else np.ones(3)
        q = self.data.qpos[s.qpos_adr[side]].copy()
        qd = self.data.qvel[s.dof_adr[side]].copy()
        if side == "right":
            q, qd = MIRROR_JOINT_SIGNS * q, MIRROR_JOINT_SIGNS * qd
        if sc.encoder_pos_noise > 0:
            q = q + self._rng.normal(0, sc.encoder_pos_noise, 7)
            qd = qd + self._rng.normal(0, sc.encoder_vel_noise, 7)
        pinch = s.pinch_pos(side)
        quat = s.pinch_quat(side)
        if side == "right":
            quat = quat * np.array([1.0, -1.0, 1.0, -1.0])
        skill = np.zeros(2)
        skill[self._skill] = 1.0
        box = self._box
        pinch_in_box = (pinch - box.centre) / np.maximum(box.half, 1e-3)
        finger = (s.finger_angle(side) - c.gripper_closed) / (c.gripper_open - c.gripper_closed)
        limit = c.grasp_max_steps if self._skill == GRASP else c.place_max_steps
        parts = [
            skill, q, qd, [finger], [s.gripper_force(side) / 7.0], mir * pinch_in_box, quat,
            mir * (b.ball_pos - pinch), mir * b.ball_vel, [float(b.ball_visible)],
            [min(b.ball_age, self.cfg.camera.max_age_ticks) / self.cfg.camera.max_age_ticks],
            [b.ball_radius / BALL_RADIUS - 1.0],
            mir * (b.box_pos - pinch), [b.box_half / BOX_HALF_SIZE - 1.0], [float(b.box_visible)],
            [pinch[2] - b.table_z], [self._skill_steps / limit], self._prev_action,
        ]
        return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts]).astype(np.float32)

    def _info(self) -> dict:
        return {"skill": SKILL_NAMES[self._skill], "side": self._side, "stage": self.stage_idx,
                "steps": self._steps}

    # =====================================================================
    # Rendering
    # =====================================================================
    def render(self):
        """RGB frame: a free side view plus the head camera's view (with detections)."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=360, width=480)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.3, 0.0, 0.3]
        cam.distance, cam.azimuth, cam.elevation = 1.3, 200.0, -30.0
        self._renderer.update_scene(self.data, camera=cam)
        side_view = self._renderer.render().copy()
        rgb = self.perception.camera.render_rgb(self.data)
        h, w = side_view.shape[:2]
        ys = (np.arange(h) * rgb.shape[0] / h).astype(int)
        xs = (np.arange(w // 2) * rgb.shape[1] / (w // 2)).astype(int)
        head = rgb[ys][:, xs]
        return np.concatenate([side_view, head], axis=1)

    def close(self):
        self.perception.close()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
