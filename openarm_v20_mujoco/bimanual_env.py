"""
BimanualPickPlaceEnv -- Gymnasium Environment for PPO Training
================================================================

Residual RL environment for the OpenArm v2.0 bimanual robot.
The IK solver provides base trajectory targets; the PPO policy outputs
small residual corrections (delta_pos, gripper_aperture).

Observation space (41 dims):
  skill_id (3)  +  arm_id (2)  +  joint_pos (7)  +  joint_vel (7)  +
  ee_pos (3)  +  ee_quat (4)  +  gripper_state (1)  +
  ball_ee_vec (3)  +  ball_vel (3)  +  ball_basket_vec (3)  +
  basket_clearance (1)  +  prev_action (7)

Action space (4 continuous in [-1, 1]):
  delta_pos (3)  +  gripper (1)

Actions are expressed in the canonical (left-arm) frame, matching the
mirrored observations. For the right arm the Y component of delta_pos is
negated before it is applied in the world frame.

Phase transitions:
  APPROACH(0) -> GRASP(1) -> DROP(2) -> DONE

After each episode (or successful drop), the active arm alternates and
objects are re-randomized with domain randomization.

Physics and sim-to-real notes:
  - The arm and gripper are driven only through their position servos: IK
    sets the joint targets (``ctrl``) and the dynamics, gains, damping and
    torque limits determine how the arm gets there. Nothing is teleported.
  - The ball is held by friction between the finger pads; there is no
    kinematic attachment or weld. "Grasped" is read off finger contacts, and
    a ball that slips out sends the phase back to APPROACH.
  - The environment's end effector is the gripper's pinch point
    (``*_pinch_site``), where a gripped ball's centre sits.
  - With domain randomization on, each episode draws ball mass/friction,
    finger-pad friction, servo gains, joint damping, 0-2 steps of command
    latency, and a bias plus per-step noise on the perceived ball/box
    positions (see the ``DR_*`` constants).

Scenarios (the world itself changes between episodes):
  - Each reset draws a ``Scenario``: table height (0.17-0.33 m) and shift,
    ball radius, basket width/height, table/basket friction, ball softness,
    joint-encoder noise, observation dropout, random nudges to a resting
    ball, and 0-2 distractor blocks. ``randomization_level`` ("off" / "mild"
    / "full") scales how far each deviates from nominal; see
    ``SCENARIO_RANGES``. Pin any field with
    ``env.reset(options={"scenario": {"table_z": 0.3}})``.
  - Placements come from reachability maps per table height
    (``workspace_maps.npz``, built by ``tools/compute_workspace.py``), so the
    ball and basket are always somewhere the arm can handle. Objects are kept
    out from under the hands' start pose; above a tall table the arms start
    from a raised pose so the open fingers clear it.
  - The gripper height in the observation is relative to the table top.
  - MuJoCo's midphase is disabled because its collision trees are fixed at
    compile time and the table/ball/basket are resized at run time.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, fields
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
    OpenArmIK,
)

# The "end effector" of this environment is the gripper's pinch point (where
# a gripped ball's centre sits), not the TCP site higher up the fingers.
LEFT_PINCH_SITE = "left_pinch_site"
RIGHT_PINCH_SITE = "right_pinch_site"
from reward_calculator import RewardCalculator, RewardConfig


# =====================================================================
# Constants
# =====================================================================
PHASE_APPROACH = 0  # Approach & Pick
PHASE_GRASP = 1     # Precision Grasp & Lift
PHASE_DROP = 2      # Collision-Free Basket Drop
PHASE_NAMES = {0: "APPROACH", 1: "GRASP", 2: "DROP"}

# Nominal world geometry (the XML's values). Each episode's actual values live
# in its ``Scenario`` (self._sc); these are the defaults and the baseline that
# the scenario ranges below scale around.
TABLE_Z = 0.25
BALL_RADIUS = 0.022
BOX_DEPTH = 0.060
BOX_HALF_SIZE = 0.045
BOX_WALL_T = 0.004               # Basket wall half-thickness [m]
BOX_BOTTOM_T = 0.004             # Basket floor half-thickness [m]
BOX_RIM_Z = TABLE_Z + BOX_DEPTH  # 0.310
START_CLEARANCE = 0.03           # The hands start at least this far above the table top [m]
N_DISTRACTORS = 3                # Distractor bodies declared in the XML

# Ball and basket placements are drawn from per-arm reachability maps
# (workspace_maps.npz, produced by tools/compute_workspace.py): every place an
# arm can pick a ball up with a vertical pinch grasp, and every place it can
# drop one into the basket. Without the file a small legacy rectangle is used.
WORKSPACE_MAPS_FILE = "workspace_maps.npz"
LEGACY_WORKSPACE = {
    "left":  {"x": (0.19, 0.29), "y": (0.15, 0.24)},
    "right": {"x": (0.19, 0.29), "y": (-0.24, -0.15)},
}


def yaw_quat(yaw_deg: float) -> np.ndarray:
    """Gripper-down grasp orientation with the given hand yaw (w, x, y, z).

    Fingers point straight down; ``yaw_deg`` rotates the pinch axis about the
    vertical. A ball is a sphere, so any yaw grasps it; the maps store, per
    location, a yaw the arm can actually reach.
    """
    half = np.radians(yaw_deg) / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


DEFAULT_GRASP_YAW = {"left": -90.0, "right": 90.0}


# =====================================================================
# Scenarios: what can differ from episode to episode
# =====================================================================
@dataclass
class Scenario:
    """The world of one episode: geometry, contact properties, sensing, disturbances.

    Defaults are the nominal world. ``reset()`` draws a scenario from
    ``SCENARIO_RANGES`` (scaled by the randomisation level) and applies it to
    the model; ``reset(options={"scenario": {...}})`` overrides fields
    explicitly, which is how the sweep tool and the tests pin a parameter.
    """
    table_z: float = TABLE_Z              # table top height [m]
    table_dx: float = 0.0                 # table shift in X / Y [m] (= a shifted robot base)
    table_dy: float = 0.0
    ball_radius: float = BALL_RADIUS
    box_half: float = BOX_HALF_SIZE       # basket half-width [m]
    box_depth: float = BOX_DEPTH          # basket wall height [m]
    table_friction_scale: float = 1.0
    box_friction_scale: float = 1.0
    ball_solref_scale: float = 1.0        # >1 = softer, bouncier-feeling ball contact
    encoder_pos_noise: float = 0.0        # std of joint position readings [rad]
    encoder_vel_noise: float = 0.0        # std of joint velocity readings [rad/s]
    dropout_prob: float = 0.0             # per-step chance the observation freezes for 1-3 steps
    push_prob: float = 0.0                # per-step chance a resting ball gets nudged (APPROACH)
    push_speed_min: float = 0.05          # nudge speed range [m/s]
    push_speed_max: float = 0.20
    n_distractors: int = 0

    @property
    def box_rim_z(self) -> float:
        return self.table_z + self.box_depth


# Full-strength ranges. A level scales each range's deviation from nominal:
# "off" = 0, "mild" = 0.5, "full" = 1.
SCENARIO_RANGES = {
    "table_z_halfrange": 0.08,            # table_z = 0.25 +/- 0.08  (0.17 ... 0.33)
    "table_shift": 0.02,                  # +/- metres in X and Y
    "ball_radius_scale": (0.85, 1.15),
    "box_half_scale": (0.85, 1.15),
    "box_depth_scale": (0.75, 1.25),
    "table_friction_scale": (0.7, 1.3),
    "box_friction_scale": (0.7, 1.3),
    "ball_solref_scale": (0.6, 1.6),
    "encoder_pos_noise": 0.002,
    "encoder_vel_noise": 0.02,
    "dropout_prob": 0.02,
    "push_prob": 0.01,
    "max_distractors": 2,
}
LEVEL_SCALE = {"off": 0.0, "mild": 0.5, "full": 1.0}


def worst_case_scenario(table_z: float) -> Scenario:
    """Largest ball and basket the ranges allow, at a given table height.

    Reachability maps are computed for this, so any smaller object is also
    placed somewhere the arm can actually handle.
    """
    r = SCENARIO_RANGES
    return Scenario(table_z=table_z,
                    ball_radius=BALL_RADIUS * r["ball_radius_scale"][1],
                    box_half=BOX_HALF_SIZE * r["box_half_scale"][1],
                    box_depth=BOX_DEPTH * r["box_depth_scale"][1])

APPROACH_HOVER_HEIGHT = 0.04     # Pinch point stays this far above the ball until aligned [m]
APPROACH_ALIGN_XY = 0.03         # ...then descends once within this XY distance [m]

# Residual action scaling
POS_RESIDUAL_SCALE = 0.015      # Max 15 mm position correction per step
GRIPPER_OPEN_MAX = 0.65         # Gripper open angle [rad]
GRIPPER_CLOSED = 0.05           # Gripper closed angle [rad]

# Phase transition thresholds
APPROACH_DIST_THRESH = 0.025     # Switch to GRASP when EE is within 25mm of ball
GRASP_LIFT_THRESH = 0.12         # Switch to DROP when ball is lifted 120mm above table
DROP_RELEASE_THRESH = 0.025      # Ball is over basket within 25mm XY

# Timing
DROP_HOVER_HEIGHT = 0.100        # TCP height above the basket rim while dropping [m]
                                 # (80 mm left the fingertips <30 mm from the walls in ~half of layouts)
CLEARANCE_MAX = 0.10             # Clearance is reported up to this distance [m]
GRASP_HOLD_DIST = 0.05           # Ball counts as "in hand" while within this of the TCP [m]
LOST_BALL_DIST = 0.06            # A dropped ball this far from the TCP sends the phase back to APPROACH [m]
# ---- Domain randomisation ranges (applied once per episode when enabled) ----
DR_BALL_MASS = (0.7, 1.3)          # x nominal (inertia scaled with it)
DR_BALL_FRICTION = (0.8, 1.2)      # x nominal, per friction coefficient
DR_PAD_FRICTION = (0.8, 1.2)       # x nominal, gripper finger geoms
DR_ACTUATOR_KP = (0.8, 1.2)        # x nominal position gain, per actuator
DR_ACTUATOR_KD = (0.7, 1.3)        # x nominal velocity gain, per actuator
DR_JOINT_DAMPING = (0.7, 1.5)      # x nominal, per arm/finger joint
DR_BOX_JITTER = 0.005              # +/- box position jitter [m]
DR_LATENCY_STEPS = (0, 2)          # whole env steps (10 ms each) between command and execution
DR_PERCEPTION_BIAS = 0.003         # per-episode std of the ball/box position estimate bias [m]
DR_PERCEPTION_NOISE = 0.0015       # per-step std of ball/box position estimate noise [m]
PHYSICS_SUBSTEPS = 5             # MuJoCo substeps per env step
MAX_EPISODE_STEPS = 500          # Steps before truncation
MAX_SETPOINT_STEP = 0.005        # The IK base target moves at most this far per env step
                                 # (0.5 m/s), so long moves are smooth trajectories rather
                                 # than full-torque jumps that shake the ball loose [m]
MIN_SEPARATION = 0.20            # Ball-to-basket distance is never below this [m]
SEPARATION_TOP_FRACTION = 0.90   # Ball/basket pairs are drawn from those at least this fraction
                                 # of the farthest pair the active arm can reach this episode
MAX_CARRY_TILT_DEG = 60.0        # Baskets are only placed where the arm can hold the ball above
                                 # them with the hand within this of vertical; tipped further
                                 # (e.g. reaching across the body) the ball slides out
# ---- What counts as success / failure ----
BOX_UPRIGHT_TILT_DEG = 15.0      # Basket must be within this of upright for a ball in it to count
BOX_FLIP_TILT_DEG = 30.0         # Tipped beyond this, the episode fails
BALL_SETTLED_SPEED = 0.10        # Ball must be slower than this inside the basket [m/s]...
SUCCESS_HOLD_STEPS = 5           # ...for this many consecutive steps


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
    randomization_level : {"off", "mild", "full"} or None
        How strongly each episode's scenario (table height, object sizes,
        friction, sensor noise, distractors, disturbances) deviates from the
        nominal world. ``None`` means "mild" when ``domain_randomization`` is
        on and "off" when it is off. Any level other than "off" also switches
        the physics randomisation (gains, damping, latency, ...) on.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # What the robot does while a delayed command has not arrived yet: hold
    # position (zero residual) with the gripper open.
    _NEUTRAL_ACTION = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def __init__(
        self,
        model_path: str = "openarm_pick_and_place.xml",
        reward_config: RewardConfig | None = None,
        domain_randomization: bool = True,
        obs_noise_scale: float = 0.002,
        render_mode: str | None = None,
        randomization_level: str | None = None,
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

        # Scenarios resize the table, ball and basket at run time. MuJoCo's
        # midphase culls collision pairs with bounding-volume trees that are
        # built once at compile time, so a table raised above its compiled
        # height would silently stop colliding (a basket sank into it and blew
        # up). Without the midphase, collisions use the per-geom bounds that
        # _set_geom_box keeps current. Measured cost: ~3 % per step.
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE

        # ---- IK solver (uses its own internal data, never corrupts sim) ----
        self.ik = OpenArmIK(self.model, data=self.data,
                            left_site=LEFT_PINCH_SITE, right_site=RIGHT_PINCH_SITE)

        # ---- Reward calculator ----
        self.reward_calc = RewardCalculator(reward_config)

        # ---- Config ----
        if randomization_level is not None and randomization_level not in LEVEL_SCALE:
            raise ValueError(f"randomization_level must be one of {list(LEVEL_SCALE)}")
        self.scenario_level = randomization_level or ("mild" if domain_randomization else "off")
        self.domain_randomization = (self.scenario_level != "off") if randomization_level else domain_randomization
        self.obs_noise_scale = obs_noise_scale

        # ---- Resolve joint/actuator/body/geom indices ----
        self._setup_indices()

        # ---- Observation & Action spaces ----
        # obs: skill(3) + arm(2) + q(7) + qdot(7) + ee_pos(3) + ee_quat(4) +
        #      gripper(1) + ball_ee(3) + ball_vel(3) + ball_basket(3) +
        #      clearance(1) + prev_action(4) = 41
        self.obs_dim = 41
        self.action_dim = 4  # delta_pos(3) + gripper(1)

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
        self._action_delay: int = 0
        self._action_queue: list = []
        self._perception_bias = {"ball": np.zeros(3), "box": np.zeros(3)}

        self._grasp_quat = yaw_quat(DEFAULT_GRASP_YAW["left"])
        self._setpoint = np.zeros(3)
        self._ball_lost = False
        self._in_box_steps = 0
        self._sc = Scenario()
        self._sc_info = asdict(self._sc)
        self._cells = None
        self._last_obs = None
        self._dropout_left = 0
        self._placements = self._load_placements()
        self._table_z_range = (
            (float(self._placements["heights"][0]), float(self._placements["heights"][-1]))
            if self._placements is not None else (TABLE_Z, TABLE_Z))

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
            m, mujoco.mjtObj.mjOBJ_SITE, LEFT_PINCH_SITE)
        self._right_site_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, RIGHT_PINCH_SITE)

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

        # Nominal physical parameters. Domain randomisation always starts from
        # these (restored every reset) so scalings never compound across episodes.
        self._nominal = {
            "actuator_gainprm": m.actuator_gainprm.copy(),
            "actuator_biasprm": m.actuator_biasprm.copy(),
            "dof_damping": m.dof_damping.copy(),
            "geom_friction": m.geom_friction.copy(),
            "body_mass": m.body_mass.copy(),
            "body_inertia": m.body_inertia.copy(),
            # geometry that scenarios edit (bounds must be kept in sync: see _set_geom_box)
            "geom_pos": m.geom_pos.copy(),
            "geom_size": m.geom_size.copy(),
            "geom_rbound": m.geom_rbound.copy(),
            "geom_aabb": m.geom_aabb.copy(),
            "geom_solref": m.geom_solref.copy(),
            "body_pos": m.body_pos.copy(),
        }
        self._table_geom_id = m.geom("workspace_table").id
        self._box_part_ids = {
            name: m.geom(name).id
            for name in ("box_bottom", "box_wall_f", "box_wall_b", "box_wall_l", "box_wall_r")
        }
        self._distractor_bodies = [m.body(f"distractor_{k}").id for k in range(N_DISTRACTORS)]
        self._distractor_geoms = [m.geom(f"distractor_{k}_geom").id for k in range(N_DISTRACTORS)]
        self._distractor_mask = np.zeros(m.ngeom, dtype=bool)
        self._distractor_mask[self._distractor_geoms] = True
        # Joints/actuators/geoms the randomisation touches
        self._dr_actuators = np.concatenate([
            self._left_actuators, self._right_actuators,
            [self._left_finger_act, self._right_finger_act]]).astype(int)
        self._dr_dofs = np.concatenate([
            self._left_dof_adr_full(), self._right_dof_adr_full()]).astype(int)
        self._finger_pad_geoms = np.array(
            [g for ids in self._finger_geom_ids.values() for g in ids], dtype=int)

        # Box body id (ball id is stored above) for direct xpos indexing
        self._box_body_id = box_bid

        # Collision geoms used for the true arm-to-basket distance. Visual-only
        # geoms (no contype/conaffinity) are excluded: they can't collide, so
        # they must not shrink the reported clearance.
        #
        # mj_geomDistance is NOT used: for mesh-vs-box pairs it can return an
        # exact 0 with the witness points tens of mm apart (observed with
        # MuJoCo 3.10, independent of ccd_iterations). Instead the distance is
        # measured from each arm collision mesh's vertices to the basket's box
        # geoms, which is exact for the box side and tight for dense meshes.
        collidable = (m.geom_contype > 0) | (m.geom_conaffinity > 0)
        self._box_clearance_geoms = np.array(
            sorted(i for i in self._box_geom_ids if collidable[i]), dtype=int)
        # The basket walls are axis-aligned in the box body frame, so their
        # centres/half-sizes are constant there and points only need to be
        # moved into that one frame per step.
        assert np.allclose(m.geom_quat[self._box_clearance_geoms], [1, 0, 0, 0]), \
            "basket geoms are expected to be axis-aligned in the box body frame"
        self._refresh_box_cache()
        self._arm_clearance = {}
        for side, ids in (("left", self._left_arm_geom_ids),
                          ("right", self._right_arm_geom_ids)):
            entries = []
            for gid in sorted(i for i in ids if collidable[i]):
                if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = m.geom_dataid[gid]
                    v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                    verts = m.mesh_vert[v0:v0 + nv].reshape(-1, 3).astype(np.float64)
                    verts = _hull_vertices(verts)
                else:
                    verts = None  # primitive: handled via mj_geomDistance
                entries.append((gid, float(m.geom_rbound[gid]), verts))
            self._arm_clearance[side] = entries
        self._fromto = np.zeros(6)
        self._home_cloud = self._build_home_cloud()
        self._home_low_z = float(self._home_cloud[:, 2].min())     # lowest point of the home-pose hands
        self._start_cache = {}

        # DoF addresses for velocity readout
        self._left_dof_adr = self.ik._left_dof_adr
        self._right_dof_adr = self.ik._right_dof_adr

    def _build_home_cloud(self, q_left=None, q_right=None) -> np.ndarray:
        """World-frame points on both arms' collision geometry at a start pose.

        Objects are never placed where the start-pose hands already are, which
        matters once the table can be raised up towards them. Defaults to the
        home pose.
        """
        m = self.model
        tmp = mujoco.MjData(m)
        tmp.qpos[self._left_qpos_adr] = LEFT_ARM_HOME if q_left is None else q_left
        tmp.qpos[self._right_qpos_adr] = RIGHT_ARM_HOME if q_right is None else q_right
        tmp.qpos[self._left_finger_qpos] = GRIPPER_OPEN_MAX
        tmp.qpos[self._right_finger_qpos] = -GRIPPER_OPEN_MAX
        mujoco.mj_kinematics(m, tmp)
        pts = []
        for entries in self._arm_clearance.values():
            for gid, _, verts in entries:
                if verts is not None:
                    pts.append(verts @ tmp.geom_xmat[gid].reshape(3, 3).T + tmp.geom_xpos[gid])
        return np.concatenate(pts)

    def _start_pose(self, table_z: float):
        """Joint angles and point cloud of the pose both arms start an episode in.

        The home pose, unless the table is high enough that the open fingers
        would touch it; then both hands start raised just enough (found by IK)
        to clear the table by ``START_CLEARANCE``.
        """
        need = table_z + START_CLEARANCE
        lift = max(0.0, need - self._home_low_z)
        key = round(lift, 3)
        if key in self._start_cache:
            return self._start_cache[key]
        if lift == 0.0:
            result = (LEFT_ARM_HOME.copy(), RIGHT_ARM_HOME.copy(), self._home_cloud_nominal())
        else:
            tmp = mujoco.MjData(self.model)
            tmp.qpos[self._left_qpos_adr] = LEFT_ARM_HOME
            tmp.qpos[self._right_qpos_adr] = RIGHT_ARM_HOME
            mujoco.mj_kinematics(self.model, tmp)
            p_left = tmp.site_xpos[self._left_site_id].copy()
            p_right = tmp.site_xpos[self._right_site_id].copy()
            total = lift
            for _ in range(4):
                self.ik.data.qpos[:] = tmp.qpos
                mujoco.mj_kinematics(self.model, self.ik.data)
                q_left = self.ik.solve_left(p_left + [0, 0, total], q_init=LEFT_ARM_HOME).q_arm
                q_right = self.ik.solve_right(p_right + [0, 0, total], q_init=RIGHT_ARM_HOME).q_arm
                cloud = self._build_home_cloud(q_left, q_right)
                deficit = need - float(cloud[:, 2].min())
                if deficit <= 0.002:
                    break
                total += deficit
            result = (q_left, q_right, cloud)
        self._start_cache[key] = result
        return result

    def _home_cloud_nominal(self):
        return self._build_home_cloud()

    def _blocked_by_home_hands(self, xy: np.ndarray, half: float, top_z: float, margin: float = 0.03):
        """Which candidate positions (n, 2) have the home-pose hands in the way?

        An object whose footprint (half-width ``half``) and height reach
        ``top_z`` is blocked if any hand geometry lies within ``margin`` of it.
        """
        pts = self._home_cloud
        low = pts[pts[:, 2] < top_z + margin]
        if len(low) == 0:
            return np.zeros(len(xy), dtype=bool)
        near = np.abs(xy[:, None, :] - low[None, :, :2]) < (half + margin)
        return (near[..., 0] & near[..., 1]).any(axis=1)

    def _refresh_box_cache(self):
        """Re-read the basket geometry used by the clearance computation.

        Call after anything changes the basket's size (a scenario).
        """
        m = self.model
        self._box_local_centres = m.geom_pos[self._box_clearance_geoms].copy()   # (B, 3)
        self._box_half_sizes = m.geom_size[self._box_clearance_geoms].copy()      # (B, 3)
        self._box_bound_radius = float(
            (np.linalg.norm(self._box_local_centres, axis=1)
             + np.linalg.norm(self._box_half_sizes, axis=1)).max())

    def _left_dof_adr_full(self):
        """DoF addresses of the left arm joints and both left finger joints."""
        m = self.model
        return np.concatenate([
            self.ik._left_dof_adr,
            [m.jnt_dofadr[m.joint(f"openarm_left_finger_joint{k}").id] for k in (1, 2)]])

    def _right_dof_adr_full(self):
        m = self.model
        return np.concatenate([
            self.ik._right_dof_adr,
            [m.jnt_dofadr[m.joint(f"openarm_right_finger_joint{k}").id] for k in (1, 2)]])

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
        self._set_gripper("left", 1.0, teleport=True)
        self._set_gripper("right", 1.0, teleport=True)
        self._gripper_open_ratio = 1.0

        # This episode's world. Everything is rebuilt from the nominal model so
        # nothing compounds from one episode to the next.
        self._restore_nominal()
        level = self.scenario_level if self.domain_randomization else "off"
        self._sc = self._sample_scenario(level, (options or {}).get("scenario"))
        self._sc_info = asdict(self._sc)
        self._apply_scenario(self._sc)

        # Both arms start from the home pose, or a raised one above a tall table
        q_left, q_right, self._home_cloud = self._start_pose(self._sc.table_z)
        self.data.qpos[self._left_qpos_adr] = q_left
        self.data.qpos[self._right_qpos_adr] = q_right
        self.data.ctrl[self._left_actuators] = q_left
        self.data.ctrl[self._right_actuators] = q_right
        self._prepare_cells(self._sc)

        # Randomize ball and box positions (within what this arm can reach)
        self._ball_pos, self._box_pos, grasp_yaw = self._sample_positions(self._active_arm)
        self._grasp_quat = yaw_quat(grasp_yaw)
        self._place_distractors(self._ball_pos, self._box_pos)
        self._relocate_objects(self._ball_pos, self._box_pos)

        # Physics / actuation / latency / perception randomisation
        self._action_delay = 0
        self._perception_bias = {"ball": np.zeros(3), "box": np.zeros(3)}
        if self.domain_randomization:
            self._apply_domain_randomization()
        self._action_queue = [self._NEUTRAL_ACTION.copy() for _ in range(self._action_delay)]

        # Settle physics
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # Sync IK solver
        self.ik.sync_from(self.data)

        # The base trajectory starts where the gripper is
        self._setpoint = self._get_ee_state()[0].astype(np.float64)

        # Reset state
        self._phase = PHASE_APPROACH
        self._step_count = 0
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._ball_grasped = False
        self._ball_released = False
        self._ball_lost = False
        self._in_box_steps = 0
        self.reward_calc.reset()

        self._dropout_left = 0
        obs = self._get_obs()
        self._last_obs = obs
        info = {
            "active_arm": self._active_arm,
            "phase": PHASE_NAMES[self._phase],
            "ball_pos": self._ball_pos.copy(),
            "box_pos": self._box_pos.copy(),
            "scenario": self._sc_info,
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
        policy_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._step_count += 1

        # Command latency: the robot executes the action issued `delay` steps
        # ago (0 unless domain randomisation drew a delay for this episode).
        # The policy still sees/pays for its own most recent command.
        self._action_queue.append(policy_action)
        action = self._action_queue.pop(0)

        # ---- 1. IK base target for current phase ----
        ik_target_pos, ik_target_quat = self._get_phase_target()

        # Move the base setpoint toward the phase target at a bounded speed.
        to_go = ik_target_pos - self._setpoint
        dist = np.linalg.norm(to_go)
        if dist > MAX_SETPOINT_STEP:
            to_go *= MAX_SETPOINT_STEP / dist
        self._setpoint = self._setpoint + to_go
        ik_target_pos = self._setpoint

        # ---- 2. Apply residual corrections ----
        delta_pos = action[:3] * POS_RESIDUAL_SCALE
        if self._active_arm == "right":
            # Observations are mirrored (Y flipped) for the right arm so the
            # policy sees a canonical left-arm world. Its action lives in that
            # same canonical frame, so un-mirror it before applying it in the
            # world frame; otherwise "+Y" moves the right arm the wrong way.
            delta_pos[1] = -delta_pos[1]
        gripper_cmd = action[3]  # [-1, 1] -> [closed, open]

        # Corrected Cartesian target
        corrected_pos = ik_target_pos + delta_pos

        # ---- 3. Solve IK and execute ----
        # (the solver recomputes kinematics itself, so skip it in the sync)
        self.ik.sync_from(self.data, forward=False)

        # With no orientation target (carrying / dropping), hold the wrist in
        # its current configuration instead of letting the solver's posture
        # bias swing the hand back toward the home pose, which tips a held
        # ball out of the fingers. While orientation is commanded the home
        # posture is the better default.
        def hold_posture(q_now):
            return q_now if ik_target_quat is None else None
        if self._active_arm == "left":
            q_now = self.data.qpos[self._left_qpos_adr].copy()
            ik_result = self.ik.solve_left(
                target_pos=corrected_pos, target_quat=ik_target_quat,
                q_init=q_now, posture=hold_posture(q_now),
            )
        else:
            q_now = self.data.qpos[self._right_qpos_adr].copy()
            ik_result = self.ik.solve_right(
                target_pos=corrected_pos, target_quat=ik_target_quat,
                q_init=q_now, posture=hold_posture(q_now),
            )

        # Apply joint commands. Only the PD servo targets are set: the arm
        # reaches them through the actuators and dynamics (finite gains,
        # damping, torque limits), it is never teleported.
        actuators = self._left_actuators if self._active_arm == "left" else self._right_actuators
        self.data.ctrl[actuators] = ik_result.q_arm

        # Apply gripper command
        self._gripper_open_ratio = 0.5 * (gripper_cmd + 1.0)  # Map [-1,1] -> [0,1]
        self._set_gripper(self._active_arm, self._gripper_open_ratio)

        self._maybe_push_ball()

        # Step physics
        for _ in range(PHYSICS_SUBSTEPS):
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
            arm_box_contact=arm_basket_contact)
        arm_obstacle_contact = self._check_arm_distractor_contact()

        # Ball in box check (inside the basket's own frame, basket upright on the table)
        ball_in_box = self._check_ball_in_box(ball_pos, box_pos)
        box_tilt_deg = self._box_tilt_deg()

        # ---- 5. Grasp state, then phase transitions ----
        # Grasp state is read off the physics: the ball is held only while
        # friction between the fingers keeps it with the hand.
        if not self._ball_released:
            ball_in_hand = np.linalg.norm(ee_pos - ball_pos) < GRASP_HOLD_DIST
            if self._ball_grasped:
                if self._gripper_open_ratio > 0.7:
                    # Gripper opened: a deliberate release over the basket,
                    # otherwise the grasp is simply given up.
                    if self._phase == PHASE_DROP:
                        self._ball_released = True
                    self._ball_grasped = False
                elif not ball_in_hand:
                    self._ball_grasped = False          # slipped out of the hand
            elif (self._phase == PHASE_GRASP and finger_contacts[0]
                  and finger_contacts[1] and self._gripper_open_ratio < 0.3):
                self._ball_grasped = True               # both fingers pinching

        self._update_phase(ee_pos, ball_pos, box_pos, finger_contacts, ball_in_box)

        # A ball dropped in flight is re-grasped only if it landed where the arm
        # can reach it with a vertical grasp; otherwise the episode is lost.
        ball_unreachable = False
        if self._ball_lost:
            self._ball_lost = False
            yaw = self._grasp_yaw_at(ball_pos[:2])
            if yaw is None:
                ball_unreachable = True
            else:
                self._grasp_quat = yaw_quat(yaw)

        # ---- 6. Success and failure ----
        # Success: the ball was let go and has come to rest inside the upright
        # basket (held for a few steps, so a ball bouncing through doesn't count).
        if (ball_in_box and self._ball_released
                and np.linalg.norm(ball_vel[:3]) < BALL_SETTLED_SPEED):
            self._in_box_steps += 1
        else:
            self._in_box_steps = 0
        # Failure: the gripper/arm touched the basket, or the basket was tipped.
        failure = None
        if arm_basket_contact:
            failure = "gripper_touched_box"
        elif box_tilt_deg > BOX_FLIP_TILT_DEG:
            failure = "box_flipped"
        success = failure is None and self._in_box_steps >= SUCCESS_HOLD_STEPS

        # ---- 7. Compute reward ----
        reward, reward_info = self.reward_calc.compute(
            phase=self._phase,
            ee_pos=ee_pos,
            ball_pos=ball_pos,
            basket_pos=box_pos,
            ball_vel=ball_vel,
            finger_contacts=finger_contacts,
            basket_clearance=basket_clearance,
            # hitting a distractor is penalised like hitting the basket
            arm_basket_contact=arm_basket_contact or arm_obstacle_contact,
            gripper_open_ratio=self._gripper_open_ratio,
            action=policy_action,
            prev_action=self._prev_action,
            table_z=self._sc.table_z,
            box_rim_z=self._sc.box_rim_z,
            ball_in_box=success,
            ball_released=self._ball_released,
            task_failed=failure is not None,
        )

        self._prev_action = policy_action.copy()

        # ---- 8. Termination ----
        # The episode ends on a verified success, or on a failure that cannot
        # be undone (basket touched or tipped).
        terminated = success or failure is not None
        truncated = False

        # Truncate on timeout
        if self._step_count >= MAX_EPISODE_STEPS:
            truncated = True

        # Truncate if ball fell off table, or landed out of the arm's reach
        if ball_pos[2] < self._sc.table_z - 0.05 or ball_unreachable:
            truncated = True

        # Nothing has touched the simulation since the readout above, so the
        # observation reuses it instead of recomputing EE state and contacts.
        obs = self._get_obs(ee_state=(ee_pos, ee_quat),
                            basket_clearance=basket_clearance)
        obs = self._apply_observation_dropout(obs)
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
            "ball_unreachable": ball_unreachable,
            "arm_basket_contact": arm_basket_contact,
            "arm_obstacle_contact": arm_obstacle_contact,
            "basket_clearance": basket_clearance,
            "scenario": self._sc_info,
            # True only for a genuine success: the released ball rests inside the
            # upright basket and the gripper never touched the basket.
            "is_success": success,
            "failure": failure,
            "box_tilt_deg": box_tilt_deg,
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

        # Encoder noise on the proprioceptive readings
        sc = self._sc
        if sc.encoder_pos_noise > 0:
            joint_pos = joint_pos + self._rng.normal(0, sc.encoder_pos_noise, 7).astype(np.float32)
        if sc.encoder_vel_noise > 0:
            joint_vel = joint_vel + self._rng.normal(0, sc.encoder_vel_noise, 7).astype(np.float32)

        ee_pos, ee_quat = ee_state if ee_state is not None else self._get_ee_state()
        gripper_state = np.array([self._gripper_open_ratio], dtype=np.float32)

        # Exteroception
        ball_pos = self._perceive("ball", self.data.xpos[self._ball_body_id])
        ball_vel = self.data.qvel[self._ball_dof_adr:self._ball_dof_adr + 3].copy()
        box_pos = self._perceive("box", self.data.xpos[self._box_body_id])

        ball_ee_vec = (ball_pos - ee_pos).astype(np.float32)
        ball_basket_vec = (box_pos - ball_pos).astype(np.float32)
        if basket_clearance is None:
            basket_clearance = self._compute_basket_clearance()
        basket_clearance = np.array([basket_clearance], dtype=np.float32)

        # The gripper height is reported relative to the table top, so the
        # observation does not change when the table is raised or lowered.
        ee_pos = ee_pos.copy()
        ee_pos[2] -= np.float32(self._sc.table_z)

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

        # Mirror quaternion (w, x, y, z) across the XZ plane (y -> -y).
        # A rotation is an axial vector, so a reflection with normal Y keeps
        # the component along Y and flips the other two: (w, -x, y, -z).
        ee_quat[1] = -ee_quat[1]  # x component
        ee_quat[3] = -ee_quat[3]  # z component

        return ee_pos, ee_quat, ball_ee_vec, ball_basket_vec

    # =================================================================
    # Phase target generation (IK base trajectory)
    # =================================================================

    def _get_phase_target(self):
        """Compute the IK base target for the current phase.

        Returns ``(pinch_pos, quat)``. ``quat`` is the gripper-down grasp
        orientation while approaching/grasping, and ``None`` (position-only)
        once the ball is lifted: a vertical hand is not reachable at carry
        and drop heights across the workspace, and a gripped ball does not
        need it.
        """
        # The base trajectory is planned from the *perceived* object poses.
        ball_pos = self._perceive("ball", self.data.xpos[self._ball_body_id])
        box_pos = self._perceive("box", self.data.xpos[self._box_body_id])
        grasp_quat = self._grasp_quat

        if self._phase == PHASE_APPROACH:
            # Hover just above the ball until aligned in XY, then descend so
            # the open fingers straddle it.
            xy_err = np.linalg.norm(self._get_ee_state()[0][:2] - ball_pos[:2])
            target = ball_pos.copy()
            if xy_err > APPROACH_ALIGN_XY:
                target[2] += APPROACH_HOVER_HEIGHT
            return target, grasp_quat

        elif self._phase == PHASE_GRASP:
            if not self._ball_grasped:
                # Close the fingers around the ball where it sits
                return ball_pos.copy(), grasp_quat
            # Lift straight up with the ball in hand
            target = ball_pos.copy()
            target[2] = self._sc.table_z + 0.170
            return target, None

        elif self._phase == PHASE_DROP:
            # Above the box with safe clearance
            target = box_pos.copy()
            target[2] = self._sc.box_rim_z + DROP_HOVER_HEIGHT
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
                lift_height = ball_pos[2] - self._sc.table_z
                if lift_height > GRASP_LIFT_THRESH:
                    self._phase = PHASE_DROP

        elif self._phase == PHASE_DROP:
            # Ball slipped out before release: go back and pick it up again.
            # (Episode ends on success or timeout otherwise.)
            if (not self._ball_grasped and not self._ball_released
                    and np.linalg.norm(ee_pos - ball_pos) > LOST_BALL_DIST):
                self._phase = PHASE_APPROACH
                self._ball_lost = True

    # =================================================================
    # Gripper control
    # =================================================================

    def _set_gripper(self, side: str, open_ratio: float, teleport: bool = False):
        """Command the gripper opening through its position servo.

        ``teleport=True`` also sets the finger joint position directly; it is
        only for placing the gripper in its start state at reset. During an
        episode the fingers move only by actuation, so they stop (and exert
        servo force) when they meet the ball.
        """
        open_ratio = np.clip(open_ratio, 0.0, 1.0)
        if side == "left":
            val = GRIPPER_CLOSED + open_ratio * (GRIPPER_OPEN_MAX - GRIPPER_CLOSED)
            self.data.ctrl[self._left_finger_act] = val
            if teleport:
                self.data.qpos[self._left_finger_qpos] = val
        else:
            val = -(GRIPPER_CLOSED + open_ratio * (GRIPPER_OPEN_MAX - GRIPPER_CLOSED))
            self.data.ctrl[self._right_finger_act] = val
            if teleport:
                self.data.qpos[self._right_finger_qpos] = val

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

    def _check_arm_distractor_contact(self) -> bool:
        """Check if the active arm touches any distractor block."""
        if self._sc.n_distractors == 0 or self.data.ncon == 0:
            return False
        geoms = self._contact_geoms()
        g1, g2 = geoms[:, 0], geoms[:, 1]
        arm = self._arm_geom_mask[self._active_arm]
        dis = self._distractor_mask
        return bool(((arm[g1] & dis[g2]) | (arm[g2] & dis[g1])).any())

    def _compute_basket_clearance(self, arm_box_contact=None) -> float:
        """Minimum distance from the active arm's collision geometry to the basket.

        Measured on the real link and finger meshes (not a point-to-box
        proxy), so hovering above the basket is correctly "clear" while a
        finger brushing a wall is not. The result is capped at
        ``CLEARANCE_MAX`` and is 0 when anything is touching.

        For each arm collision mesh the distance from its vertices to the
        basket's box geoms is used; this slightly overestimates the true
        distance by at most the mesh vertex spacing. Geoms whose bounding
        sphere is farther than ``CLEARANCE_MAX`` from the basket are skipped.

        ``arm_box_contact`` may be passed in when already known for the
        current simulation state, to avoid rescanning the contacts.
        """
        if arm_box_contact is None:
            arm_box_contact = self._check_arm_box_contact()
        if arm_box_contact:
            return 0.0

        m, d = self.model, self.data
        centre = d.xpos[self._box_body_id]
        box_R = d.xmat[self._box_body_id].reshape(3, 3)

        # Gather the vertices of every arm geom that could be within range.
        best = CLEARANCE_MAX
        near = []
        for gid, rbound, verts in self._arm_clearance[self._active_arm]:
            gpos = d.geom_xpos[gid]
            # Bounding-sphere reject: this geom cannot be closer than `best`.
            if np.linalg.norm(gpos - centre) - rbound - self._box_bound_radius >= best:
                continue
            if verts is None:
                for g2 in self._box_clearance_geoms:
                    dist = mujoco.mj_geomDistance(m, d, gid, g2, best, self._fromto)
                    if dist < best:
                        best = dist
                continue
            near.append(verts @ d.geom_xmat[gid].reshape(3, 3).T + gpos)
        if not near:
            return max(best, 0.0)

        # One transform into the basket frame, then point-to-box distances.
        pts = (np.concatenate(near) - centre) @ box_R                     # (n, 3)
        excess = np.abs(pts[None] - self._box_local_centres[:, None]) - self._box_half_sizes[:, None]
        np.maximum(excess, 0.0, out=excess)
        dist = float(np.sqrt(np.einsum("bnk,bnk->bn", excess, excess).min()))
        return max(min(best, dist), 0.0)

    def _box_tilt_deg(self) -> float:
        """How far the basket is tipped away from upright, in degrees."""
        up_z = self.data.xmat[self._box_body_id][8]          # basket z-axis . world z
        return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))

    def _check_ball_in_box(self, ball_pos, box_pos) -> bool:
        """Is the ball really inside the basket?

        Measured in the basket's own frame, so it stays correct if the basket
        has been pushed or turned: the ball centre must be within the inner
        walls and below the rim. The basket must also be upright and sitting on
        the table - a ball lying next to (or under) a tipped basket is not in it.
        """
        sc = self._sc
        if self._box_tilt_deg() > BOX_UPRIGHT_TILT_DEG:
            return False
        if abs(box_pos[2] - sc.table_z) > 0.01:
            return False
        R = self.data.xmat[self._box_body_id].reshape(3, 3)
        p = R.T @ (np.asarray(ball_pos, dtype=np.float64) - box_pos)
        inner = sc.box_half - 2 * BOX_WALL_T
        return bool(abs(p[0]) < inner and abs(p[1]) < inner
                    and 2 * BOX_BOTTOM_T < p[2] < sc.box_depth)

    # =================================================================
    # Object placement & randomization
    # =================================================================

    def _load_placements(self):
        """Read the reachability maps (one per table height).

        For each height and arm: which grid cells a ball can be grasped in
        (with the hand yaw that works there) and which a basket can be dropped
        into. Cells are kept only if all eight neighbours are reachable too, so
        a position jittered anywhere inside the cell stays reachable. Returns
        None when the maps file is absent (a small legacy rectangle is used).
        """
        path = SCRIPT_DIR / WORKSPACE_MAPS_FILE
        if not path.exists():
            return None
        maps = np.load(path)
        xs, ys, res = maps["xs"], maps["ys"], float(maps["grid_res"])
        heights = np.atleast_1d(maps["heights"]).astype(float) if "heights" in maps.files else np.array([TABLE_Z])
        gx, gy = np.meshgrid(xs, ys, indexing="ij")

        def interior(mask):
            core = mask.copy()
            core[0, :] = core[-1, :] = core[:, 0] = core[:, -1] = False
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    core &= np.roll(np.roll(mask, di, axis=0), dj, axis=1)
            return core

        def per_height(arr):
            arr = np.asarray(arr)
            return arr[None] if arr.ndim == 2 else arr          # single-height files are 2-D

        out = {"res": res, "heights": heights, "gx": gx, "gy": gy,
               "ball_ok": {}, "ball_yaw": {}, "box_ok": {}}
        for side in ("left", "right"):
            yaw = per_height(maps[f"{side}_ball_yaw"])
            box = per_height(maps[f"{side}_box_ok"])
            if f"{side}_box_tilt" in maps.files:
                tilt = per_height(maps[f"{side}_box_tilt"])
                box = box & (np.nan_to_num(tilt, nan=180.0) <= MAX_CARRY_TILT_DEG)
            out["ball_yaw"][side] = yaw
            out["ball_ok"][side] = np.stack([interior(np.isfinite(y)) for y in yaw])
            out["box_ok"][side] = np.stack([interior(b) for b in box])
        return out

    def _prepare_cells(self, sc):
        """Candidate ball/basket cells for this episode's table and objects.

        A cell must be reachable at both map heights bracketing the table
        height, and the object must sit entirely on this episode's table top.
        """
        self._cells = None
        P = self._placements
        if P is None:
            return
        hs, res, gx, gy = P["heights"], P["res"], P["gx"], P["gy"]
        z = float(np.clip(sc.table_z, hs[0], hs[-1]))     # beyond the maps: nearest edge map
        hi = min(int(np.searchsorted(hs, z)), len(hs) - 1)
        lo = hi if hs[hi] <= z + 1e-9 else max(hi - 1, 0)
        near = lo if abs(hs[lo] - z) <= abs(hs[hi] - z) else hi

        tab = self._table_geom_id
        cx, cy = self.model.geom_pos[tab][:2]
        hx, hy = self.model.geom_size[tab][:2]

        def on_table(half):
            return (np.abs(gx - cx) <= hx - half - 0.01) & (np.abs(gy - cy) <= hy - half - 0.01)

        cells = {}
        for side in ("left", "right"):
            ball = P["ball_ok"][side][lo] & P["ball_ok"][side][hi] & on_table(sc.ball_radius + res / 2)
            box = P["box_ok"][side][lo] & P["box_ok"][side][hi] & on_table(sc.box_half + res / 2)
            yaw = P["ball_yaw"][side][near]
            ball_xy = np.stack([gx[ball], gy[ball]], axis=1)
            box_xy = np.stack([gx[box], gy[box]], axis=1)
            # ...and clear of the hands standing at their home pose
            keep_b = ~self._blocked_by_home_hands(ball_xy, sc.ball_radius + res / 2, sc.table_z + 2 * sc.ball_radius)
            keep_x = ~self._blocked_by_home_hands(box_xy, sc.box_half + res / 2, sc.box_rim_z)
            cells[side] = {
                "ball_xy": ball_xy[keep_b],
                "ball_yaw": yaw[ball][keep_b],
                "box_xy": box_xy[keep_x],
            }
            if len(cells[side]["ball_xy"]) == 0 or len(cells[side]["box_xy"]) == 0:
                raise RuntimeError(
                    f"No reachable ball/basket cells for the {side} arm at table height "
                    f"{sc.table_z:.3f} m; recompute the maps (tools/compute_workspace.py).")
        self._cells = cells

    def _grasp_yaw_at(self, xy):
        """Hand yaw for a vertical grasp at ``xy``, or None if not graspable there."""
        if self._cells is None:
            return DEFAULT_GRASP_YAW[self._active_arm]
        pl = self._cells[self._active_arm]
        dist = np.linalg.norm(pl["ball_xy"] - xy, axis=1)
        i = int(np.argmin(dist))
        return float(pl["ball_yaw"][i]) if dist[i] <= self._placements["res"] else None

    def _sample_positions(self, arm: str):
        """Sample ball and basket positions over everything the arm can do.

        The pair is drawn from the farthest-apart combinations of graspable
        ball cell and reachable basket cell: every pair whose distance is at
        least ``SEPARATION_TOP_FRACTION`` of the largest distance the active
        arm can cover this episode. That keeps the carry as long as the arm
        allows while still varying where the two objects are.

        Returns ``(ball_pos, box_pos, grasp_yaw_deg)``.
        """
        rng, sc = self._rng, self._sc
        if self._cells is None:
            return self._sample_positions_legacy(arm)

        pl, half = self._cells[arm], self._placements["res"] / 2.0
        dist = np.linalg.norm(pl["ball_xy"][:, None, :] - pl["box_xy"][None, :, :], axis=2)
        floor = max(MIN_SEPARATION, SEPARATION_TOP_FRACTION * float(dist.max()))
        ii, jj = np.nonzero(dist >= floor)
        if ii.size:
            k = rng.integers(ii.size)
            i, j = int(ii[k]), int(jj[k])
            ball_xy = pl["ball_xy"][i] + rng.uniform(-half, half, size=2)
            box_xy = pl["box_xy"][j] + rng.uniform(-half, half, size=2)
            return (np.array([ball_xy[0], ball_xy[1], sc.table_z + sc.ball_radius]),
                    np.array([box_xy[0], box_xy[1], sc.table_z]),
                    float(pl["ball_yaw"][i]))
        raise RuntimeError("No basket cell is far enough from any sampled ball cell.")

    def _sample_positions_legacy(self, arm: str):
        """Placement used when no reachability maps are available."""
        ws, sc = LEGACY_WORKSPACE[arm], self._sc
        ball = np.array([self._rng.uniform(*ws["x"]), self._rng.uniform(*ws["y"]),
                         sc.table_z + sc.ball_radius])
        box = ball.copy()
        box[0] = ws["x"][1] + 0.05 if ball[0] < 0.24 else ws["x"][0] - 0.05
        box[2] = sc.table_z
        return ball, box, DEFAULT_GRASP_YAW[arm]

    # =================================================================
    # Scenarios
    # =================================================================

    def _sample_scenario(self, level: str, overrides: dict | None = None) -> Scenario:
        """Draw this episode's scenario; ``overrides`` pins fields explicitly."""
        k, R, rng = LEVEL_SCALE[level], SCENARIO_RANGES, self._rng
        sc = Scenario()
        if k > 0:
            lo, hi = self._table_z_range
            sc.table_z = float(np.clip(TABLE_Z + k * rng.uniform(-1, 1) * R["table_z_halfrange"], lo, hi))
            sc.table_dx, sc.table_dy = (k * rng.uniform(-R["table_shift"], R["table_shift"], size=2)).tolist()

            def scale(key):                      # 1 + k * (draw - 1)
                return 1.0 + k * (rng.uniform(*R[key]) - 1.0)

            sc.ball_radius = BALL_RADIUS * scale("ball_radius_scale")
            sc.box_half = BOX_HALF_SIZE * scale("box_half_scale")
            sc.box_depth = BOX_DEPTH * scale("box_depth_scale")
            sc.table_friction_scale = scale("table_friction_scale")
            sc.box_friction_scale = scale("box_friction_scale")
            sc.ball_solref_scale = scale("ball_solref_scale")
            sc.encoder_pos_noise = k * R["encoder_pos_noise"]
            sc.encoder_vel_noise = k * R["encoder_vel_noise"]
            sc.dropout_prob = k * R["dropout_prob"]
            sc.push_prob = k * R["push_prob"]
            sc.n_distractors = int(rng.integers(0, int(round(k * R["max_distractors"])) + 1))
        if overrides:
            valid = {f.name for f in fields(Scenario)}
            unknown = set(overrides) - valid
            if unknown:
                raise ValueError(f"Unknown scenario field(s) {sorted(unknown)}; valid: {sorted(valid)}")
            for name, value in overrides.items():
                setattr(sc, name, type(getattr(sc, name))(value))
        return sc

    def _set_geom_box(self, gid, pos=None, half=None):
        """Edit a box geom's position/size AND the bounding data derived from it.

        MuJoCo's collision pre-filter reads ``geom_rbound`` / ``geom_aabb``,
        which are only computed at compile time. Resizing a geom without
        refreshing them makes collisions get skipped (a ball sank into a table
        that had been raised).
        """
        m = self.model
        if pos is not None:
            m.geom_pos[gid] = pos
        if half is not None:
            m.geom_size[gid] = half
            m.geom_rbound[gid] = float(np.linalg.norm(half))
            m.geom_aabb[gid] = [0.0, 0.0, 0.0, *half]

    def _apply_scenario(self, sc: Scenario):
        """Edit the (nominal) model so it matches ``sc``. Call after _restore_nominal()."""
        m, nom = self.model, self._nominal
        tab = self._table_geom_id

        # Table: top at table_z, standing on the floor
        npos, nsize = nom["geom_pos"][tab], nom["geom_size"][tab]
        self._set_geom_box(tab, pos=[npos[0] + sc.table_dx, npos[1] + sc.table_dy, sc.table_z / 2],
                           half=[nsize[0], nsize[1], sc.table_z / 2])
        m.geom_friction[tab, 0] = nom["geom_friction"][tab, 0] * sc.table_friction_scale

        # Ball: radius, and mass/inertia of a solid ball of the same density
        b, r = self._ball_geom_id, sc.ball_radius
        m.geom_size[b, 0] = r
        m.geom_rbound[b] = r
        m.geom_aabb[b] = [0.0, 0.0, 0.0, r, r, r]
        ratio = r / BALL_RADIUS
        m.body_mass[self._ball_body_id] = nom["body_mass"][self._ball_body_id] * ratio ** 3
        m.body_inertia[self._ball_body_id] = nom["body_inertia"][self._ball_body_id] * ratio ** 5
        m.geom_solref[b, 0] = nom["geom_solref"][b, 0] * sc.ball_solref_scale

        # Basket: an open box of half-width W and wall height D
        W, D, t, f = sc.box_half, sc.box_depth, BOX_WALL_T, BOX_BOTTOM_T
        parts = self._box_part_ids
        self._set_geom_box(parts["box_bottom"], pos=[0, 0, f], half=[W, W, f])
        self._set_geom_box(parts["box_wall_f"], pos=[W - t, 0, D / 2], half=[t, W, D / 2])
        self._set_geom_box(parts["box_wall_b"], pos=[-(W - t), 0, D / 2], half=[t, W, D / 2])
        self._set_geom_box(parts["box_wall_l"], pos=[0, W - t, D / 2], half=[W - 2 * t, t, D / 2])
        self._set_geom_box(parts["box_wall_r"], pos=[0, -(W - t), D / 2], half=[W - 2 * t, t, D / 2])
        for gid in parts.values():
            m.geom_friction[gid, 0] = nom["geom_friction"][gid, 0] * sc.box_friction_scale
        self._refresh_box_cache()

    def _place_distractors(self, ball_pos, box_pos):
        """Put this episode's distractor blocks on the table, clear of everything.

        They stay away from the ball, the basket and the straight line between
        them (the arm's path), and from each other. Unused ones are parked
        under the table.
        """
        m, sc, rng = self.model, self._sc, self._rng
        placed = []
        seg = box_pos[:2] - ball_pos[:2]
        seg_len2 = float(seg @ seg) + 1e-12
        side = 1.0 if self._active_arm == "left" else -1.0
        for k in range(N_DISTRACTORS):
            b, g = self._distractor_bodies[k], self._distractor_geoms[k]
            m.body_pos[b] = [0.0, 0.0, -5.0]                      # parked
            if k >= sc.n_distractors:
                continue
            half = np.array([rng.uniform(0.015, 0.03), rng.uniform(0.015, 0.03), rng.uniform(0.015, 0.03)])
            for _ in range(80):
                xy = np.array([rng.uniform(0.14, 0.50), side * rng.uniform(0.05, 0.45)])
                t = np.clip(((xy - ball_pos[:2]) @ seg) / seg_len2, 0.0, 1.0)
                path_dist = np.linalg.norm(xy - (ball_pos[:2] + t * seg))
                top = sc.table_z + 2 * half[2]
                if (not self._blocked_by_home_hands(xy[None], float(half[:2].max()), top)[0]
                        and np.linalg.norm(xy - ball_pos[:2]) > 0.10
                        and np.linalg.norm(xy - box_pos[:2]) > 0.12
                        and path_dist > 0.09
                        and all(np.linalg.norm(xy - p) > 0.10 for p in placed)):
                    tab = self._table_geom_id
                    cx, cy = m.geom_pos[tab][:2]
                    hx, hy = m.geom_size[tab][:2]
                    if abs(xy[0] - cx) < hx - 0.04 and abs(xy[1] - cy) < hy - 0.04:
                        self._set_geom_box(g, pos=[0, 0, 0], half=half)
                        m.body_pos[b] = [xy[0], xy[1], sc.table_z + half[2]]
                        placed.append(xy)
                        break

    def _maybe_push_ball(self):
        """Occasionally nudge a ball that is resting on the table (APPROACH only)."""
        sc = self._sc
        if sc.push_prob <= 0 or self._phase != PHASE_APPROACH or self._ball_grasped:
            return
        d, adr = self.data, self._ball_dof_adr
        resting = (np.linalg.norm(d.qvel[adr:adr + 3]) < 0.02
                   and d.xpos[self._ball_body_id][2] < sc.table_z + sc.ball_radius + 0.01)
        if resting and self._rng.random() < sc.push_prob:
            speed = self._rng.uniform(sc.push_speed_min, sc.push_speed_max)
            ang = self._rng.uniform(0.0, 2.0 * np.pi)
            d.qvel[adr:adr + 2] += speed * np.array([np.cos(ang), np.sin(ang)])

    def _apply_observation_dropout(self, obs):
        """With probability ``dropout_prob`` per step, repeat the last delivered
        observation for 1-3 steps (a lost sensor packet)."""
        sc = self._sc
        if sc.dropout_prob <= 0 or self._last_obs is None:
            self._last_obs = obs
            return obs
        if self._dropout_left == 0 and self._rng.random() < sc.dropout_prob:
            self._dropout_left = int(self._rng.integers(1, 4))
        if self._dropout_left > 0:
            self._dropout_left -= 1
            return self._last_obs
        self._last_obs = obs
        return obs

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

    def _restore_nominal(self):
        """Put every randomised model parameter back to its nominal value."""
        m = self.model
        m.actuator_gainprm[:] = self._nominal["actuator_gainprm"]
        m.actuator_biasprm[:] = self._nominal["actuator_biasprm"]
        m.dof_damping[:] = self._nominal["dof_damping"]
        m.geom_friction[:] = self._nominal["geom_friction"]
        m.body_mass[:] = self._nominal["body_mass"]
        m.body_inertia[:] = self._nominal["body_inertia"]
        for key in ("geom_pos", "geom_size", "geom_rbound", "geom_aabb", "geom_solref", "body_pos"):
            getattr(m, key)[:] = self._nominal[key]
        self._refresh_box_cache()

    def _apply_domain_randomization(self):
        """Randomise physics, actuation, latency and perception for sim-to-real.

        Runs once per episode on top of the nominal model:
        - ball mass (inertia scaled to match), ball and finger-pad friction
        - per-actuator servo gains (kp, kd) and per-joint damping
        - box position jitter
        - command latency (0-2 env steps)
        - a per-episode bias on the perceived ball/box positions (the
          per-step noise part is drawn in ``_perceive``)
        """
        m, rng, nom = self.model, self._rng, self._nominal

        # Ball mass (+ matching inertia) and friction
        # (on top of the scenario's ball size, which already set mass/inertia)
        scale = rng.uniform(*DR_BALL_MASS)
        m.body_mass[self._ball_body_id] *= scale
        m.body_inertia[self._ball_body_id] *= scale
        m.geom_friction[self._ball_geom_id] = (
            nom["geom_friction"][self._ball_geom_id]
            * rng.uniform(*DR_BALL_FRICTION, size=3))

        # Gripper finger-pad friction (one factor per pad)
        for g in self._finger_pad_geoms:
            m.geom_friction[g] = nom["geom_friction"][g] * rng.uniform(*DR_PAD_FRICTION)

        # Servo gains. For these position servos force = kp*(ctrl - q) - kd*qdot,
        # stored as gainprm[0] = kp, biasprm = [0, -kp, -kd].
        acts = self._dr_actuators
        kp_scale = rng.uniform(*DR_ACTUATOR_KP, size=len(acts))
        kd_scale = rng.uniform(*DR_ACTUATOR_KD, size=len(acts))
        m.actuator_gainprm[acts, 0] = nom["actuator_gainprm"][acts, 0] * kp_scale
        m.actuator_biasprm[acts, 1] = nom["actuator_biasprm"][acts, 1] * kp_scale
        m.actuator_biasprm[acts, 2] = nom["actuator_biasprm"][acts, 2] * kd_scale

        # Joint damping
        dofs = self._dr_dofs
        m.dof_damping[dofs] = nom["dof_damping"][dofs] * rng.uniform(
            *DR_JOINT_DAMPING, size=len(dofs))

        # Box position jitter
        self.data.qpos[self._box_qpos_adr:self._box_qpos_adr + 2] += rng.uniform(
            -DR_BOX_JITTER, DR_BOX_JITTER, size=2)

        # Command latency, in whole env steps
        self._action_delay = int(rng.integers(DR_LATENCY_STEPS[0], DR_LATENCY_STEPS[1] + 1))

        # Systematic error of the object pose estimate for this episode
        self._perception_bias = {
            "ball": rng.normal(0.0, DR_PERCEPTION_BIAS, size=3),
            "box": rng.normal(0.0, DR_PERCEPTION_BIAS, size=3),
        }

        mujoco.mj_forward(m, self.data)

    def _perceive(self, which: str, true_pos: np.ndarray) -> np.ndarray:
        """The policy/planner's estimate of an object position.

        Equals the true position unless domain randomisation is on, in which
        case it carries a per-episode bias plus per-step noise (a stand-in
        for camera calibration error and detection jitter).
        """
        pos = np.array(true_pos, dtype=np.float64)
        if self.domain_randomization:
            pos += self._perception_bias[which]
            pos += self._rng.normal(0.0, DR_PERCEPTION_NOISE, size=3)
        return pos


# =====================================================================
# Helper: reduce a mesh to its convex-hull vertices
# =====================================================================
def _hull_vertices(verts: np.ndarray) -> np.ndarray:
    """Keep only the convex-hull vertices of a point set.

    MuJoCo collides meshes by their convex hull, and interior vertices can
    never be the closest point of a convex shape, so dropping them makes
    the per-step clearance query several times cheaper. Falls back to all
    vertices if scipy is unavailable or the hull cannot be built.
    """
    try:
        from scipy.spatial import ConvexHull
        return verts[ConvexHull(verts).vertices]
    except Exception:
        return verts


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
