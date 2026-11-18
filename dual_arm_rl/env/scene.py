"""
The simulated world: model loading, per-episode scenarios, object placement,
randomisation (physics + visual) and state queries.

Everything here is environment generation or measurement. Nothing in this
module moves the robot toward a goal.

What changes from episode to episode
------------------------------------
* Always (every stage): table height and shift, ball radius, basket width and
  depth, friction and contact softness, and the ball-to-basket distance.
* With the curriculum stage: distractor blocks, random nudges to a resting
  ball, a ball that starts rolling, physics randomisation (mass, friction,
  servo gains, damping, command latency), visual randomisation (lighting,
  camera pose error, colour shifts) and sensor noise.

Placements come from per-arm reachability maps (``assets/workspace_maps.npz``,
built by ``dual_arm_rl/tools/compute_workspace.py``) so the ball and basket are
always somewhere the chosen arm can handle. MuJoCo's midphase is disabled
because its collision trees are fixed at compile time and the table, ball and
basket are resized at run time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import mujoco
import numpy as np

from ..config import MODEL_XML, WORKSPACE_MAPS, Config, CurriculumStage
from .controller import FINGERS_DOWN_SOFT, READY, OpenArmIK, tilt_from_vertical_deg, yaw_quat

SIDES = ("left", "right")

# Nominal geometry (the XML's values)
TABLE_Z = 0.25
BALL_RADIUS = 0.022
BOX_DEPTH = 0.060
BOX_HALF_SIZE = 0.045
BOX_WALL_T = 0.004               # basket wall half-thickness [m]
BOX_BOTTOM_T = 0.004             # basket floor half-thickness [m]
START_CLEARANCE = 0.03           # hands start at least this far above the table top [m]
N_DISTRACTORS = 3                # distractor bodies declared in the XML
CLEARANCE_MAX = 0.10             # basket clearance is reported up to this distance [m]

# Physics randomisation at full strength (scaled by the stage's physics level)
DR_BALL_MASS = (0.7, 1.3)
DR_BALL_FRICTION = (0.8, 1.2)
DR_PAD_FRICTION = (0.8, 1.2)
DR_ACTUATOR_KP = (0.8, 1.2)
DR_ACTUATOR_KD = (0.7, 1.3)
DR_JOINT_DAMPING = (0.7, 1.5)
DR_BOX_JITTER = 0.005            # +/- [m]
DR_MAX_LATENCY_TICKS = 2         # control ticks between command and execution

# Visual randomisation at full strength (scaled by the stage's visual level)
VIS_LIGHT_SCALE = (0.6, 1.3)
VIS_LIGHT_DIR_JITTER = 0.3       # [rad]
VIS_CAM_POS_JITTER = 0.005       # [m]   camera calibration error
VIS_CAM_ROT_JITTER_DEG = 0.5     # [deg] (0.5 deg ~ 5 mm at the table)
VIS_COLOUR_JITTER = 0.08         # per RGB channel


@dataclass
class Scenario:
    """The world of one episode."""
    table_z: float = TABLE_Z
    table_dx: float = 0.0
    table_dy: float = 0.0
    ball_radius: float = BALL_RADIUS
    box_half: float = BOX_HALF_SIZE
    box_depth: float = BOX_DEPTH
    table_friction_scale: float = 1.0
    box_friction_scale: float = 1.0
    ball_solref_scale: float = 1.0
    target_distance: float = 0.0          # drawn ball-to-basket distance [m] (0 = unset)
    n_distractors: int = 0
    push_prob: float = 0.0
    rolling: bool = False
    encoder_pos_noise: float = 0.0        # [rad]
    encoder_vel_noise: float = 0.0        # [rad/s]
    physics_level: float = 0.0
    visual_level: float = 0.0
    sensor_level: float = 0.0

    @property
    def box_rim_z(self) -> float:
        return self.table_z + self.box_depth


def worst_case_scenario(table_z: float, cfg: Config | None = None) -> Scenario:
    """Largest ball and basket the ranges allow, at a given table height
    (reachability maps are computed for this, so smaller objects also fit)."""
    s = (cfg or Config()).scene
    return Scenario(table_z=table_z,
                    ball_radius=BALL_RADIUS * s.ball_radius_scale[1],
                    box_half=BOX_HALF_SIZE * s.box_half_scale[1],
                    box_depth=BOX_DEPTH * s.box_depth_scale[1])


def _hull_vertices(verts: np.ndarray) -> np.ndarray:
    """Convex-hull vertices only (MuJoCo collides meshes by their hull)."""
    try:
        from scipy.spatial import ConvexHull
        return verts[ConvexHull(verts).vertices]
    except Exception:
        return verts


def _random_small_rotation(rng, max_angle_rad: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    ang = rng.uniform(0.0, max_angle_rad)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, axis, ang)
    return q


class Scene:
    """Owns the MuJoCo model/data and everything that edits or measures them."""

    def __init__(self, cfg: Config | None = None, model_path=MODEL_XML):
        self.cfg = cfg or Config()
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        # Resized geoms would otherwise stop colliding (midphase trees are static).
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
        self.ik = OpenArmIK(self.model)
        self.sc = Scenario()
        self._setup_indices()
        self.placements = self._load_placements()
        self.table_z_range = ((float(self.placements["heights"][0]), float(self.placements["heights"][-1]))
                              if self.placements is not None else (TABLE_Z, TABLE_Z))
        self.cells = None
        self._build_camera_projector()
        self.home_cloud = self._build_home_cloud()
        self._home_low_z = float(self.home_cloud[:, 2].min())
        self._start_cache = {}

    # =====================================================================
    # Index lookups (once)
    # =====================================================================
    def _setup_indices(self):
        m = self.model
        name = lambda obj, i: mujoco.mj_id2name(m, obj, i) or ""   # noqa: E731
        self.qpos_adr = self.ik.qpos_adr
        self.dof_adr = self.ik.dof_adr
        self.site_id = self.ik.site_id
        self.actuators = {s: np.array([m.actuator(f"openarm_{s}_joint{i}").id for i in range(1, 8)])
                          for s in SIDES}
        self.finger_act = {s: m.actuator(f"openarm_{s}_finger_joint1").id for s in SIDES}
        self.finger_qpos = {s: m.jnt_qposadr[m.joint(f"openarm_{s}_finger_joint1").id] for s in SIDES}
        self.finger_sign = {"left": 1.0, "right": -1.0}     # right finger angles are negative
        self.finger_dofs = {s: np.array([m.jnt_dofadr[m.joint(f"openarm_{s}_finger_joint{k}").id]
                                         for k in (1, 2)]) for s in SIDES}

        ball_bid = m.body("ball").id
        self.ball_body = ball_bid
        self.ball_qpos = m.jnt_qposadr[m.body_jntadr[ball_bid]]
        self.ball_dof = m.jnt_dofadr[m.body_jntadr[ball_bid]]
        self.ball_geom = m.geom("ball_geom").id
        box_bid = m.body("box").id
        self.box_body = box_bid
        self.box_qpos = m.jnt_qposadr[m.body_jntadr[box_bid]]
        self.box_dof = m.jnt_dofadr[m.body_jntadr[box_bid]]
        self.table_geom = m.geom("workspace_table").id
        self.box_parts = {n: m.geom(n).id for n in
                          ("box_bottom", "box_wall_f", "box_wall_b", "box_wall_l", "box_wall_r")}
        self.distractor_bodies = [m.body(f"distractor_{k}").id for k in range(N_DISTRACTORS)]
        self.distractor_geoms = [m.geom(f"distractor_{k}_geom").id for k in range(N_DISTRACTORS)]

        arm_words = ("openarm", "finger", "ee_")
        self.box_geom_mask = np.zeros(m.ngeom, dtype=bool)
        self.arm_geom_mask = {s: np.zeros(m.ngeom, dtype=bool) for s in SIDES}
        for i in range(m.ngeom):
            g = name(mujoco.mjtObj.mjOBJ_GEOM, i)
            if "box" in g:
                self.box_geom_mask[i] = True
            for s in SIDES:
                if s in g and any(k in g for k in arm_words):
                    self.arm_geom_mask[s][i] = True
        self.distractor_mask = np.zeros(m.ngeom, dtype=bool)
        self.distractor_mask[self.distractor_geoms] = True
        self.finger_geoms = {s: tuple(m.geom(f"openarm_{s}_ee_link{k}_collision").id for k in (1, 2))
                             for s in SIDES}
        self.pad_geoms = np.array([g for ids in self.finger_geoms.values() for g in ids], dtype=int)
        self.dr_actuators = np.concatenate([self.actuators["left"], self.actuators["right"],
                                            [self.finger_act["left"], self.finger_act["right"]]]).astype(int)
        self.dr_dofs = np.concatenate([np.concatenate([self.dof_adr[s], self.finger_dofs[s]])
                                       for s in SIDES]).astype(int)
        self.cam_id = m.camera(self.cfg.camera.camera).id
        self.colour_geoms = {
            "ball": [self.ball_geom],
            "box": list(self.box_parts.values()),
            "table": [self.table_geom],
        }

        # Collision geometry for the true arm-to-basket clearance (used by the
        # workspace tool): arm collision-mesh hull vertices vs the basket boxes.
        collidable = (m.geom_contype > 0) | (m.geom_conaffinity > 0)
        self.box_clearance_geoms = np.array(
            sorted(i for i in np.flatnonzero(self.box_geom_mask) if collidable[i]), dtype=int)
        self.arm_clearance = {}
        for s in SIDES:
            entries = []
            for gid in sorted(i for i in np.flatnonzero(self.arm_geom_mask[s]) if collidable[i]):
                if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = m.geom_dataid[gid]
                    v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                    verts = _hull_vertices(m.mesh_vert[v0:v0 + nv].reshape(-1, 3).astype(np.float64))
                else:
                    verts = None
                entries.append((gid, float(m.geom_rbound[gid]), verts))
            self.arm_clearance[s] = entries
        self._fromto = np.zeros(6)

        self.nominal = {
            "actuator_gainprm": m.actuator_gainprm.copy(),
            "actuator_biasprm": m.actuator_biasprm.copy(),
            "dof_damping": m.dof_damping.copy(),
            "geom_friction": m.geom_friction.copy(),
            "body_mass": m.body_mass.copy(),
            "body_inertia": m.body_inertia.copy(),
            "geom_pos": m.geom_pos.copy(),
            "geom_size": m.geom_size.copy(),
            "geom_rbound": m.geom_rbound.copy(),
            "geom_aabb": m.geom_aabb.copy(),
            "geom_solref": m.geom_solref.copy(),
            "geom_rgba": m.geom_rgba.copy(),
            "body_pos": m.body_pos.copy(),
            "light_diffuse": m.light_diffuse.copy(),
            "light_dir": m.light_dir.copy(),
            "cam_pos": m.cam_pos.copy(),
            "cam_quat": m.cam_quat.copy(),
            "headlight_diffuse": np.array(m.vis.headlight.diffuse),
            "headlight_ambient": np.array(m.vis.headlight.ambient),
        }
        self._refresh_box_cache()

    def _refresh_box_cache(self):
        m = self.model
        self.box_local_centres = m.geom_pos[self.box_clearance_geoms].copy()
        self.box_half_sizes = m.geom_size[self.box_clearance_geoms].copy()
        self.box_bound_radius = float((np.linalg.norm(self.box_local_centres, axis=1)
                                       + np.linalg.norm(self.box_half_sizes, axis=1)).max())

    # =====================================================================
    # Episode generation
    # =====================================================================
    def reset(self, rng: np.random.Generator, stage: CurriculumStage,
              overrides: dict | None = None) -> dict:
        """Build a new world. Returns a dict describing it (ground truth).

        The robot is not told any of this: it has to find the objects with
        its camera.
        """
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self.restore_nominal()
        self.sc = self.sample_scenario(rng, stage, overrides)
        self.apply_scenario(self.sc)

        q_left, q_right, self.home_cloud = self.start_pose(self.sc.table_z)
        self.q_start = {"left": q_left, "right": q_right}
        for s in SIDES:
            d.qpos[self.qpos_adr[s]] = self.q_start[s]
            d.ctrl[self.actuators[s]] = self.q_start[s]
            self.set_gripper(s, 1.0, teleport=True)
        self.prepare_cells(self.sc)

        side, ball, box, yaw = self.sample_layout(rng, self.sc, stage)
        self.place_distractors(rng, side, ball, box)
        self.relocate_objects(ball, box)

        latency = 0
        if self.sc.physics_level > 0:
            latency = self.apply_physics_dr(rng, self.sc.physics_level)
        if self.sc.visual_level > 0:
            self.apply_visual_dr(rng, self.sc.visual_level)

        for _ in range(20):                      # settle
            mujoco.mj_step(m, d)
        if self.sc.rolling:
            self.set_ball_rolling(rng, side, box)

        return {"side": side, "ball_pos": ball.copy(), "box_pos": box.copy(), "grasp_yaw": yaw,
                "latency": latency, "scenario": asdict(self.sc),
                "distance": float(np.linalg.norm(ball[:2] - box[:2]))}

    def sample_scenario(self, rng, stage: CurriculumStage, overrides=None) -> Scenario:
        s = self.cfg.scene
        sc = Scenario()
        lo, hi = self.table_z_range
        sc.table_z = float(np.clip(s.table_z + rng.uniform(-1, 1) * s.table_z_halfrange, lo, hi))
        sc.table_dx, sc.table_dy = rng.uniform(-s.table_shift, s.table_shift, size=2).tolist()
        sc.ball_radius = BALL_RADIUS * rng.uniform(*s.ball_radius_scale)
        sc.box_half = BOX_HALF_SIZE * rng.uniform(*s.box_half_scale)
        sc.box_depth = BOX_DEPTH * rng.uniform(*s.box_depth_scale)
        sc.table_friction_scale = rng.uniform(*s.table_friction_scale)
        sc.box_friction_scale = rng.uniform(*s.box_friction_scale)
        sc.ball_solref_scale = rng.uniform(*s.ball_solref_scale)
        sc.n_distractors = int(rng.integers(0, stage.max_distractors + 1))
        sc.push_prob = stage.push_prob
        sc.rolling = bool(rng.random() < stage.roll_prob)
        sc.encoder_pos_noise = 0.002 * stage.sensor_level
        sc.encoder_vel_noise = 0.02 * stage.sensor_level
        sc.physics_level = stage.physics_level
        sc.visual_level = stage.visual_level
        sc.sensor_level = stage.sensor_level
        if overrides:
            valid = {f.name for f in fields(Scenario)}
            unknown = set(overrides) - valid
            if unknown:
                raise ValueError(f"Unknown scenario field(s) {sorted(unknown)}; valid: {sorted(valid)}")
            for k, v in overrides.items():
                setattr(sc, k, type(getattr(sc, k))(v))
        return sc

    @staticmethod
    def _scale(rng, rng_range, level, size=None):
        """1 + level * (uniform(range) - 1): level 0 = nominal, 1 = full range."""
        return 1.0 + level * (rng.uniform(*rng_range, size=size) - 1.0)

    def _set_geom_box(self, gid, pos=None, half=None):
        """Edit a box geom AND the bounds MuJoCo's collision pre-filter reads."""
        m = self.model
        if pos is not None:
            m.geom_pos[gid] = pos
        if half is not None:
            m.geom_size[gid] = half
            m.geom_rbound[gid] = float(np.linalg.norm(half))
            m.geom_aabb[gid] = [0.0, 0.0, 0.0, *half]

    def apply_scenario(self, sc: Scenario):
        """Edit the nominal model to match ``sc`` (call after restore_nominal)."""
        m, nom = self.model, self.nominal
        tab = self.table_geom
        npos, nsize = nom["geom_pos"][tab], nom["geom_size"][tab]
        self._set_geom_box(tab, pos=[npos[0] + sc.table_dx, npos[1] + sc.table_dy, sc.table_z / 2],
                           half=[nsize[0], nsize[1], sc.table_z / 2])
        m.geom_friction[tab, 0] = nom["geom_friction"][tab, 0] * sc.table_friction_scale

        b, r = self.ball_geom, sc.ball_radius
        m.geom_size[b, 0] = r
        m.geom_rbound[b] = r
        m.geom_aabb[b] = [0.0, 0.0, 0.0, r, r, r]
        ratio = r / BALL_RADIUS
        m.body_mass[self.ball_body] = nom["body_mass"][self.ball_body] * ratio ** 3
        m.body_inertia[self.ball_body] = nom["body_inertia"][self.ball_body] * ratio ** 5
        m.geom_solref[b, 0] = nom["geom_solref"][b, 0] * sc.ball_solref_scale

        W, D, t, f = sc.box_half, sc.box_depth, BOX_WALL_T, BOX_BOTTOM_T
        p = self.box_parts
        self._set_geom_box(p["box_bottom"], pos=[0, 0, f], half=[W, W, f])
        self._set_geom_box(p["box_wall_f"], pos=[W - t, 0, D / 2], half=[t, W, D / 2])
        self._set_geom_box(p["box_wall_b"], pos=[-(W - t), 0, D / 2], half=[t, W, D / 2])
        self._set_geom_box(p["box_wall_l"], pos=[0, W - t, D / 2], half=[W - 2 * t, t, D / 2])
        self._set_geom_box(p["box_wall_r"], pos=[0, -(W - t), D / 2], half=[W - 2 * t, t, D / 2])
        for gid in p.values():
            m.geom_friction[gid, 0] = nom["geom_friction"][gid, 0] * sc.box_friction_scale
        self._refresh_box_cache()

    def restore_nominal(self):
        """Every randomised model parameter back to its nominal value."""
        m, nom = self.model, self.nominal
        for key in ("actuator_gainprm", "actuator_biasprm", "dof_damping", "geom_friction", "body_mass",
                    "body_inertia", "geom_pos", "geom_size", "geom_rbound", "geom_aabb", "geom_solref",
                    "geom_rgba", "body_pos", "light_diffuse", "light_dir", "cam_pos", "cam_quat"):
            getattr(m, key)[:] = nom[key]
        m.vis.headlight.diffuse[:] = nom["headlight_diffuse"]
        m.vis.headlight.ambient[:] = nom["headlight_ambient"]
        self._refresh_box_cache()

    def apply_physics_dr(self, rng, level: float) -> int:
        """Mass, friction, servo gains, damping, basket jitter; returns the latency in ticks."""
        m, nom = self.model, self.nominal
        sc = self._scale
        mass_scale = sc(rng, DR_BALL_MASS, level)          # on top of the scenario's size scaling
        m.body_mass[self.ball_body] *= mass_scale
        m.body_inertia[self.ball_body] *= mass_scale
        m.geom_friction[self.ball_geom] = nom["geom_friction"][self.ball_geom] * sc(rng, DR_BALL_FRICTION, level, 3)
        for g in self.pad_geoms:
            m.geom_friction[g] = nom["geom_friction"][g] * sc(rng, DR_PAD_FRICTION, level)
        acts = self.dr_actuators
        kp = sc(rng, DR_ACTUATOR_KP, level, len(acts))
        kd = sc(rng, DR_ACTUATOR_KD, level, len(acts))
        m.actuator_gainprm[acts, 0] = nom["actuator_gainprm"][acts, 0] * kp
        m.actuator_biasprm[acts, 1] = nom["actuator_biasprm"][acts, 1] * kp
        m.actuator_biasprm[acts, 2] = nom["actuator_biasprm"][acts, 2] * kd
        dofs = self.dr_dofs
        m.dof_damping[dofs] = nom["dof_damping"][dofs] * sc(rng, DR_JOINT_DAMPING, level, len(dofs))
        self.data.qpos[self.box_qpos:self.box_qpos + 2] += level * rng.uniform(-DR_BOX_JITTER, DR_BOX_JITTER, 2)
        mujoco.mj_forward(m, self.data)
        return int(rng.integers(0, int(round(DR_MAX_LATENCY_TICKS * level)) + 1))

    def apply_visual_dr(self, rng, level: float):
        """Lighting, camera pose (calibration error) and object colours."""
        m, nom = self.model, self.nominal
        for i in range(m.nlight):
            m.light_diffuse[i] = nom["light_diffuse"][i] * self._scale(rng, VIS_LIGHT_SCALE, level)
            dirv = nom["light_dir"][i] + rng.normal(0, VIS_LIGHT_DIR_JITTER * level, 3)
            m.light_dir[i] = dirv / (np.linalg.norm(dirv) + 1e-12)
        m.vis.headlight.diffuse[:] = nom["headlight_diffuse"] * self._scale(rng, VIS_LIGHT_SCALE, level)
        m.vis.headlight.ambient[:] = nom["headlight_ambient"] * self._scale(rng, VIS_LIGHT_SCALE, level)
        c = self.cam_id
        m.cam_pos[c] = nom["cam_pos"][c] + rng.uniform(-1, 1, 3) * VIS_CAM_POS_JITTER * level
        dq = _random_small_rotation(rng, np.radians(VIS_CAM_ROT_JITTER_DEG) * level)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, nom["cam_quat"][c], dq)
        m.cam_quat[c] = q
        for gids in self.colour_geoms.values():
            shift = rng.uniform(-1, 1, 3) * VIS_COLOUR_JITTER * level     # one shift per object
            for g in gids:
                m.geom_rgba[g, :3] = np.clip(nom["geom_rgba"][g, :3] + shift, 0.0, 1.0)

    # =====================================================================
    # Start pose
    # =====================================================================
    def _build_home_cloud(self, q_left=None, q_right=None) -> np.ndarray:
        """World points on both arms' collision geometry at a start pose (ready pose by default)."""
        m = self.model
        tmp = mujoco.MjData(m)
        tmp.qpos[self.qpos_adr["left"]] = READY["left"] if q_left is None else q_left
        tmp.qpos[self.qpos_adr["right"]] = READY["right"] if q_right is None else q_right
        ctrl = self.cfg.control
        tmp.qpos[self.finger_qpos["left"]] = ctrl.gripper_open
        tmp.qpos[self.finger_qpos["right"]] = -ctrl.gripper_open
        mujoco.mj_kinematics(m, tmp)
        pts = []
        for entries in self.arm_clearance.values():
            for gid, _, verts in entries:
                if verts is not None:
                    pts.append(verts @ tmp.geom_xmat[gid].reshape(3, 3).T + tmp.geom_xpos[gid])
        return np.concatenate(pts)

    def start_pose(self, table_z: float):
        """The ready pose: hands raised beside the camera, clear of any table
        (checked against ``table_z + START_CLEARANCE``)."""
        if "ready" not in self._start_cache:
            self._start_cache["ready"] = (READY["left"].copy(), READY["right"].copy(), self._build_home_cloud())
        q_left, q_right, cloud = self._start_cache["ready"]
        if cloud[:, 2].min() < table_z + START_CLEARANCE:
            raise RuntimeError(f"Ready pose is too low for a {table_z:.2f} m table")
        return q_left.copy(), q_right.copy(), cloud

    def _blocked_by_home_hands(self, xy, half, top_z, margin=0.03):
        pts = self.home_cloud
        low = pts[pts[:, 2] < top_z + margin]
        if len(low) == 0:
            return np.zeros(len(xy), dtype=bool)
        near = np.abs(xy[:, None, :] - low[None, :, :2]) < (half + margin)
        return (near[..., 0] & near[..., 1]).any(axis=1)


    # =====================================================================
    # Where the head camera can see (objects are only placed in view)
    # =====================================================================
    def _build_camera_projector(self):
        m, c, cam = self.model, self.cam_id, self.cfg.camera
        b = m.cam_bodyid[c]
        assert m.body_parentid[b] == 0, "camera mount is expected to be fixed to the world"
        Rb, Rc = np.zeros(9), np.zeros(9)
        mujoco.mju_quat2Mat(Rb, m.body_quat[b])
        mujoco.mju_quat2Mat(Rc, self.nominal["cam_quat"][c])
        Rb, Rc = Rb.reshape(3, 3), Rc.reshape(3, 3)
        self._cam_pos = m.body_pos[b] + Rb @ self.nominal["cam_pos"][c]
        self._cam_R = Rb @ Rc
        self._cam_f = (cam.height / 2.0) / np.tan(np.radians(m.cam_fovy[c]) / 2.0)

    def camera_visible(self, points: np.ndarray, margin_px: float | None = None) -> np.ndarray:
        """Which world points (n, 3) project inside the head camera image."""
        cam = self.cfg.camera
        margin = cam.visibility_margin_px if margin_px is None else margin_px
        pc = (np.atleast_2d(points) - self._cam_pos) @ self._cam_R
        depth = -pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = cam.width / 2.0 + self._cam_f * pc[:, 0] / depth
            v = cam.height / 2.0 - self._cam_f * pc[:, 1] / depth
        return ((depth > 0.05) & (u >= margin) & (u <= cam.width - margin)
                & (v >= margin) & (v <= cam.height - margin))

    # =====================================================================
    # Placement from the reachability maps
    # =====================================================================
    def _load_placements(self):
        if not WORKSPACE_MAPS.exists():
            return None
        maps = np.load(WORKSPACE_MAPS)
        xs, ys, res = maps["xs"], maps["ys"], float(maps["grid_res"])
        heights = np.atleast_1d(maps["heights"]).astype(float)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")

        def interior(mask):
            """Cells whose 8 neighbours are reachable too (jitter stays reachable)."""
            core = mask.copy()
            core[0, :] = core[-1, :] = core[:, 0] = core[:, -1] = False
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    core &= np.roll(np.roll(mask, di, axis=0), dj, axis=1)
            return core

        def per_height(arr):
            arr = np.asarray(arr)
            return arr[None] if arr.ndim == 2 else arr

        out = {"res": res, "heights": heights, "gx": gx, "gy": gy, "ball_ok": {}, "ball_yaw": {}, "box_ok": {}}
        for s in SIDES:
            yaw = per_height(maps[f"{s}_ball_yaw"])
            box = per_height(maps[f"{s}_box_ok"])
            if f"{s}_box_tilt" in maps.files:
                tilt = per_height(maps[f"{s}_box_tilt"])
                box = box & (np.nan_to_num(tilt, nan=180.0) <= self.cfg.scene.max_carry_tilt_deg)
            out["ball_yaw"][s] = yaw
            out["ball_ok"][s] = np.stack([interior(np.isfinite(y)) for y in yaw])
            out["box_ok"][s] = np.stack([interior(b) for b in box])
        return out

    def prepare_cells(self, sc: Scenario):
        """Candidate ball/basket cells per arm for this episode's table and objects.

        A ball cell belongs to the arm on its side of the robot (beyond the
        midline band), so the camera's choice of arm is unambiguous; the
        basket stays on the same side.
        """
        P = self.placements
        if P is None:
            raise RuntimeError(f"Reachability maps not found at {WORKSPACE_MAPS}; "
                               "run `python main.py maps` to build them.")
        hs, res, gx, gy = P["heights"], P["res"], P["gx"], P["gy"]
        z = float(np.clip(sc.table_z, hs[0], hs[-1]))
        hi = min(int(np.searchsorted(hs, z)), len(hs) - 1)
        lo = hi if hs[hi] <= z + 1e-9 else max(hi - 1, 0)
        near = lo if abs(hs[lo] - z) <= abs(hs[hi] - z) else hi
        m = self.model
        cx, cy = m.geom_pos[self.table_geom][:2]
        hx, hy = m.geom_size[self.table_geom][:2]
        band = self.cfg.scene.midline_band

        def on_table(half):
            return (np.abs(gx - cx) <= hx - half - 0.01) & (np.abs(gy - cy) <= hy - half - 0.01)

        cells = {}
        for s in SIDES:
            sign = 1.0 if s == "left" else -1.0
            ball = (P["ball_ok"][s][lo] & P["ball_ok"][s][hi] & on_table(sc.ball_radius + res / 2)
                    & (sign * gy >= band))
            box = (P["box_ok"][s][lo] & P["box_ok"][s][hi] & on_table(sc.box_half + res / 2)
                   & (sign * gy >= 0.0))
            yaw = P["ball_yaw"][s][near]
            ball_xy = np.stack([gx[ball], gy[ball]], axis=1)
            box_xy = np.stack([gx[box], gy[box]], axis=1)
            keep_b = ~self._blocked_by_home_hands(ball_xy, sc.ball_radius + res / 2,
                                                  sc.table_z + 2 * sc.ball_radius)
            keep_x = ~self._blocked_by_home_hands(box_xy, sc.box_half + res / 2, sc.box_rim_z)
            # ...and inside the head camera's view: the ball centre well inside
            # the image, every corner of the basket inside it
            keep_b &= self.camera_visible(np.column_stack(
                [ball_xy, np.full(len(ball_xy), sc.table_z + sc.ball_radius)]))
            h = sc.box_half + res / 2
            for dx, dy in ((h, h), (h, -h), (-h, h), (-h, -h)):
                for z in (sc.table_z, sc.box_rim_z):
                    keep_x &= self.camera_visible(np.column_stack(
                        [box_xy[:, 0] + dx, box_xy[:, 1] + dy, np.full(len(box_xy), z)]), margin_px=2)
            cells[s] = {"ball_xy": ball_xy[keep_b], "ball_yaw": yaw[ball][keep_b], "box_xy": box_xy[keep_x]}
            if len(cells[s]["ball_xy"]) == 0 or len(cells[s]["box_xy"]) == 0:
                raise RuntimeError(f"No reachable ball/basket cells for the {s} arm at table height "
                                   f"{sc.table_z:.3f} m; rebuild the maps (`python main.py maps`).")
        self.cells = cells

    def sample_layout(self, rng, sc: Scenario, stage: CurriculumStage):
        """Ball side, ball and basket positions with a freshly drawn distance."""
        side = SIDES[int(rng.integers(2))]
        pl, half = self.cells[side], self.placements["res"] / 2.0
        D = np.linalg.norm(pl["ball_xy"][:, None, :] - pl["box_xy"][None, :, :], axis=2)
        s = self.cfg.scene
        d_hi = min(stage.max_distance, float(D.max()))
        d_lo = min(s.min_distance, d_hi)
        target = float(rng.uniform(d_lo, d_hi)) if sc.target_distance <= 0 else sc.target_distance
        sc.target_distance = target
        ok = (np.abs(D - target) <= s.distance_tolerance) & (D >= s.min_distance)
        ii, jj = np.nonzero(ok)
        if ii.size == 0:                                   # nearest achievable distance
            err = np.abs(D - target) + 1e3 * (D < s.min_distance)
            i, j = np.unravel_index(int(np.argmin(err)), D.shape)
        else:
            k = int(rng.integers(ii.size))
            i, j = int(ii[k]), int(jj[k])
        ball_xy = pl["ball_xy"][i] + rng.uniform(-half, half, size=2)
        box_xy = pl["box_xy"][j] + rng.uniform(-half, half, size=2)
        ball = np.array([ball_xy[0], ball_xy[1], sc.table_z + sc.ball_radius])
        box = np.array([box_xy[0], box_xy[1], sc.table_z])
        return side, ball, box, float(pl["ball_yaw"][i])

    def grasp_yaw_at(self, side: str, xy) -> float | None:
        """Hand yaw for a vertical grasp at ``xy``; None if the arm can't grasp there.

        This is the robot's own knowledge of its workspace (the maps), used by
        the scripted planner.
        """
        pl = self.cells[side]
        dist = np.linalg.norm(pl["ball_xy"] - np.asarray(xy)[:2], axis=1)
        i = int(np.argmin(dist))
        return float(pl["ball_yaw"][i]) if dist[i] <= self.placements["res"] * 1.5 else None

    def place_distractors(self, rng, side, ball_pos, box_pos):
        """Distractor blocks on the table, clear of the ball, basket and carry path."""
        m, sc = self.model, self.sc
        placed = []
        seg = box_pos[:2] - ball_pos[:2]
        seg_len2 = float(seg @ seg) + 1e-12
        sign = 1.0 if side == "left" else -1.0
        tab = self.table_geom
        cx, cy = m.geom_pos[tab][:2]
        hx, hy = m.geom_size[tab][:2]
        for k in range(N_DISTRACTORS):
            b, g = self.distractor_bodies[k], self.distractor_geoms[k]
            m.body_pos[b] = [0.0, 0.0, -5.0]                       # parked under the floor
            if k >= sc.n_distractors:
                continue
            half = rng.uniform(0.015, 0.03, size=3)
            for _ in range(80):
                xy = np.array([rng.uniform(0.14, 0.50), sign * rng.uniform(0.05, 0.45)])
                t = np.clip(((xy - ball_pos[:2]) @ seg) / seg_len2, 0.0, 1.0)
                path_dist = np.linalg.norm(xy - (ball_pos[:2] + t * seg))
                top = sc.table_z + 2 * half[2]
                if (not self._blocked_by_home_hands(xy[None], float(half[:2].max()), top)[0]
                        and np.linalg.norm(xy - ball_pos[:2]) > 0.10
                        and np.linalg.norm(xy - box_pos[:2]) > 0.12
                        and path_dist > 0.09
                        and all(np.linalg.norm(xy - p) > 0.10 for p in placed)
                        and abs(xy[0] - cx) < hx - 0.04 and abs(xy[1] - cy) < hy - 0.04):
                    self._set_geom_box(g, pos=[0, 0, 0], half=half)
                    m.body_pos[b] = [xy[0], xy[1], sc.table_z + half[2]]
                    placed.append(xy)
                    break

    def relocate_objects(self, ball_pos, box_pos):
        d = self.data
        d.qpos[self.ball_qpos:self.ball_qpos + 3] = ball_pos
        d.qpos[self.ball_qpos + 3:self.ball_qpos + 7] = [1, 0, 0, 0]
        d.qvel[self.ball_dof:self.ball_dof + 6] = 0.0
        d.qpos[self.box_qpos:self.box_qpos + 3] = box_pos
        d.qpos[self.box_qpos + 3:self.box_qpos + 7] = [1, 0, 0, 0]
        d.qvel[self.box_dof:self.box_dof + 6] = 0.0
        mujoco.mj_forward(self.model, d)

    def set_ball_rolling(self, rng, side, box_pos):
        """Start the ball rolling toward another graspable cell on its side."""
        d, sc = self.data, self.sc
        ball = d.xpos[self.ball_body].copy()
        cand = self.cells[side]["ball_xy"]
        dist = np.linalg.norm(cand - ball[:2], axis=1)
        clear = np.linalg.norm(cand - box_pos[:2], axis=1) > sc.box_half + sc.ball_radius + 0.05
        lo, hi = self.cfg.scene.roll_distance
        ok = np.flatnonzero((dist > lo) & (dist < hi) & clear)
        if ok.size == 0:
            sc.rolling = False
            return
        goal = cand[int(rng.choice(ok))]
        travel = np.linalg.norm(goal - ball[:2])
        direction = (goal - ball[:2]) / (travel + 1e-12)
        # measured on this table (condim 6, rolling friction 0.001): distance ~= 2.8 v^2.2
        speed = (travel / 2.8) ** (1.0 / 2.2)
        v = np.array([direction[0], direction[1], 0.0]) * speed
        d.qvel[self.ball_dof:self.ball_dof + 3] = v
        d.qvel[self.ball_dof + 3:self.ball_dof + 6] = np.cross([0, 0, 1.0], v) / sc.ball_radius

    def maybe_push_ball(self, rng):
        """Occasionally nudge a ball that is resting on the table."""
        sc, d = self.sc, self.data
        if sc.push_prob <= 0:
            return False
        adr = self.ball_dof
        resting = (np.linalg.norm(d.qvel[adr:adr + 3]) < 0.02
                   and d.xpos[self.ball_body][2] < sc.table_z + sc.ball_radius + 0.01)
        if resting and rng.random() < sc.push_prob:
            speed = rng.uniform(*self.cfg.scene.push_speed)
            ang = rng.uniform(0.0, 2.0 * np.pi)
            d.qvel[adr:adr + 2] += speed * np.array([np.cos(ang), np.sin(ang)])
            return True
        return False

    def place_ball_in_hand(self, side: str, pinch_target: np.ndarray, rng) -> bool:
        """Initial state for PLACE-start episodes: the hand holds the ball at
        ``pinch_target``. This sets up a starting configuration only; the
        policy is never shown how to grasp. Returns False if the ball does
        not end up held (the caller then starts a normal episode).
        """
        m, d, ctrl = self.model, self.data, self.cfg.control
        res = None
        sign = 1.0 if side == "left" else -1.0
        for yaw in (-90, -60, -120, -30, -150, 0, 180, 30, 150, 60, 120, 90):   # fingers straight down
            r = self.ik.solve(side, pinch_target, yaw_quat(sign * yaw))
            if r.pos_error < 0.002 and r.rot_error < 0.05:
                res = r
                break
        if res is None:                                  # vertical not reachable: least tilt
            r = self.ik.solve(side, pinch_target, FINGERS_DOWN_SOFT)
            if r.pos_error > 0.003 or tilt_from_vertical_deg(self.ik.data.site_xmat[self.site_id[side]]) > 60:
                return False
            res = r
        d.qpos[self.qpos_adr[side]] = res.q_arm
        d.ctrl[self.actuators[side]] = res.q_arm
        d.qvel[:] = 0.0
        self.set_gripper(side, 1.0, teleport=True)
        mujoco.mj_forward(m, d)
        self.set_gripper(side, 0.0)                         # close on the ball
        for _ in range(150):                                # hold the ball still while the fingers close
            d.qpos[self.ball_qpos:self.ball_qpos + 3] = d.site_xpos[self.site_id[side]]
            d.qvel[self.ball_dof:self.ball_dof + 6] = 0.0
            mujoco.mj_step(m, d)
        for _ in range(100):                                # let go of the ball: does the grip hold it?
            mujoco.mj_step(m, d)
        held = np.linalg.norm(d.xpos[self.ball_body] - d.site_xpos[self.site_id[side]]) < 0.02
        return bool(held and all(self.finger_ball_contacts(side)))

    # =====================================================================
    # Actuation helpers
    # =====================================================================
    def set_gripper(self, side: str, open_ratio: float, teleport: bool = False):
        """Command the gripper through its position servo (teleport only at reset)."""
        c = self.cfg.control
        val = self.finger_sign[side] * (c.gripper_closed + float(np.clip(open_ratio, 0, 1))
                                        * (c.gripper_open - c.gripper_closed))
        self.data.ctrl[self.finger_act[side]] = val
        if teleport:
            self.data.qpos[self.finger_qpos[side]] = val

    def step_physics(self, n: int):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    # =====================================================================
    # Measurements
    # =====================================================================
    def pinch_pos(self, side: str) -> np.ndarray:
        return self.data.site_xpos[self.site_id[side]].copy()

    def pinch_vel(self, side: str) -> np.ndarray:
        """Linear velocity of the pinch point [m/s] (from the joint encoders via kinematics)."""
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.site_id[side], res, 0)
        return res[3:].copy()

    def pinch_quat(self, side: str) -> np.ndarray:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.data.site_xmat[self.site_id[side]])
        return q

    def hand_tilt_deg(self, side: str) -> float:
        return tilt_from_vertical_deg(self.data.site_xmat[self.site_id[side]])

    def finger_angle(self, side: str) -> float:
        """Measured finger opening angle (positive for both arms) [rad]."""
        return float(self.finger_sign[side] * self.data.qpos[self.finger_qpos[side]])

    def finger_speed(self, side: str) -> float:
        """Finger joint speed [rad/s] (a stalled finger has squeezed something)."""
        return float(abs(self.data.qvel[self.finger_dofs[side][0]]))

    def gripper_force(self, side: str) -> float:
        """Gripper motor force, positive when squeezing (sign-corrected per arm)."""
        return float(-self.finger_sign[side] * self.data.actuator_force[self.finger_act[side]])

    def ball_pos(self) -> np.ndarray:
        return self.data.xpos[self.ball_body].copy()

    def ball_vel(self) -> np.ndarray:
        return self.data.qvel[self.ball_dof:self.ball_dof + 3].copy()

    def box_pos(self) -> np.ndarray:
        return self.data.xpos[self.box_body].copy()

    def _contacts(self) -> np.ndarray:
        return self.data.contact.geom[:self.data.ncon]

    def finger_ball_contacts(self, side: str) -> tuple[bool, bool]:
        if self.data.ncon == 0:
            return (False, False)
        geoms = self._contacts()
        f1, f2 = self.finger_geoms[side]
        with_ball = geoms[(geoms == self.ball_geom).any(axis=1)]
        return bool((with_ball == f1).any()), bool((with_ball == f2).any())

    def _arm_touches(self, side: str, mask: np.ndarray) -> bool:
        if self.data.ncon == 0:
            return False
        geoms = self._contacts()
        g1, g2 = geoms[:, 0], geoms[:, 1]
        arm = self.arm_geom_mask[side]
        return bool(((arm[g1] & mask[g2]) | (arm[g2] & mask[g1])).any())

    def arm_box_contact(self, side: str) -> bool:
        return self._arm_touches(side, self.box_geom_mask)

    def arm_distractor_contact(self, side: str) -> bool:
        return self.sc.n_distractors > 0 and self._arm_touches(side, self.distractor_mask)

    def box_tilt_deg(self) -> float:
        up_z = self.data.xmat[self.box_body][8]
        return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))

    def ball_in_box(self, upright_deg: float = 15.0) -> bool:
        """Ball centre inside the inner walls and below the rim of an upright
        basket that sits on the table (measured in the basket's own frame)."""
        sc, d = self.sc, self.data
        if self.box_tilt_deg() > upright_deg:
            return False
        box = d.xpos[self.box_body]
        if abs(box[2] - sc.table_z) > 0.01:
            return False
        R = d.xmat[self.box_body].reshape(3, 3)
        p = R.T @ (d.xpos[self.ball_body] - box)
        inner = sc.box_half - 2 * BOX_WALL_T
        return bool(abs(p[0]) < inner and abs(p[1]) < inner and 2 * BOX_BOTTOM_T < p[2] < sc.box_depth)

    def ball_off_table(self) -> bool:
        return bool(self.data.xpos[self.ball_body][2] < self.sc.table_z - 0.05)

    def compute_basket_clearance(self, side: str) -> float:
        """Minimum distance from the arm's collision geometry to the basket
        (capped at CLEARANCE_MAX, 0 on contact). Used by the workspace tool."""
        if self.arm_box_contact(side):
            return 0.0
        m, d = self.model, self.data
        centre = d.xpos[self.box_body]
        box_R = d.xmat[self.box_body].reshape(3, 3)
        best, near = CLEARANCE_MAX, []
        for gid, rbound, verts in self.arm_clearance[side]:
            gpos = d.geom_xpos[gid]
            if np.linalg.norm(gpos - centre) - rbound - self.box_bound_radius >= best:
                continue
            if verts is None:
                for g2 in self.box_clearance_geoms:
                    best = min(best, mujoco.mj_geomDistance(m, d, gid, g2, best, self._fromto))
                continue
            near.append(verts @ d.geom_xmat[gid].reshape(3, 3).T + gpos)
        if not near:
            return max(best, 0.0)
        pts = (np.concatenate(near) - centre) @ box_R
        excess = np.abs(pts[None] - self.box_local_centres[:, None]) - self.box_half_sizes[:, None]
        np.maximum(excess, 0.0, out=excess)
        dist = float(np.sqrt(np.einsum("bnk,bnk->bn", excess, excess).min()))
        return max(min(best, dist), 0.0)
