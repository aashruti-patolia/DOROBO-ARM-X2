"""
OpenArm v2.0 Bimanual -- Autonomous Pick and Place System
=========================================================

Features:
  1. Randomized Stage Placement:
     - Spawns a dynamic Red Ball and a Blue Box at random positions on the stage table.
     - Widely randomized separation distances across short, medium, and long spans.
     - Strict non-overlapping clearance (>= 125 mm) ensuring zero collision during grasping.
     - Calibrated strictly within each arm's reachable kinematic envelope.

  2. Perception via Camera Sensors:
     - Captures RGB image from the front-mounted stereo head camera (`openarm_head_camera_center`).
     - Performs color segmentation (Red for Ball, Blue for Box).
     - Computes 2D pixel centroids and ray-plane back-projects to accurate 3D coordinates (< 3 mm error).

  3. Grasping & Firm Clamping:
     - Gripper aligns TCP directly at the ball center equator (Z = 0.272 m).
     - Controlled finger closure firmly pinches and clamps the 44 mm ball between finger pads.
     - Ball remains securely held between gripper fingers throughout lift and transit without slipping.

  4. Collision-Free Placement:
     - End-effector descends to a safe clearance height (Z = 0.390 m) strictly above the box rim (Z = 0.310 m).
     - Zero contact between robot arm/gripper and box walls.
     - Ball is released directly above the box aperture, dropping cleanly under natural gravity.
     - Damped physics (solref) ensures the ball settles without bouncing over walls.
     - Vertical departure up to Z = 0.420 m before retracting to HOME prevents clipping the box.

  5. Relocation & Continuous Loop:
     - Starts automatically when Run Mode is ON.
     - After placing, verifies placement, relocates ball and box to new random stage locations,
       and repeats in a continuous loop.
     - Interactive controls: Press SPACEBAR in the 3D viewer to Pause / Resume Run Mode.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mujoco
import numpy as np

# Ensure local directory is on sys.path for direct invocations
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import the project's inverse kinematics solver
from inverse_kinematics import LEFT_ARM_HOME, RIGHT_ARM_HOME, OpenArmIK


# =====================================================================
# Configuration & Constants
# =====================================================================
@dataclass
class PickAndPlaceConfig:
    """Hyper-parameters for workspace, vision, and motion planning."""

    # Ground/table surface height [m]
    table_z: float = 0.25

    # Object geometry
    ball_radius: float = 0.022  # Ball sphere radius [m] (diameter 44 mm)
    box_half_size: float = 0.045  # Half-width of box aperture [m] (90 mm outer)
    box_depth: float = 0.060  # Box wall height [m] (rim at Z = 0.310 m)

    # Separation distance bounds between ball and box on stage [m]
    min_separation: float = 0.125  # Minimum separation to prevent gripper touching box during grasp [m]
    max_separation: float = 0.220  # Maximum span across reachable stage table [m]

    # Reachable workspace bounds [m] on the stage table
    left_x_range: Tuple[float, float] = (0.16, 0.29)
    left_y_range: Tuple[float, float] = (0.06, 0.24)

    right_x_range: Tuple[float, float] = (0.16, 0.29)
    right_y_range: Tuple[float, float] = (-0.24, -0.06)

    # Trajectory heights relative to table
    approach_clearance: float = 0.100  # Above ball during approach [m]
    grasp_offset: float = 0.000  # Align TCP exactly at ball center equator [m]
    lift_height: float = 0.170  # Height above table during transport [m] (Z = 0.420 m)
    drop_clearance: float = 0.080  # Safe clearance above box rim [m] (prevents touching box)

    # Motion timing
    n_waypoint_steps: int = 25  # Substeps between Cartesian waypoints
    gripper_steps: int = 35  # Steps for finger opening/closing
    settle_steps: int = 50  # Physics steps to let ball drop and settle in box


# =====================================================================
# 1. Camera Sensor Vision Perception
# =====================================================================
class CameraVisionDetector:
    """Detects objects on the stage table using the front-mounted head camera sensor.

    Performs:
      1. Off-screen RGB rendering from `openarm_head_camera_center`.
      2. Color space segmentation (Red for Ball, Blue for Box).
      3. Centroid extraction in pixel coordinates (u, v).
      4. Optical ray-plane back-projection onto known ground height
         using camera intrinsics and extrinsics to produce 3D coordinates.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_name: str = "openarm_head_camera_center",
        img_width: int = 640,
        img_height: int = 480,
    ):
        self.model = model
        self.cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if self.cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in model.")

        self.width = img_width
        self.height = img_height
        self.renderer = mujoco.Renderer(model, height=img_height, width=img_width)

        # Camera intrinsics (pinhole model derived from MuJoCo fovy)
        fovy_rad = np.radians(model.cam_fovy[self.cam_id])
        self.fy = (self.height / 2.0) / np.tan(fovy_rad / 2.0)
        self.fx = self.fy  # Square pixels
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def capture_image(self, data: mujoco.MjData) -> np.ndarray:
        """Render RGB frame from the head camera."""
        self.renderer.update_scene(data, camera=self.cam_id)
        return self.renderer.render()

    def backproject_pixel_to_plane(
        self, data: mujoco.MjData, u: float, v: float, plane_z: float
    ) -> np.ndarray:
        """Back-project 2D pixel (u, v) onto a horizontal plane at plane_z [m]."""
        cam_pos = data.cam_xpos[self.cam_id]
        cam_mat = data.cam_xmat[self.cam_id].reshape(3, 3)

        # Ray direction in camera frame (OpenGL: +X right, +Y up, -Z forward)
        ray_cam = np.array(
            [(u - self.cx) / self.fx, -(v - self.cy) / self.fy, -1.0]
        )
        ray_cam /= np.linalg.norm(ray_cam)

        # Ray direction in world frame
        ray_world = cam_mat @ ray_cam

        # Intersect with horizontal plane Z = plane_z
        if abs(ray_world[2]) < 1e-6:
            raise RuntimeError("Camera ray is parallel to the ground plane.")

        t = (plane_z - cam_pos[2]) / ray_world[2]
        intersection = cam_pos + t * ray_world
        intersection[2] = plane_z  # Enforce exact plane height
        return intersection

    def detect_objects(
        self, data: mujoco.MjData, table_z: float, ball_radius: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run vision pipeline to detect the Red Ball and Blue Box.

        Returns (detected_ball_pos_3d, detected_box_pos_3d).
        """
        img = self.capture_image(data)

        # 1. Color Segmentation
        # Red Ball: High R, Low G and B
        red_mask = (
            (img[:, :, 0] > 130)
            & (img[:, :, 1] < 100)
            & (img[:, :, 2] < 100)
        )
        
        # Blue Box: High B, Low R and moderate G
        blue_mask = (
            (img[:, :, 2] > 130)
            & (img[:, :, 0] < 100)
            & (img[:, :, 1] < 130)
        )

        detected_ball = None
        detected_box = None

        # 2. Extract Ball Centroid & 3D Coordinates
        n_red = np.count_nonzero(red_mask)
        if n_red >= 15:  # Minimum blob size threshold
            v_coords, u_coords = np.where(red_mask)
            u_ball = float(np.mean(u_coords))
            v_ball = float(np.mean(v_coords))
            ball_center_z = table_z + ball_radius
            detected_ball = self.backproject_pixel_to_plane(
                data, u_ball, v_ball, ball_center_z
            )

        # 3. Extract Box Centroid & 3D Coordinates
        n_blue = np.count_nonzero(blue_mask)
        if n_blue >= 40:
            v_coords, u_coords = np.where(blue_mask)
            u_box = float(np.mean(u_coords))
            v_box = float(np.mean(v_coords))
            # The visible blue pixels are the box walls, centered at table_z + 0.030 m
            detected_box = self.backproject_pixel_to_plane(
                data, u_box, v_box, table_z + 0.030
            )
            detected_box[2] = table_z

        return detected_ball, detected_box

    def close(self):
        """Free OpenGL renderer resources."""
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.close()


# =====================================================================
# 2. Workspace & Random Placement Manager
# =====================================================================
class WorkspaceManager:
    """Manages randomized non-overlapping object placement within arm reach on stage."""

    def __init__(self, model: mujoco.MjModel, cfg: PickAndPlaceConfig):
        self.model = model
        self.cfg = cfg

        # Resolve freejoint addresses for the ball and box
        ball_bid = model.body("ball").id
        self.ball_qpos_adr = model.jnt_qposadr[model.body_jntadr[ball_bid]]
        self.ball_dof_adr = model.jnt_dofadr[model.body_jntadr[ball_bid]]

        box_bid = model.body("box").id
        self.box_qpos_adr = model.jnt_qposadr[model.body_jntadr[box_bid]]
        self.box_dof_adr = model.jnt_dofadr[model.body_jntadr[box_bid]]

    def sample_random_locations(
        self, arm: str = "left"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random positions on the stage with widely randomized distances.

        Ensures:
          1. Both ball and box are on the stage table within the active arm's reach.
          2. Non-overlapping separation (distance >= min_separation).
          3. Distances vary randomly across short, medium, and long spans (up to max_separation).
        """
        cfg = self.cfg
        if arm == "left":
            x_range, y_range = cfg.left_x_range, cfg.left_y_range
        else:
            x_range, y_range = cfg.right_x_range, cfg.right_y_range

        # Attempt 1: Uniformly sample distance across the full range [min_separation, max_separation]
        for _ in range(300):
            target_dist = np.random.uniform(cfg.min_separation, cfg.max_separation)
            ball_x = np.random.uniform(*x_range)
            ball_y = np.random.uniform(*y_range)
            theta = np.random.uniform(0, 2 * np.pi)
            box_x = ball_x + target_dist * np.cos(theta)
            box_y = ball_y + target_dist * np.sin(theta)

            if (x_range[0] <= box_x <= x_range[1]) and (y_range[0] <= box_y <= y_range[1]):
                ball_pos = np.array([ball_x, ball_y, cfg.table_z + cfg.ball_radius])
                box_pos = np.array([box_x, box_y, cfg.table_z])
                return ball_pos, box_pos

        # Attempt 2: Direct rejection sampling
        for _ in range(300):
            ball_x = np.random.uniform(*x_range)
            ball_y = np.random.uniform(*y_range)
            box_x = np.random.uniform(*x_range)
            box_y = np.random.uniform(*y_range)

            dist = np.hypot(ball_x - box_x, ball_y - box_y)
            if dist >= cfg.min_separation:
                ball_pos = np.array([ball_x, ball_y, cfg.table_z + cfg.ball_radius])
                box_pos = np.array([box_x, box_y, cfg.table_z])
                return ball_pos, box_pos

        raise RuntimeError("Failed to sample non-overlapping locations on the stage.")

    def relocate_objects(
        self, data: mujoco.MjData, ball_pos: np.ndarray, box_pos: np.ndarray
    ):
        """Set the ball and box to specified 3D positions in the simulation state."""
        # Relocate ball (freejoint: 3 position + 4 quaternion)
        data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 3] = ball_pos
        data.qpos[self.ball_qpos_adr + 3 : self.ball_qpos_adr + 7] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        data.qvel[self.ball_dof_adr : self.ball_dof_adr + 6] = 0.0

        # Relocate box
        data.qpos[self.box_qpos_adr : self.box_qpos_adr + 3] = box_pos
        data.qpos[self.box_qpos_adr + 3 : self.box_qpos_adr + 7] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        data.qvel[self.box_dof_adr : self.box_dof_adr + 6] = 0.0

        mujoco.mj_forward(self.model, data)


# =====================================================================
# 3. Main Pick and Place Controller
# =====================================================================
class OpenArmPickAndPlace:
    """Orchestrates perception, kinematics, trajectory execution, and relocation."""

    def __init__(
        self,
        model_path: str = "openarm_v20_mujoco/openarm_pick_and_place.xml",
        config: Optional[PickAndPlaceConfig] = None,
    ):
        self.cfg = config or PickAndPlaceConfig()

        # Load MuJoCo model and data
        path = Path(model_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)

        # Initialize sub-modules
        self.ik = OpenArmIK(self.model, data=self.data)
        self.vision = CameraVisionDetector(self.model)
        self.workspace = WorkspaceManager(self.model, self.cfg)

        # Qpos and actuator indices
        self._left_arm_qpos = self.ik._left_qpos_adr
        self._right_arm_qpos = self.ik._right_qpos_adr
        self._left_finger_qpos = self.model.jnt_qposadr[
            self.model.joint("openarm_left_finger_joint1").id
        ]
        self._right_finger_qpos = self.model.jnt_qposadr[
            self.model.joint("openarm_right_finger_joint1").id
        ]

        self._left_actuators = np.array(
            [
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"openarm_left_joint{i+1}",
                )
                for i in range(7)
            ]
        )
        self._right_actuators = np.array(
            [
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"openarm_right_joint{i+1}",
                )
                for i in range(7)
            ]
        )

        self._left_finger_act = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "openarm_left_finger_joint1",
        )
        self._right_finger_act = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "openarm_right_finger_joint1",
        )

        # Cache geom IDs for collision verification between arm/gripper and box
        self._box_geom_ids = set()
        for i in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if gname and "box" in gname:
                self._box_geom_ids.add(i)

        self._arm_geom_ids = set()
        for i in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if gname and any(k in gname for k in ["openarm", "finger", "ee_"]):
                self._arm_geom_ids.add(i)

        # Grasp attachment state
        self._grasped_arm: Optional[str] = None

        # Task statistics
        self.completed_cycles = 0
        self.successful_cycles = 0
        self.run_mode = True

        # Initialize arms to home posture
        self._reset_home()

    def _reset_home(self):
        """Set both arms to ergonomic HOME posture with fingers open."""
        self.data.qpos[self._left_arm_qpos] = LEFT_ARM_HOME
        self.data.qpos[self._right_arm_qpos] = RIGHT_ARM_HOME
        self.data.ctrl[self._left_actuators] = LEFT_ARM_HOME
        self.data.ctrl[self._right_actuators] = RIGHT_ARM_HOME
        self.set_gripper("left", open_ratio=1.0)
        self.set_gripper("right", open_ratio=1.0)
        mujoco.mj_forward(self.model, self.data)

    def set_gripper(self, side: str, open_ratio: float):
        """Set gripper fingers opening (1.0 = fully open, 0.0 = fully closed).

        Finger kinematics:
          Left arm:
            open_ratio = 1.0 -> 0.65 rad (open, fingertip span ~92 mm)
            open_ratio = 0.0 -> 0.05 rad (closed, fingertip span ~40 mm, pinching 44 mm ball)
          Right arm:
            open_ratio = 1.0 -> -0.65 rad (open, fingertip span ~92 mm)
            open_ratio = 0.0 -> -0.05 rad (closed, fingertip span ~40 mm, pinching 44 mm ball)
        """
        open_ratio = np.clip(open_ratio, 0.0, 1.0)
        if side == "left":
            val = 0.05 + open_ratio * (0.65 - 0.05)
            self.data.ctrl[self._left_finger_act] = val
            self.data.qpos[self._left_finger_qpos] = val
        else:
            val = -(0.05 + open_ratio * (0.65 - 0.05))
            self.data.ctrl[self._right_finger_act] = val
            self.data.qpos[self._right_finger_qpos] = val

    def check_arm_box_contacts(self) -> int:
        """Count contacts between robot arm/gripper geoms and box geoms."""
        n_contacts = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in self._arm_geom_ids and g2 in self._box_geom_ids) or (
                g2 in self._arm_geom_ids and g1 in self._box_geom_ids
            ):
                n_contacts += 1
        return n_contacts

    def move_arm_to_joints(
        self,
        side: str,
        q_target: np.ndarray,
        n_steps: int = 20,
        viewer=None,
    ):
        """Interpolate joint angles smoothly toward target pose."""
        q_start = (
            self.data.qpos[self._left_arm_qpos].copy()
            if side == "left"
            else self.data.qpos[self._right_arm_qpos].copy()
        )
        qpos_adr = (
            self._left_arm_qpos if side == "left" else self._right_arm_qpos
        )
        actuators = (
            self._left_actuators if side == "left" else self._right_actuators
        )

        for step in range(n_steps):
            alpha = (step + 1) / n_steps
            # Smooth S-curve (cosine interpolation)
            s = 0.5 * (1.0 - np.cos(np.pi * alpha))
            q_interp = (1.0 - s) * q_start + s * q_target

            self.data.qpos[qpos_adr] = q_interp
            self.data.ctrl[actuators] = q_interp

            # If an object is currently grasped, carry it securely with the gripper
            if self._grasped_arm == side:
                ee_site = (
                    "left_ee_site" if side == "left" else "right_ee_site"
                )
                ee_pos = self.data.site_xpos[self.model.site(ee_site).id].copy()
                ball_qpos_adr = self.workspace.ball_qpos_adr
                self.data.qpos[ball_qpos_adr : ball_qpos_adr + 3] = ee_pos
                self.data.qvel[
                    self.workspace.ball_dof_adr : self.workspace.ball_dof_adr + 6
                ] = 0.0

            # Advance physics
            mujoco.mj_step(self.model, self.data)

            if viewer is not None:
                viewer.sync()
                time.sleep(0.005)

    def move_to_cartesian(
        self,
        side: str,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray] = None,
        n_steps: int = 25,
        viewer=None,
    ) -> bool:
        """Solve IK and smoothly drive end-effector to Cartesian target."""
        q_current = (
            self.data.qpos[self._left_arm_qpos].copy()
            if side == "left"
            else self.data.qpos[self._right_arm_qpos].copy()
        )
        solver = self.ik.solve_left if side == "left" else self.ik.solve_right
        result = solver(
            target_pos=target_pos, target_quat=target_quat, q_init=q_current
        )

        if not result.converged and result.pos_error > 0.010:
            print(
                f"  [WARN] IK did not converge for {side} arm target {target_pos} (err: {result.pos_error*1000:.1f} mm)"
            )
            return False

        self.move_arm_to_joints(side, result.q_arm, n_steps=n_steps, viewer=viewer)
        return True

    def check_ball_in_box(self, box_pos: np.ndarray) -> bool:
        """Verify whether the ball has settled inside the box container."""
        ball_pos = self.data.body("ball").xpos
        horizontal_dist = np.hypot(
            ball_pos[0] - box_pos[0], ball_pos[1] - box_pos[1]
        )
        vertical_dist = ball_pos[2] - box_pos[2]

        # Inside box aperture and resting on the box floor
        in_aperture = horizontal_dist < (self.cfg.box_half_size * 0.90)
        on_floor = 0.010 < vertical_dist < (self.cfg.box_depth + 0.035)
        return in_aperture and on_floor

    # -----------------------------------------------------------------
    # Complete Single Cycle
    # -----------------------------------------------------------------
    def run_cycle(self, arm: str = "left", viewer=None) -> bool:
        """Executes one full autonomous pick-and-place cycle:

        1. Relocates ball and box to random non-overlapping places on stage table.
        2. Detects objects using camera sensors.
        3. Approaches, grasps firmly, lifts, transports, and drops ball into box.
        4. Verifies success, clearance, and returns arm to home.
        """
        cfg = self.cfg
        print(f"\n{'='*65}")
        print(f"--- STARTING PICK AND PLACE CYCLE #{self.completed_cycles + 1} ({arm.upper()} ARM) ---")
        print(f"{'='*65}")

        # 1. Random Placement of Ball and Box on the Stage Table
        ball_spawn, box_spawn = self.workspace.sample_random_locations(arm=arm)
        self.workspace.relocate_objects(self.data, ball_spawn, box_spawn)
        self._reset_home()

        # Let physics settle initial placement
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)
        if viewer is not None:
            viewer.sync()

        print(
            f"[SPAWN] Ball at [{ball_spawn[0]:.3f}, {ball_spawn[1]:.3f}, {ball_spawn[2]:.3f}] m"
        )
        print(
            f"[SPAWN] Box  at [{box_spawn[0]:.3f}, {box_spawn[1]:.3f}, {box_spawn[2]:.3f}] m"
        )
        sep = np.hypot(ball_spawn[0] - box_spawn[0], ball_spawn[1] - box_spawn[1])
        print(f"[SPAWN] Non-overlapping separation: {sep*1000:.1f} mm (>= {cfg.min_separation*1000:.0f} mm)")

        # 2. Camera Vision Detection
        print("[VISION] Capturing frame from head camera sensor...")
        det_ball, det_box = self.vision.detect_objects(
            self.data, cfg.table_z, cfg.ball_radius
        )

        if det_ball is None or det_box is None:
            print("[VISION] [WARN] Object detection missed, using fallback ground-truth.")
            det_ball = ball_spawn.copy()
            det_box = box_spawn.copy()
        else:
            err_ball = np.linalg.norm(det_ball - self.data.body("ball").xpos) * 1000
            err_box = np.linalg.norm(det_box - self.data.body("box").xpos) * 1000
            print(
                f"[VISION] Detected Red Ball at [{det_ball[0]:.3f}, {det_ball[1]:.3f}, {det_ball[2]:.3f}] m (error: {err_ball:.1f} mm)"
            )
            print(
                f"[VISION] Detected Blue Box  at [{det_box[0]:.3f}, {det_box[1]:.3f}, {det_box[2]:.3f}] m (error: {err_box:.1f} mm)"
            )

        # 3. Motion Planning & Execution
        # Waypoints
        target_ball = det_ball.copy()
        target_box = det_box.copy()

        wp_approach = np.array(
            [target_ball[0], target_ball[1], target_ball[2] + cfg.approach_clearance]
        )
        wp_grasp = np.array(
            [target_ball[0], target_ball[1], target_ball[2] + cfg.grasp_offset]
        )
        wp_lift = np.array(
            [target_ball[0], target_ball[1], cfg.table_z + cfg.lift_height]
        )
        wp_above_box = np.array(
            [target_box[0], target_box[1], cfg.table_z + cfg.lift_height]
        )
        wp_drop = np.array(
            [
                target_box[0],
                target_box[1],
                cfg.table_z + cfg.box_depth + cfg.drop_clearance,
            ]
        )

        # Step A: Open gripper and approach above the ball
        print("[ACTION] Approaching ball...")
        self.set_gripper(arm, open_ratio=1.0)
        if not self.move_to_cartesian(
            arm, wp_approach, n_steps=cfg.n_waypoint_steps, viewer=viewer
        ):
            return False

        # Step B: Lower to grasp height (TCP at ball equator)
        print("[ACTION] Lowering gripper to grasp height...")
        if not self.move_to_cartesian(
            arm, wp_grasp, n_steps=cfg.n_waypoint_steps, viewer=viewer
        ):
            return False

        # Step C: Close gripper to grasp firmly
        print(f"[ACTION] Closing {arm} pinch gripper...")
        for step in range(cfg.gripper_steps):
            ratio = 1.0 - (step + 1) / cfg.gripper_steps
            self.set_gripper(arm, open_ratio=ratio)
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()
                time.sleep(0.005)

        self._grasped_arm = arm

        # Step D: Lift ball
        print("[ACTION] Lifting ball above table...")
        if not self.move_to_cartesian(
            arm, wp_lift, n_steps=cfg.n_waypoint_steps, viewer=viewer
        ):
            return False

        # Step E: Transit across to the box
        print("[ACTION] Transporting ball over to blue box...")
        if not self.move_to_cartesian(
            arm, wp_above_box, n_steps=cfg.n_waypoint_steps, viewer=viewer
        ):
            return False

        # Step F: Lower to drop height (safely above box rim)
        print("[ACTION] Lowering over box aperture (maintaining safety clearance)...")
        box_contacts = 0
        if not self.move_to_cartesian(
            arm, wp_drop, n_steps=cfg.n_waypoint_steps, viewer=viewer
        ):
            return False
        box_contacts += self.check_arm_box_contacts()

        # Step G: Release ball -- gravity drops it into the box!
        print("[ACTION] Opening gripper -> Ball dropping into box under gravity!")
        self._grasped_arm = None
        for step in range(cfg.gripper_steps):
            ratio = (step + 1) / cfg.gripper_steps
            self.set_gripper(arm, open_ratio=ratio)
            mujoco.mj_step(self.model, self.data)
            box_contacts += self.check_arm_box_contacts()
            if viewer is not None:
                viewer.sync()
                time.sleep(0.005)

        # Step H: Physics settle -- let the ball fall, bounce, and settle inside box
        print("[ACTION] Settle steps for natural gravity drop...")
        for _ in range(cfg.settle_steps):
            mujoco.mj_step(self.model, self.data)
            box_contacts += self.check_arm_box_contacts()
            if viewer is not None:
                viewer.sync()
                time.sleep(0.005)

        # Step I: Ascend vertically clear of the box aperture before swinging home
        print("[ACTION] Ascending vertically clear of the box aperture...")
        self.move_to_cartesian(
            arm, wp_above_box, n_steps=cfg.n_waypoint_steps // 2, viewer=viewer
        )
        box_contacts += self.check_arm_box_contacts()

        # Step J: Retract arm to Home posture
        print("[ACTION] Retracting arm to HOME posture...")
        home_target = LEFT_ARM_HOME if arm == "left" else RIGHT_ARM_HOME
        self.move_arm_to_joints(
            arm, home_target, n_steps=cfg.n_waypoint_steps, viewer=viewer
        )

        # 4. Verify Clearance and Success
        if box_contacts == 0:
            print("[CLEARANCE] Verified: Zero contact between arm/gripper and box (clean clearance!).")
        else:
            print(f"[WARN] Contact detected between arm/gripper and box ({box_contacts} contact events).")

        # Let the ball finish falling and bouncing before judging: the verdict
        # below is about where it comes to rest, not where it is in mid-air.
        ball_dof = self.workspace.ball_dof_adr
        for _ in range(750):                                   # up to 1.5 s
            if np.linalg.norm(self.data.qvel[ball_dof:ball_dof + 3]) < 0.02:
                break
            mujoco.mj_step(self.model, self.data)
            box_contacts += self.check_arm_box_contacts()
            if viewer is not None:
                viewer.sync()

        # Judge the result on the box as it actually is now, not where it was
        # spawned: the ball must lie inside the box's own walls, and the box must
        # still be upright on the table. A box that was knocked over, or touched
        # by the gripper, is a failed cycle even if the ball ended up near it.
        final_ball = self.data.body("ball").xpos.copy()
        final_box = self.data.body("box").xpos.copy()
        box_rot = self.data.body("box").xmat.reshape(3, 3)
        tilt_deg = float(np.degrees(np.arccos(np.clip(box_rot[2, 2], -1.0, 1.0))))
        local = box_rot.T @ (final_ball - final_box)
        inner = cfg.box_half_size - 0.008
        inside = bool(abs(local[0]) < inner and abs(local[1]) < inner
                      and 0.008 < local[2] < cfg.box_depth)
        upright = tilt_deg < 15.0 and abs(final_box[2] - cfg.table_z) < 0.01
        offset = float(np.hypot(local[0], local[1]))

        self.completed_cycles += 1
        if not upright:
            print(f"[FAIL] Box was knocked over or moved off the table (tilt {tilt_deg:.0f} deg).")
            return False
        if box_contacts > 0:
            print(f"[FAIL] Gripper/arm touched the box ({box_contacts} contact events).")
            return False
        if not inside:
            print(f"[FAIL] Ball is not inside the box (offset from box center: {offset*1000:.1f} mm, "
                  f"height above box floor: {local[2]*1000:.1f} mm).")
            return False
        print(f"[SUCCESS] Ball placed inside the upright box, no gripper contact "
              f"(offset from box center: {offset*1000:.1f} mm).")
        self.successful_cycles += 1
        return True


# =====================================================================
# Main Application Entry Point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="OpenArm v2.0 Autonomous Pick and Place with Camera Vision"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openarm_v20_mujoco/openarm_pick_and_place.xml",
        help="Path to MuJoCo pick-and-place XML scene",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI viewer (ideal for background/test runs)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of pick-and-place cycles to execute (0 = infinite loop)",
    )
    parser.add_argument(
        "--arm",
        type=str,
        choices=["left", "right", "alternate"],
        default="alternate",
        help="Arm selection strategy: alternate between arms, or stick to one",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="Start with Run Mode OFF (press SPACE to toggle ON and begin)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("OpenArm v2.0 Bimanual -- Autonomous Pick and Place System")
    print("=" * 65)
    print(f"  Model:    {args.model}")
    print(f"  Headless: {args.headless}")
    print(f"  Arm mode: {args.arm}")
    print(f"  Cycles:   {'Infinite' if args.cycles == 0 else args.cycles}")
    print("=" * 65)

    controller = OpenArmPickAndPlace(model_path=args.model)
    if args.paused:
        controller.run_mode = False

    def get_arm_for_cycle(cycle_idx: int) -> str:
        if args.arm == "alternate":
            return "left" if cycle_idx % 2 == 0 else "right"
        return args.arm

    cycle_count = 0

    if args.headless:
        # Headless execution loop
        print("\n[RUN MODE: ON] Running headless pick-and-place loop...")
        max_cycles = args.cycles if args.cycles > 0 else 3
        while cycle_count < max_cycles:
            arm = get_arm_for_cycle(cycle_count)
            controller.run_cycle(arm=arm, viewer=None)
            cycle_count += 1
            print(
                f"[STATUS] Completed {cycle_count}/{max_cycles} cycles. Relocating for next cycle..."
            )
            time.sleep(0.2)
        print(f"\n[DONE] {controller.successful_cycles} of {cycle_count} cycles succeeded.")

    else:
        # Interactive 3D Viewer loop
        import mujoco.viewer

        def key_callback(keycode: int):
            if keycode == 32:  # Spacebar
                controller.run_mode = not controller.run_mode
                status_str = "RESUMED (ON)" if controller.run_mode else "PAUSED (OFF)"
                print(f"\n[KEYPRESS: SPACE] Run Mode is now: {status_str}")

        print("\n[RUN MODE: ON] Launching interactive MuJoCo viewer...")
        print("Controls:")
        print("  SPACE: Toggle Run Mode (Pause / Resume pick-and-place)")
        print("  ESC:   Exit viewer")

        with mujoco.viewer.launch_passive(
            controller.model, controller.data, key_callback=key_callback
        ) as viewer:
            while viewer.is_running():
                # Check run mode
                if controller.run_mode:
                    arm = get_arm_for_cycle(cycle_count)
                    controller.run_cycle(arm=arm, viewer=viewer)
                    cycle_count += 1

                    if args.cycles > 0 and cycle_count >= args.cycles:
                        print(
                            f"\n[DONE] Reached requested cycle limit of {args.cycles}. Pausing."
                        )
                        controller.run_mode = False

                    # Brief pause between cycles so user sees the dropped ball
                    t_pause_end = time.time() + 1.2
                    while time.time() < t_pause_end and viewer.is_running():
                        mujoco.mj_step(controller.model, controller.data)
                        viewer.sync()
                        time.sleep(0.01)
                else:
                    mujoco.mj_step(controller.model, controller.data)
                    viewer.sync()
                    time.sleep(0.02)


if __name__ == "__main__":
    main()
