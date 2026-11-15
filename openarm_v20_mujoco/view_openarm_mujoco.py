#!/usr/bin/env python3
"""
OpenArm v2.0 Bimanual Gymnasium Environment
=============================================
Full sensor integration matching the physical OpenArm hardware:
  1. In-Hand Cameras  — RGB image from gripper-embedded cameras (left & right)
  2. Motor Encoders   — Joint position & velocity via 9 kHz CAN-FD bus
  3. Torque Feedback  — Actuator current/force from QDD motors
  4. Safety Curtains  — Area rangefinder sensors around the OpenArm Cell
  5. Head Camera      — ZED Mini stereo depth camera on central pedestal
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from pathlib import Path

# Import your existing build functions (assuming they are in view_openarm_mujoco.py)
# Or paste your prepare() and build_model() functions directly here
from view_openarm_mujoco import prepare, build_model


class OpenArmBimanualEnv(gym.Env):
    """Custom Gymnasium Environment for the OpenArm v2.0 Bimanual Robot.

    Observation space includes:
      - Joint positions and velocities  (motor encoders)
      - Actuator forces                 (torque / current feedback)
      - Rangefinder readings             (safety light curtains)
      - Optional: RGB images from in-hand cameras

    The full sensor vector is:
      [ qpos (nq) | qvel (nv) | sensordata (nsensordata) ]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # --- Named sensor indices (matching <sensor> order in openarm_v20.xml) ---
    # These constants let you slice self.data.sensordata by semantic group.
    N_JOINTS_PER_ARM = 8          # 7 arm joints + 1 gripper
    N_ENCODER_POS    = 16         # 8 left + 8 right  position encoders
    N_ENCODER_VEL    = 16         # 8 left + 8 right  velocity encoders
    N_TORQUE         = 16         # 8 left + 8 right  actuator force sensors
    N_SAFETY         = 4          # 4 rangefinder curtains

    # --- Camera configuration ---
    CAMERA_RES_W          = 640
    CAMERA_RES_H          = 480
    HEAD_CAMERA_RES_W     = 1280   # ZED Mini HD mode
    HEAD_CAMERA_RES_H     = 720
    HEAD_CAMERA_BASELINE  = 0.063  # 63 mm stereo baseline

    def __init__(self, urdf_path: str, out_dir: str = "mj_out",
                 render_mode: str = None, use_cameras: bool = False):
        super().__init__()

        self.render_mode = render_mode
        self.use_cameras = use_cameras
        urdf_p = Path(urdf_path).resolve()
        out_p = Path(out_dir).resolve()

        # 1. Compile the MuJoCo Model using your custom spec adjustments
        prepared_urdf = prepare(urdf_p, out_p)
        self.spec, self.model = build_model(prepared_urdf, urdf_p)
        self.data = mujoco.MjData(self.model)

        # 2. Define Action Space (One continuous target per actuator)
        self.num_actuators = self.model.nu
        bounds = self.model.actuator_ctrlrange.copy()

        self.action_space = spaces.Box(
            low=bounds[:, 0],
            high=bounds[:, 1],
            dtype=np.float32
        )

        # 3. Define Observation Space
        #    Vector obs = qpos + qvel + sensordata (encoders + torque + safety)
        self._sensor_dim = self.model.nsensordata  # total scalar sensor readings
        obs_dim = self.model.nq + self.model.nv + self._sensor_dim

        obs_spaces = {
            "state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_dim,), dtype=np.float32
            ),
        }

        # Optional camera observations (Dict space when cameras enabled)
        if self.use_cameras:
            hand_img_shape = (self.CAMERA_RES_H, self.CAMERA_RES_W, 3)
            head_img_shape = (self.HEAD_CAMERA_RES_H, self.HEAD_CAMERA_RES_W, 3)
            obs_spaces["left_hand_camera"]  = spaces.Box(
                low=0, high=255, shape=hand_img_shape, dtype=np.uint8)
            obs_spaces["right_hand_camera"] = spaces.Box(
                low=0, high=255, shape=hand_img_shape, dtype=np.uint8)
            obs_spaces["head_camera_left"]  = spaces.Box(
                low=0, high=255, shape=head_img_shape, dtype=np.uint8)
            obs_spaces["head_camera_right"] = spaces.Box(
                low=0, high=255, shape=head_img_shape, dtype=np.uint8)
            obs_spaces["head_camera_center"] = spaces.Box(
                low=0, high=255, shape=head_img_shape, dtype=np.uint8)

        if self.use_cameras:
            self.observation_space = spaces.Dict(obs_spaces)
        else:
            self.observation_space = obs_spaces["state"]

        # 4. Look up camera IDs for the in-hand cameras
        self._left_cam_id  = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "openarm_left_in_hand_camera")
        self._right_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "openarm_right_in_hand_camera")

        # 4b. Look up camera IDs for the head-mounted ZED stereo pair
        self._head_cam_left_id   = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "openarm_head_camera_left")
        self._head_cam_right_id  = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "openarm_head_camera_right")
        self._head_cam_center_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "openarm_head_camera_center")

        # 5. Resolve sensor address ranges for semantic slicing
        self._build_sensor_index()

        # 6. Set up rendering
        self.renderer = None
        self._cam_renderer = None
        self._head_cam_renderer = None
        if self.render_mode == "human":
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
        elif self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(
                self.model, height=self.CAMERA_RES_H, width=self.CAMERA_RES_W)

        if self.use_cameras:
            self._cam_renderer = mujoco.Renderer(
                self.model, height=self.CAMERA_RES_H, width=self.CAMERA_RES_W)
            self._head_cam_renderer = mujoco.Renderer(
                self.model,
                height=self.HEAD_CAMERA_RES_H,
                width=self.HEAD_CAMERA_RES_W)

    # ------------------------------------------------------------------
    # Sensor index helper
    # ------------------------------------------------------------------
    def _build_sensor_index(self):
        """Build a dict mapping sensor names → slices into sensordata."""
        self.sensor_slices = {}
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            adr  = self.model.sensor_adr[i]
            dim  = self.model.sensor_dim[i]
            self.sensor_slices[name] = slice(adr, adr + dim)

    def get_sensor(self, name: str) -> np.ndarray:
        """Read a named sensor value from the current sensordata buffer."""
        return self.data.sensordata[self.sensor_slices[name]].copy()

    # ------------------------------------------------------------------
    # Convenience accessors for each sensor category
    # ------------------------------------------------------------------
    def get_encoder_positions(self, side: str = "left") -> np.ndarray:
        """Return 8-element array of joint positions [rad] for one arm.

        Sensor 2: Motor Encoders — position channel.
        Order: shoulder_pan, shoulder_lift, shoulder_rot, elbow_flex,
               wrist_roll, wrist_pitch, wrist_rot, gripper
        """
        prefix = f"{side}_encoder_pos_"
        names = ["shoulder_pan", "shoulder_lift", "shoulder_rot",
                 "elbow_flex", "wrist_roll", "wrist_pitch",
                 "wrist_rot", "gripper"]
        return np.array([self.get_sensor(prefix + n)[0] for n in names],
                        dtype=np.float32)

    def get_encoder_velocities(self, side: str = "left") -> np.ndarray:
        """Return 8-element array of joint velocities [rad/s] for one arm.

        Sensor 2: Motor Encoders — velocity channel.
        """
        prefix = f"{side}_encoder_vel_"
        names = ["shoulder_pan", "shoulder_lift", "shoulder_rot",
                 "elbow_flex", "wrist_roll", "wrist_pitch",
                 "wrist_rot", "gripper"]
        return np.array([self.get_sensor(prefix + n)[0] for n in names],
                        dtype=np.float32)

    def get_actuator_torques(self, side: str = "left") -> np.ndarray:
        """Return 8-element array of actuator forces [N·m] for one arm.

        Sensor 3: Current/Torque Feedback from QDD actuators.
        Used for bilateral force sensing, gravity compensation,
        and contact-rich teleoperation without standalone F/T sensors.
        """
        prefix = f"{side}_torque_"
        names = ["shoulder_pan", "shoulder_lift", "shoulder_rot",
                 "elbow_flex", "wrist_roll", "wrist_pitch",
                 "wrist_rot", "gripper"]
        return np.array([self.get_sensor(prefix + n)[0] for n in names],
                        dtype=np.float32)

    def get_safety_curtain_readings(self) -> dict[str, float]:
        """Return rangefinder distances [m] for all 4 safety curtain planes.

        Sensor 4: Area Sensors / Light Curtains.
        A reading of -1 means no object detected in range.
        Any value >= 0 indicates an intrusion into the cell boundary.
        """
        return {
            "front": float(self.get_sensor("safety_rangefinder_front")[0]),
            "back":  float(self.get_sensor("safety_rangefinder_back")[0]),
            "left":  float(self.get_sensor("safety_rangefinder_left")[0]),
            "right": float(self.get_sensor("safety_rangefinder_right")[0]),
        }

    def get_in_hand_camera_image(self, side: str = "left") -> np.ndarray:
        """Render and return an RGB image from the specified in-hand camera.

        Sensor 1: In-Hand Camera (Vision Sensor).
        Returns (480, 640, 3) uint8 array.
        """
        if self._cam_renderer is None:
            raise RuntimeError(
                "Camera renderer not initialised. "
                "Set use_cameras=True when creating the environment.")
        cam_id = (self._left_cam_id if side == "left"
                  else self._right_cam_id)
        self._cam_renderer.update_scene(self.data, camera=cam_id)
        return self._cam_renderer.render().copy()

    def get_head_camera_image(self, view: str = "center") -> np.ndarray:
        """Render and return an RGB image from the head-mounted ZED camera.

        Sensor 5: Head Camera (ZED Stereo Depth Camera).
        Mounted on top of the central pedestal column (MISUMI extrusion frame).
        view options: "center" (mono convenience view), "left" (left stereo lens),
                      "right" (right stereo lens, 63 mm baseline).
        Returns (720, 1280, 3) uint8 array.
        """
        if self._head_cam_renderer is None:
            raise RuntimeError(
                "Head camera renderer not initialised. "
                "Set use_cameras=True when creating the environment.")
        if view == "left":
            cam_id = self._head_cam_left_id
        elif view == "right":
            cam_id = self._head_cam_right_id
        else:
            cam_id = self._head_cam_center_id

        self._head_cam_renderer.update_scene(self.data, camera=cam_id)
        return self._head_cam_renderer.render().copy()

    def is_safety_breached(self) -> bool:
        """Check if any safety curtain detects an intrusion (distance >= 0)."""
        readings = self.get_safety_curtain_readings()
        return any(v >= 0.0 for v in readings.values())

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------
    def _get_obs(self):
        """Build the observation from physics state + all scalar sensors."""
        state_vec = np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
            self.data.sensordata.flat,
        ]).astype(np.float32)

        if self.use_cameras:
            return {
                "state":              state_vec,
                "left_hand_camera":   self.get_in_hand_camera_image("left"),
                "right_hand_camera":  self.get_in_hand_camera_image("right"),
                "head_camera_left":   self.get_head_camera_image("left"),
                "head_camera_right":  self.get_head_camera_image("right"),
                "head_camera_center": self.get_head_camera_image("center"),
            }
        return state_vec

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Resets the environment to an initial state."""
        super().reset(seed=seed)

        # Reset physics state variables
        mujoco.mj_resetData(self.model, self.data)

        # (Optional) Inject slight random noise to initial position
        # self.data.qpos[:] += self.np_random.uniform(
        #     low=-0.01, high=0.01, size=self.model.nq)

        mujoco.mj_forward(self.model, self.data)

        if self.render_mode == "human":
            self.viewer.sync()

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple:
        """Applies actions, steps physics, computes rewards.

        Returns (obs, reward, terminated, truncated, info).
        The info dict includes structured sensor readings for debugging.
        """
        # 1. Clip and apply the actions to the actuators control vector
        clipped_action = np.clip(
            action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = clipped_action

        # 2. Advance the simulation (multiple sub-steps per control cycle)
        control_substeps = 5
        for _ in range(control_substeps):
            mujoco.mj_step(self.model, self.data)

        # 3. Compute rewards
        reward = -np.sum(np.square(action)) * 0.01

        # 4. Check safety curtain — terminate if breached
        safety_breached = self.is_safety_breached()
        terminated = safety_breached
        truncated = False

        # 5. Build info dict with structured sensor data
        info = {
            "sensors": {
                "left_encoder_pos":   self.get_encoder_positions("left"),
                "left_encoder_vel":   self.get_encoder_velocities("left"),
                "right_encoder_pos":  self.get_encoder_positions("right"),
                "right_encoder_vel":  self.get_encoder_velocities("right"),
                "left_torque":        self.get_actuator_torques("left"),
                "right_torque":       self.get_actuator_torques("right"),
                "safety_curtains":    self.get_safety_curtain_readings(),
                "safety_breached":    safety_breached,
            }
        }

        if self.render_mode == "human":
            self.viewer.sync()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        """Renders frames if using 'rgb_array' mode."""
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data)
            return self.renderer.render()

    def close(self):
        """Cleans up resources."""
        if self.render_mode == "human" and hasattr(self, "viewer"):
            self.viewer.close()
        if self._cam_renderer is not None:
            self._cam_renderer.close()
        if self._head_cam_renderer is not None:
            self._head_cam_renderer.close()


# --- Quick Test Loop ---
if __name__ == "__main__":
    # Path to your actual URDF
    URDF_FILE = "openarm_v20_bimanual/urdf/openarm_v20_bimanual.urdf"

    # Initialize the environment in 'human' mode to open the interactive viewer
    env = OpenArmBimanualEnv(urdf_path=URDF_FILE, render_mode="human")

    obs, info = env.reset()
    print(f"Initial Observation shape: {obs.shape}")
    print(f"Action Space bounds: {env.action_space}")
    print(f"Sensor data dimension: {env._sensor_dim}")

    # Print initial sensor readings
    print("\n--- Initial Sensor Readings ---")
    print(f"Left encoder positions:  {env.get_encoder_positions('left')}")
    print(f"Right encoder positions: {env.get_encoder_positions('right')}")
    print(f"Left actuator torques:   {env.get_actuator_torques('left')}")
    print(f"Safety curtain readings: {env.get_safety_curtain_readings()}")

    # Run a dummy random action loop for 200 steps
    for step_idx in range(200):
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)

        # Check safety curtain each step
        if info["sensors"]["safety_breached"]:
            print(f"Step {step_idx}: SAFETY BREACH DETECTED — stopping!")
            break

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("Test finished successfully!")
