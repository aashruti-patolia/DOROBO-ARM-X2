"""
Camera perception (scripted): find the ball and the basket with the head camera.

Pipeline, every ``every_n_ticks`` control ticks:

1. Render RGB from ``openarm_head_camera_center`` (the ZED's centre view) at
   low resolution, and measure depth for the pixels of interest by ray casting
   from the real camera. That gives exactly the depth a depth buffer would
   hold, without the slow GPU depth readback. A sensor model adds depth noise
   that grows with distance, pixel noise, and occasional dropped frames.
2. Segment by colour on chromaticity (each channel divided by the sum), which
   is robust to lighting changes: red ball, blue basket, neutral table.
3. Back-project the segmented pixels to 3-D with the depth image, the camera
   intrinsics (``fovy``) and the *nominal* camera pose. Calibration error
   enters naturally when visual randomisation moves the real camera.
4. Estimate the following:
   * Ball centre: mean of the visible surface points, pushed back by 2/3 of
     the estimated radius along the viewing ray (the mean depth of a
     camera-facing hemisphere).
   * Basket: centre and half-width from the robust x/y extent of its points,
     and rim height from the upper z percentile.
   * Table height: the dominant height of the neutral pixels.
5. Track over time: a low-pass velocity estimate, and when the ball is not
   seen, the last estimate is kept with ``visible = False`` and an age.

Only the outputs of this module (and proprioception) reach the policy. The
simulator's true object positions never do.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from ..config import CameraConfig


@dataclass
class Detection:
    """One camera frame's estimates (None = not seen in this frame)."""
    ball_pos: np.ndarray | None = None
    ball_radius: float | None = None
    box_pos: np.ndarray | None = None       # basket rim centre (x, y, rim z)
    box_half: float | None = None
    table_z: float | None = None
    ball_pixels: int = 0
    box_pixels: int = 0
    ball_dist: float | None = None          # camera-to-ball distance (for the occlusion check)


@dataclass
class TrackedState:
    """What the robot believes about the world (fed to the policy)."""
    ball_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ball_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ball_radius: float = 0.022
    ball_visible: bool = False
    ball_age: int = 0                       # ticks since the ball was last seen
    ball_ever_seen: bool = False
    box_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    box_half: float = 0.045
    box_visible: bool = False
    box_ever_seen: bool = False
    table_z: float = 0.25


class HeadCamera:
    """Renders the head camera and turns frames into object estimates."""

    def __init__(self, model: mujoco.MjModel, cfg: CameraConfig):
        self.model = model
        self.cfg = cfg
        self.cam_id = model.camera(cfg.camera).id
        self.body_id = model.cam_bodyid[self.cam_id]
        # Nominal calibration (perception keeps using it even when the real
        # camera is jittered by visual randomisation).
        self.cam_pos_nom = model.cam_pos[self.cam_id].copy()
        self.cam_quat_nom = model.cam_quat[self.cam_id].copy()
        self.f = (cfg.height / 2.0) / np.tan(np.radians(model.cam_fovy[self.cam_id]) / 2.0)
        self.cx, self.cy = cfg.width / 2.0, cfg.height / 2.0
        v, u = np.mgrid[0:cfg.height, 0:cfg.width]
        self._ray_x = (u + 0.5 - self.cx) / self.f      # camera-frame ray per pixel (z = -1)
        self._ray_y = -(v + 0.5 - self.cy) / self.f
        self._renderer = None                           # created lazily (per process)
        self._ray_groups = np.array([1, 0, 0, 0, 0, 0], np.uint8)   # world geoms only
        # Render the world (group 0) and the robot's simple collision meshes
        # (group 3, grey) instead of its heavy visual meshes: same occlusion,
        # cheaper frames.
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[:] = [1, 0, 0, 1, 0, 0]

    # -- rendering ----------------------------------------------------------
    def _get_renderer(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.cfg.height, width=self.cfg.width)
        return self._renderer

    def render_rgb(self, data: mujoco.MjData) -> np.ndarray:
        """RGB frame (uint8). Shadows/reflections are off: 5x cheaper, same colours."""
        r = self._get_renderer()
        r.update_scene(data, camera=self.cam_id, scene_option=self._scene_option)
        for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION,
                     mujoco.mjtRndFlag.mjRND_SKYBOX):
            r.scene.flags[flag] = 0
        return r.render().copy()

    def render_depth(self, data: mujoco.MjData) -> np.ndarray:
        """Full depth image from the GPU (slow; for tests and debugging only)."""
        r = self._get_renderer()
        r.enable_depth_rendering()
        r.update_scene(data, camera=self.cam_id, scene_option=self._scene_option)
        depth = r.render().copy()
        r.disable_depth_rendering()
        return depth

    def ray_depth(self, data: mujoco.MjData, flat_idx: np.ndarray) -> np.ndarray:
        """Depth along the optical axis for the given pixels, by ray casting from
        the real camera against the world geometry (group 0: table, ball,
        basket, distractors, floor). Pixels with no hit get 0."""
        n = len(flat_idx)
        if n == 0:
            return np.zeros(0)
        pos = data.cam_xpos[self.cam_id].copy()
        R = data.cam_xmat[self.cam_id].reshape(3, 3)
        dirs = np.stack([self._ray_x.ravel()[flat_idx], self._ray_y.ravel()[flat_idx], -np.ones(n)], axis=1)
        norms = np.linalg.norm(dirs, axis=1)
        vec = np.ascontiguousarray(((dirs / norms[:, None]) @ R.T).ravel())
        geomid = np.zeros(n, np.int32)
        dist = np.zeros(n)
        mujoco.mj_multiRay(self.model, data, pos, vec, self._ray_groups, True, -1, geomid, dist, None, n, 5.0)
        return np.where(geomid >= 0, dist / norms, 0.0)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -- geometry -------------------------------------------------------------
    def nominal_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """World position and rotation of the camera per the nominal calibration."""
        bpos = data.xpos[self.body_id]
        bmat = data.xmat[self.body_id].reshape(3, 3)
        R_local = np.zeros(9)
        mujoco.mju_quat2Mat(R_local, self.cam_quat_nom)
        return bpos + bmat @ self.cam_pos_nom, bmat @ R_local.reshape(3, 3)

    def backproject(self, flat_idx: np.ndarray, depth: np.ndarray, cam_pos, cam_R) -> np.ndarray:
        """World points (n, 3) for pixels ``flat_idx`` with the given depths."""
        x = self._ray_x.ravel()[flat_idx] * depth
        y = self._ray_y.ravel()[flat_idx] * depth
        return np.stack([x, y, -depth], axis=1) @ cam_R.T + cam_pos

    def pixel_to_world(self, u: float, v: float, d: float, cam_pos, cam_R) -> np.ndarray:
        p = np.array([(u + 0.5 - self.cx) / self.f * d, -(v + 0.5 - self.cy) / self.f * d, -d])
        return cam_R @ p + cam_pos

    # -- detection ------------------------------------------------------------
    def segment(self, rgb: np.ndarray):
        """Colour masks on chromaticity, in integer arithmetic (channel / sum)."""
        c = self.cfg
        r, g, b = (rgb[..., k].astype(np.int32) for k in range(3))
        s = r + g + b
        red = (100 * r > int(100 * c.red_chroma) * s) & (r > c.red_min_value)
        blue = ((100 * b > int(100 * c.blue_chroma) * s) & (b > c.blue_min_value)
                & (100 * (b - r) > 15 * s))
        neutral = ((100 * np.maximum(np.maximum(r, g), b) < int(100 * c.neutral_max_chroma) * s)
                   & (s > 150))
        return red, blue, neutral

    def capture(self, data: mujoco.MjData, rng=None, noise_level: float = 0.0) -> Detection:
        """Render a frame, apply the sensor model and detect the objects."""
        c = self.cfg
        rgb = self.render_rgb(data)
        if noise_level > 0 and rng is not None:
            noise = rng.standard_normal(rgb.shape, dtype=np.float32) * (c.rgb_noise * noise_level)
            rgb = np.clip(rgb + noise, 0, 255).astype(np.uint8)
        return self.detect(rgb, data, rng, noise_level)

    def _depth_for(self, data, flat_idx, rng, noise_level):
        depth = self.ray_depth(data, flat_idx)
        if noise_level > 0 and rng is not None and len(depth):
            c = self.cfg
            std = (c.depth_noise_base + c.depth_noise_quad * depth ** 2) * noise_level
            depth = depth + rng.normal(0, 1, len(depth)) * std
        return depth

    def detect(self, rgb: np.ndarray, data: mujoco.MjData, rng=None, noise_level: float = 0.0) -> Detection:
        c = self.cfg
        det = Detection()
        red, blue, neutral = self.segment(rgb)
        cam_pos, cam_R = self.nominal_pose(data)

        # Table height: dominant height of (a sample of) neutral pixels
        idx = np.flatnonzero(neutral.ravel())
        if idx.size >= 30:
            if idx.size > 200:
                idx = (rng.choice(idx, 200, replace=False) if rng is not None
                       else idx[np.linspace(0, idx.size - 1, 200).astype(int)])
            dep = self._depth_for(data, idx, rng, noise_level)
            ok = dep > 0.05
            z = self.backproject(idx[ok], dep[ok], cam_pos, cam_R)[:, 2]
            z = z[(z > 0.08) & (z < 0.5)]                       # the floor is at 0
            if z.size >= 30:
                hist, edges = np.histogram(z, bins=np.arange(0.08, 0.5, 0.005))
                k = int(np.argmax(hist))
                peak = 0.5 * (edges[k] + edges[k + 1])
                det.table_z = float(np.median(z[np.abs(z - peak) < 0.006]))

        # Ball
        idx = np.flatnonzero(red.ravel())
        det.ball_pixels = int(idx.size)
        if idx.size >= c.min_ball_pixels:
            dep = self._depth_for(data, idx, rng, noise_level)
            ok = dep > 0.05
            if ok.sum() >= c.min_ball_pixels:
                pts = self.backproject(idx[ok], dep[ok], cam_pos, cam_R)
                mean = np.median(pts, axis=0) if len(pts) < 12 else pts.mean(axis=0)
                dist = np.linalg.norm(mean - cam_pos)
                r_est = float(np.sqrt(idx.size / np.pi) * dist / self.f)    # blob area -> metres
                ray = (mean - cam_pos) / (dist + 1e-12)
                det.ball_radius = r_est
                det.ball_pos = mean + ray * (2.0 / 3.0) * r_est              # surface mean -> centre
                det.ball_dist = float(dist)

        # Basket
        idx = np.flatnonzero(blue.ravel())
        if idx.size >= c.min_box_pixels:
            dep = self._depth_for(data, idx, rng, noise_level)
            ok = dep > 0.05
            pts = self.backproject(idx[ok], dep[ok], cam_pos, cam_R)
            if det.table_z is not None:                          # the basket stands on the table
                pts = pts[(pts[:, 2] > det.table_z - 0.02) & (pts[:, 2] < det.table_z + 0.15)]
            if len(pts) >= c.min_box_pixels:                     # keep the main blob
                med = np.median(pts[:, :2], axis=0)
                pts = pts[np.linalg.norm(pts[:, :2] - med, axis=1) < 0.12]
            det.box_pixels = len(pts)
            if len(pts) >= c.min_box_pixels:
                lo = np.percentile(pts[:, :2], 3, axis=0)
                hi = np.percentile(pts[:, :2], 97, axis=0)
                centre = 0.5 * (lo + hi)
                det.box_pos = np.array([centre[0], centre[1], float(np.percentile(pts[:, 2], 97))])
                det.box_half = float(np.mean(0.5 * (hi - lo)))
        return det


class ObjectTracker:
    """Keeps the robot's belief about the ball and basket between frames."""

    def __init__(self, cfg: CameraConfig, dt_tick: float, focal_px: float | None = None):
        self.cfg = cfg
        self.dt_tick = dt_tick
        self.focal_px = focal_px            # lets a partly hidden ball be recognised
        self.state = TrackedState()
        self._ticks_since_frame = 0
        self._radius_samples: list[float] = []

    def reset(self):
        self.state = TrackedState()
        self._ticks_since_frame = 0
        self._radius_samples = []

    def tick(self):
        """Advance time by one control tick without a new frame."""
        self._ticks_since_frame += 1
        self.state.ball_age += 1
        # constant-velocity prediction while the ball is not observed
        if self.state.ball_ever_seen and not self.state.ball_visible:
            self.state.ball_pos = self.state.ball_pos + self.state.ball_vel * self.dt_tick

    def update(self, det: Detection):
        """Fold in a new frame (call after ``tick`` on frame ticks)."""
        s, c = self.state, self.cfg
        dt = max(self._ticks_since_frame, 1) * self.dt_tick
        self._ticks_since_frame = 0
        if (det.ball_pos is not None and self.focal_px and det.ball_dist and len(self._radius_samples) >= 2):
            expected = np.pi * (self.focal_px * s.ball_radius / det.ball_dist) ** 2
            if det.ball_pixels < c.occlusion_ratio * expected:
                det = replace(det, ball_pos=None)     # partly hidden (e.g. by the hand): centre is biased
        if det.ball_pos is not None:
            if s.ball_visible and s.ball_ever_seen:
                v = (det.ball_pos - s.ball_pos) / dt
                s.ball_vel = c.velocity_smoothing * s.ball_vel + (1 - c.velocity_smoothing) * v
            else:
                s.ball_vel = np.zeros(3)
            s.ball_pos = det.ball_pos.copy()
            s.ball_visible = True
            s.ball_ever_seen = True
            s.ball_age = 0
            if det.ball_radius is not None and det.ball_pixels >= 20:
                self._radius_samples.append(det.ball_radius)
                s.ball_radius = float(np.median(self._radius_samples[-15:]))
        else:
            s.ball_visible = False
            s.ball_vel = s.ball_vel * 0.8
        if det.box_pos is not None:
            s.box_pos = det.box_pos.copy()
            s.box_half = det.box_half
            s.box_visible = True
            s.box_ever_seen = True
        else:
            s.box_visible = False
        if det.table_z is not None:
            s.table_z = det.table_z


class Perception:
    """Camera + tracker with a frame schedule and frame drops."""

    def __init__(self, model: mujoco.MjModel, cfg: CameraConfig, dt_tick: float):
        self.cfg = cfg
        self.camera = HeadCamera(model, cfg)
        self.tracker = ObjectTracker(cfg, dt_tick, focal_px=self.camera.f)
        self._tick = 0
        self.last_detection = Detection()

    @property
    def state(self) -> TrackedState:
        return self.tracker.state

    def reset(self, data, rng, visual_level: float, sensor_level: float, n_frames: int = 2):
        """Start a new episode: a couple of frames to initialise the belief."""
        self.tracker.reset()
        self._tick = 0
        for _ in range(n_frames):
            self._capture(data, rng, visual_level, sensor_level=0.0)
        return self.state

    def _capture(self, data, rng, visual_level, sensor_level):
        if sensor_level > 0 and rng.random() < self.cfg.frame_drop_prob * sensor_level:
            self.tracker.update(Detection())          # lost frame: nothing seen
            return
        det = self.camera.capture(data, rng, visual_level)
        self.last_detection = det
        self.tracker.update(det)

    def tick(self, data, rng, visual_level: float, sensor_level: float, every: int | None = None):
        """One control tick; captures a frame every ``every`` (default ``every_n_ticks``) ticks."""
        self._tick += 1
        self.tracker.tick()
        if self._tick % (every or self.cfg.every_n_ticks) == 0:
            self._capture(data, rng, visual_level, sensor_level)

    def close(self):
        self.camera.close()
