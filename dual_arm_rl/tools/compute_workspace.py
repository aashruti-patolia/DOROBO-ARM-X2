"""
Reachability maps: where each arm can pick up the ball and where it can hold
the ball above the basket, for every table height the scenarios produce.

Writes ``dual_arm_rl/assets/workspace_maps.npz``. The scene samples ball and
basket placements from these maps, and the scripted planner uses them to know
whether (and with which hand yaw) a ball is graspable.

For every table height and every cell of a grid over the table (each arm
separately):

* **ball map**: the arm reaches the ball with the fingers pointing straight
  down at the pre-grasp hover and at the grasp height, can then lift it to the
  travel height, and the grasp pose touches nothing but the ball. The hand yaw
  is free for a sphere, so several are tried; the first that works is stored.
* **box map**: with the basket in the cell, the arm can hold the ball above it
  at the release height, keeps clear of the basket, and the basket does not
  intersect the robot. The hand tilt at that pose is stored too.

Maps are computed for the largest ball and basket the scenario ranges allow,
so smaller objects fit too. Heights are computed in parallel processes.
Re-run after changing the robot model, the table, the scenario ranges or the
hover heights (``python main.py maps``).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ..config import WORKSPACE_MAPS, Config  # noqa: E402
from ..env.controller import HOME, READY, tilt_from_vertical_deg, yaw_quat  # noqa: E402
from ..env.scene import Scene, worst_case_scenario  # noqa: E402

DEFAULT_HEIGHTS = tuple(round(0.25 + k * 0.02, 3) for k in range(-4, 5))   # 0.17 ... 0.33
X_RANGE = (-0.02, 0.62)
Y_RANGE = (-0.50, 0.50)
POS_TOL = 2e-3              # IK must land within 2 mm ...
ROT_TOL = 0.03              # ... and 0.03 rad of the vertical grasp orientation
MAX_REACH = 0.72            # skip cells farther than this from the shoulder
CLEARANCE_MARGIN = 0.03     # arm must stay this far from the basket at the release pose
YAWS_LEFT_DEG = [-90, -60, -120, -30, -150, 0, 180, 30, 150, 60, 120, 90]


class Mapper:
    def __init__(self, table_z: float, grid_res: float, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.grid_res = grid_res
        self.scene = Scene(self.cfg)
        self.m, self.d = self.scene.model, self.scene.data
        self.sc = worst_case_scenario(table_z, self.cfg)
        self.scene.restore_nominal()
        self.scene.apply_scenario(self.sc)
        self.scene.sc = self.sc
        m = self.m
        self.robot_body_gids = {
            g for g in range(m.ngeom)
            if m.geom_bodyid[g] == 0 and g != self.scene.table_geom
            and (m.geom_contype[g] or m.geom_conaffinity[g]) and not self.scene.distractor_mask[g]
        }

    def _set_arm(self, side, q):
        self.d.qpos[self.scene.qpos_adr[side]] = q

    def _place(self, ball_xyz, box_xy):
        s, d = self.scene, self.d
        d.qpos[s.ball_qpos:s.ball_qpos + 3] = ball_xyz
        d.qpos[s.ball_qpos + 3:s.ball_qpos + 7] = [1, 0, 0, 0]
        d.qpos[s.box_qpos:s.box_qpos + 3] = [box_xy[0], box_xy[1], self.sc.table_z]
        d.qpos[s.box_qpos + 3:s.box_qpos + 7] = [1, 0, 0, 0]

    def _arm_collides(self, side, ignore=()):
        mujoco.mj_forward(self.m, self.d)
        arm = self.scene.arm_geom_mask[side]
        for i in range(self.d.ncon):
            g1, g2 = self.d.contact.geom1[i], self.d.contact.geom2[i]
            if arm[g1] or arm[g2]:
                if (g2 if arm[g1] else g1) not in ignore:
                    return True
        return False

    def _solve(self, side, pos, quat=None, q_init=None, posture=None):
        ik = self.scene.ik
        ik.sync_from(self.d)
        r = ik.solve(side, pos, quat, q_init=q_init, posture=posture)
        return r.pos_error < POS_TOL and (quat is None or r.rot_error < ROT_TOL), r

    def ball_cell(self, side, x, y, yaws):
        sc, c = self.sc, self.cfg.control
        far = (5.0, 5.0)
        ball = np.array([x, y, sc.table_z + sc.ball_radius])
        for yaw in yaws:
            q = yaw_quat(yaw)
            ok, hover = self._solve(side, ball + [0, 0, c.pregrasp_hover], q, HOME[side])
            if not ok:
                continue
            ok, grasp = self._solve(side, ball, q, hover.q_arm)
            if not ok:
                continue
            ok, lift = self._solve(side, [x, y, sc.table_z + c.safe_height], None, grasp.q_arm,
                                   posture=grasp.q_arm)
            if not ok:
                continue
            self._place(ball, far)
            self._set_arm(side, grasp.q_arm)
            if self._arm_collides(side, ignore={self.scene.ball_geom}):
                continue
            self._place((x, y, -1.0), far)
            self._set_arm(side, lift.q_arm)
            if self._arm_collides(side):
                continue
            return yaw, lift.q_arm
        return None, None

    def box_cell(self, side, x, y, carry_seed):
        s = self.scene
        hover = np.array([x, y, self.sc.box_rim_z + self.cfg.control.preplace_hover])
        ok, r = self._solve(side, hover, None, carry_seed, posture=carry_seed)
        if not ok:
            return False, np.nan
        tilt = tilt_from_vertical_deg(s.ik.data.site_xmat[s.site_id[side]])
        self._place((5.0, 5.0, -1.0), (x, y))
        self._set_arm(side, r.q_arm)
        mujoco.mj_forward(self.m, self.d)
        box = s.box_geom_mask
        for i in range(self.d.ncon):
            g1, g2 = self.d.contact.geom1[i], self.d.contact.geom2[i]
            if (box[g1] and g2 in self.robot_body_gids) or (box[g2] and g1 in self.robot_body_gids):
                return False, tilt
        if self._arm_collides(side):
            return False, tilt
        return s.compute_basket_clearance(side) > CLEARANCE_MARGIN, tilt

    def run(self, side):
        mujoco.mj_resetData(self.m, self.d)
        other = "right" if side == "left" else "left"
        yaws = YAWS_LEFT_DEG if side == "left" else [-y for y in YAWS_LEFT_DEG]
        self._set_arm(side, HOME[side])
        self._set_arm(other, READY[other])
        for sd in ("left", "right"):
            self.scene.set_gripper(sd, 1.0, teleport=True)      # open fingers, as at the grasp
        mujoco.mj_forward(self.m, self.d)
        shoulder = self.d.xpos[self.m.body(f"openarm_{side}_link1").id].copy()
        xs = np.arange(X_RANGE[0], X_RANGE[1] + 1e-9, self.grid_res)
        ys = np.arange(Y_RANGE[0], Y_RANGE[1] + 1e-9, self.grid_res)
        ball_yaw = np.full((len(xs), len(ys)), np.nan)
        lift_q = {}
        t0 = time.time()
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                if np.linalg.norm(np.array([x, y, self.sc.table_z]) - shoulder) > MAX_REACH:
                    continue
                self._set_arm(other, READY[other])
                yaw, q = self.ball_cell(side, x, y, yaws)
                if yaw is not None:
                    ball_yaw[i, j] = yaw
                    lift_q[(i, j)] = q
        print(f"[h={self.sc.table_z:.2f} {side}] ball map: {np.isfinite(ball_yaw).sum()} cells "
              f"({time.time() - t0:.0f}s)", flush=True)
        box_ok = np.zeros((len(xs), len(ys)), dtype=bool)
        box_tilt = np.full((len(xs), len(ys)), np.nan)
        if not lift_q:
            print(f"[h={self.sc.table_z:.2f} {side}] WARNING: no graspable cells", flush=True)
            return xs, ys, ball_yaw, box_ok, box_tilt
        cells = np.argwhere(np.isfinite(ball_yaw))
        seed = lift_q[tuple(cells[np.argmin(np.linalg.norm(cells - cells.mean(0), axis=1))])]
        t0 = time.time()
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                if np.linalg.norm(np.array([x, y, self.sc.table_z]) - shoulder) > MAX_REACH:
                    continue
                self._set_arm(other, READY[other])
                box_ok[i, j], box_tilt[i, j] = self.box_cell(side, x, y, seed)
        print(f"[h={self.sc.table_z:.2f} {side}] box map: {box_ok.sum()} cells ({time.time() - t0:.0f}s)",
              flush=True)
        return xs, ys, ball_yaw, box_ok, box_tilt


def map_height(args):
    table_z, grid_res = args
    mapper = Mapper(table_z, grid_res)
    out = {}
    for side in ("left", "right"):
        xs, ys, ball_yaw, box_ok, box_tilt = mapper.run(side)
        out[f"{side}_ball_yaw"], out[f"{side}_box_ok"], out[f"{side}_box_tilt"] = ball_yaw, box_ok, box_tilt
    out["xs"], out["ys"] = xs, ys
    return table_z, out


def build_maps(heights=DEFAULT_HEIGHTS, grid_res: float = 0.02, workers: int | None = None,
               out_path=WORKSPACE_MAPS):
    workers = workers or max(1, min((os.cpu_count() or 4) - 2, 9))
    heights = sorted(float(h) for h in heights)
    print(f"mapping {len(heights)} heights {heights} at {grid_res * 100:.0f} cm with {workers} worker(s)",
          flush=True)
    t0 = time.time()
    if workers > 1 and len(heights) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(map_height, [(h, grid_res) for h in heights]))
    else:
        results = [map_height((h, grid_res)) for h in heights]
    results.sort(key=lambda r: r[0])
    out = {"heights": np.array([r[0] for r in results]), "xs": results[0][1]["xs"],
           "ys": results[0][1]["ys"], "grid_res": np.array(grid_res)}
    for key in ("left_ball_yaw", "left_box_ok", "left_box_tilt",
                "right_ball_yaw", "right_box_ok", "right_box_tilt"):
        out[key] = np.stack([r[1][key] for r in results])
    np.savez_compressed(out_path, **out)
    print(f"wrote {out_path} in {time.time() - t0:.0f}s")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--heights", default=",".join(f"{h:.3f}" for h in DEFAULT_HEIGHTS))
    ap.add_argument("--grid-res", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=str(WORKSPACE_MAPS))
    a = ap.parse_args()
    build_maps([float(h) for h in a.heights.split(",")], a.grid_res, a.workers, a.out)


if __name__ == "__main__":
    main()
