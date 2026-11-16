#!/usr/bin/env python3
"""
Compute where each arm can pick up a ball and drop it into the basket, for
every table height the scenarios can produce.

Writes ``workspace_maps.npz`` next to ``bimanual_env.py``. The environment
samples ball and basket placements from these maps, so placements cover
everything an arm can actually do instead of a hand-picked rectangle.

For every table height and every cell of a grid over the table (both arms,
computed separately):

* **ball map** - the arm can reach the ball with the gripper pointing straight
  down (needed for a pinch grasp) at the ball's height *and* at the approach
  hover height, can then lift it, and the grasp pose collides with nothing
  but the ball. The hand yaw is free for a sphere, so several yaws are tried
  and the first that works is stored per cell.
* **box map** - with the basket at the cell, the arm can hold the ball above
  it at the drop height, keeps its geometry outside the basket clearance
  margin, and the basket itself does not intersect the robot. The hand's tilt
  from vertical at that pose is stored too: a strongly tipped hand lets the
  ball slide out, so the environment only uses cells below its tilt limit.

Maps are computed for the *largest* ball and basket the scenario ranges
allow (``bimanual_env.worst_case_scenario``), so every smaller object is also
placed somewhere the arm can handle.

Heights are independent, so they are computed in parallel processes.
Re-run this after changing the robot model, the table, the scenario ranges,
or the drop/approach constants in ``bimanual_env.py``:

    python tools/compute_workspace.py
    python tools/compute_workspace.py --heights 0.21,0.25,0.29 --grid-res 0.03
"""
from __future__ import annotations

import os

# One BLAS thread per process: the workers already use every core.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bimanual_env as be  # noqa: E402
from inverse_kinematics import LEFT_ARM_HOME, RIGHT_ARM_HOME  # noqa: E402

DEFAULT_HEIGHTS = tuple(round(be.TABLE_Z + k * 0.02, 3) for k in range(-4, 5))   # 0.17 ... 0.33

# Grid over the region an arm could conceivably reach at table height.
X_RANGE = (-0.02, 0.62)
Y_RANGE = (-0.50, 0.50)

POS_TOL = 2e-3          # IK must land within 2 mm ...
ROT_TOL = 0.03          # ... and 0.03 rad of the vertical grasp orientation
MAX_REACH = 0.72        # shoulder-to-point distance beyond which we skip the cell

# Hand yaws to try, in order of preference (left arm; mirrored for the right).
YAWS_LEFT_DEG = [-90, -60, -120, -30, -150, 0, 180, 30, 150, 60, 120, 90]


def yaw_quat(deg: float) -> np.ndarray:
    return be.yaw_quat(deg)


class Mapper:
    def __init__(self, table_z: float, grid_res: float):
        self.grid_res = grid_res
        self.env = be.BimanualPickPlaceEnv(domain_randomization=False, obs_noise_scale=0.0)
        self.m, self.d = self.env.model, self.env.data
        m = self.m
        # The world for this height, with the largest ball and basket.
        self.sc = be.worst_case_scenario(table_z)
        self.env._restore_nominal()
        self.env._apply_scenario(self.sc)
        # Every fixed collidable geom that is not part of the table is "robot"
        table = {g for g in range(m.ngeom)
                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("workspace_table")}
        self.robot_body_gids = {
            g for g in range(m.ngeom)
            if m.geom_bodyid[g] == 0 and g not in table
            and (m.geom_contype[g] or m.geom_conaffinity[g])
            and not self.env._distractor_mask[g]
        }

    # -- helpers -------------------------------------------------------
    def _set_arm(self, side, q):
        adr = self.env._left_qpos_adr if side == "left" else self.env._right_qpos_adr
        self.d.qpos[adr] = q

    def _place(self, ball_xyz, box_xy):
        e = self.env
        self.d.qpos[e._ball_qpos_adr:e._ball_qpos_adr + 3] = ball_xyz
        self.d.qpos[e._ball_qpos_adr + 3:e._ball_qpos_adr + 7] = [1, 0, 0, 0]
        self.d.qpos[e._box_qpos_adr:e._box_qpos_adr + 3] = [box_xy[0], box_xy[1], self.sc.table_z]
        self.d.qpos[e._box_qpos_adr + 3:e._box_qpos_adr + 7] = [1, 0, 0, 0]

    def _arm_collides(self, side, ignore=()):
        """True if the active arm touches anything except geoms in `ignore`."""
        mujoco.mj_forward(self.m, self.d)
        arm = self.env._arm_geom_mask[side]
        for i in range(self.d.ncon):
            g1, g2 = self.d.contact.geom1[i], self.d.contact.geom2[i]
            if arm[g1] or arm[g2]:
                other = g2 if arm[g1] else g1
                if other in ignore:
                    continue
                return True
        return False

    def _solve(self, side, pos, quat=None, q_init=None, posture=None):
        ik = self.env.ik
        ik.sync_from(self.d)
        solve = ik.solve_left if side == "left" else ik.solve_right
        r = solve(target_pos=pos, target_quat=quat, q_init=q_init, posture=posture)
        ok = r.pos_error < POS_TOL and (quat is None or r.rot_error < ROT_TOL)
        return ok, r

    # -- ball ----------------------------------------------------------
    def ball_cell(self, side, x, y, home, yaws):
        sc = self.sc
        far = (5.0, 5.0)
        ball = np.array([x, y, sc.table_z + sc.ball_radius])
        for yaw in yaws:
            q = yaw_quat(yaw)
            ok, hover = self._solve(side, ball + [0, 0, be.APPROACH_HOVER_HEIGHT], q, home)
            if not ok:
                continue
            ok, grasp = self._solve(side, ball, q, hover.q_arm)
            if not ok:
                continue
            ok, lift = self._solve(side, [x, y, sc.table_z + 0.170], None,
                                   grasp.q_arm, posture=grasp.q_arm)
            if not ok:
                continue
            # Collision check at the grasp pose (ball present, basket far away)
            self._place(ball, far)
            self._set_arm(side, grasp.q_arm)
            if self._arm_collides(side, ignore={self.env._ball_geom_id}):
                continue
            # ...and at the lift pose (no ball in the scene to collide with)
            self._place((x, y, -1.0), far)
            self._set_arm(side, lift.q_arm)
            if self._arm_collides(side):
                continue
            return yaw, lift.q_arm
        return None, None

    # -- box -----------------------------------------------------------
    def box_cell(self, side, x, y, carry_seed):
        e = self.env
        hover = np.array([x, y, self.sc.box_rim_z + be.DROP_HOVER_HEIGHT])
        ok, r = self._solve(side, hover, None, carry_seed, posture=carry_seed)
        if not ok:
            return False, np.nan
        # The solver's scratch data now holds the hover pose: how tipped is the hand?
        # (Stored per cell; the environment decides how much tilt it accepts.)
        site = e.ik._left_site_id if side == "left" else e.ik._right_site_id
        tilt = float(np.degrees(np.arccos(np.clip(e.ik.data.site_xmat[site][8], -1.0, 1.0))))
        self._place((5.0, 5.0, -1.0), (x, y))
        self._set_arm(side, r.q_arm)
        mujoco.mj_forward(self.m, self.d)
        # Basket must not intersect the robot
        box = e._box_geom_mask
        for i in range(self.d.ncon):
            g1, g2 = self.d.contact.geom1[i], self.d.contact.geom2[i]
            if (box[g1] and g2 in self.robot_body_gids) or (box[g2] and g1 in self.robot_body_gids):
                return False, tilt
        if self._arm_collides(side):
            return False, tilt
        e._active_arm = side
        return e._compute_basket_clearance() > e.reward_calc.cfg.collision_margin, tilt

    # -- driver --------------------------------------------------------
    def run(self, side):
        e = self.env
        e.reset(seed=0)
        e._restore_nominal()
        e._apply_scenario(self.sc)           # reset() applied a nominal scenario; redo ours
        home = LEFT_ARM_HOME if side == "left" else RIGHT_ARM_HOME
        other_home = RIGHT_ARM_HOME if side == "left" else LEFT_ARM_HOME
        yaws = YAWS_LEFT_DEG if side == "left" else [-y for y in YAWS_LEFT_DEG]
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
                # other arm parked at home throughout
                self._set_arm("right" if side == "left" else "left", other_home)
                yaw, q = self.ball_cell(side, x, y, home, yaws)
                if yaw is not None:
                    ball_yaw[i, j] = yaw
                    lift_q[(i, j)] = q
        print(f"[h={self.sc.table_z:.2f} {side}] ball map: {np.isfinite(ball_yaw).sum()} cells "
              f"({time.time()-t0:.0f}s)", flush=True)

        box_ok = np.zeros((len(xs), len(ys)), dtype=bool)
        box_tilt = np.full((len(xs), len(ys)), np.nan)
        if not lift_q:
            print(f"[h={self.sc.table_z:.2f} {side}] WARNING: no graspable cells", flush=True)
            return xs, ys, ball_yaw, box_ok, box_tilt
        # Carry seed: the lift pose of the most central reachable ball cell.
        cells = np.argwhere(np.isfinite(ball_yaw))
        centre = cells[np.argmin(np.linalg.norm(cells - cells.mean(0), axis=1))]
        seed = lift_q[tuple(centre)]
        t0 = time.time()
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                if np.linalg.norm(np.array([x, y, self.sc.table_z]) - shoulder) > MAX_REACH:
                    continue
                self._set_arm("right" if side == "left" else "left", other_home)
                box_ok[i, j], box_tilt[i, j] = self.box_cell(side, x, y, seed)
        print(f"[h={self.sc.table_z:.2f} {side}] box map: {box_ok.sum()} cells ({time.time()-t0:.0f}s)",
              flush=True)
        return xs, ys, ball_yaw, box_ok, box_tilt


def map_height(args):
    """Worker: both arms' maps for one table height."""
    table_z, grid_res = args
    mapper = Mapper(table_z, grid_res)
    out = {}
    for side in ("left", "right"):
        xs, ys, ball_yaw, box_ok, box_tilt = mapper.run(side)
        out[f"{side}_ball_yaw"], out[f"{side}_box_ok"] = ball_yaw, box_ok
        out[f"{side}_box_tilt"] = box_tilt
    out["xs"], out["ys"] = xs, ys
    return table_z, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--heights", default=",".join(f"{h:.3f}" for h in DEFAULT_HEIGHTS),
                    help="comma-separated table-top heights in metres")
    ap.add_argument("--grid-res", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=max(1, min((os.cpu_count() or 4) - 2, 9)))
    ap.add_argument("--out", default=str(ROOT / "workspace_maps.npz"))
    args = ap.parse_args()

    heights = sorted(float(h) for h in args.heights.split(","))
    print(f"mapping {len(heights)} heights {heights} at {args.grid_res*100:.0f} cm "
          f"with {args.workers} worker(s)", flush=True)
    t0 = time.time()
    if args.workers > 1 and len(heights) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(map_height, [(h, args.grid_res) for h in heights]))
    else:
        results = [map_height((h, args.grid_res)) for h in heights]
    results.sort(key=lambda r: r[0])

    out = {"heights": np.array([r[0] for r in results]),
           "xs": results[0][1]["xs"], "ys": results[0][1]["ys"],
           "grid_res": np.array(args.grid_res)}
    for key in ("left_ball_yaw", "left_box_ok", "left_box_tilt",
                "right_ball_yaw", "right_box_ok", "right_box_tilt"):
        out[key] = np.stack([r[1][key] for r in results])
    np.savez_compressed(args.out, **out)

    print(f"\nwrote {args.out} in {time.time()-t0:.0f}s")
    print("height  | left ball  left box | right ball right box   (cells)")
    for k, h in enumerate(out["heights"]):
        print(f" {h:.2f}   | {np.isfinite(out['left_ball_yaw'][k]).sum():8d} {out['left_box_ok'][k].sum():8d} |"
              f" {np.isfinite(out['right_ball_yaw'][k]).sum():9d} {out['right_box_ok'][k].sum():8d}")


if __name__ == "__main__":
    main()
