"""Low-level control (scripted): IK, mirroring, Cartesian control and its safety limits."""
import mujoco
import numpy as np
import pytest

from conftest import BASIC, stage
from dual_arm_rl.env.controller import (FINGERS_DOWN_SOFT, MIRROR_JOINT_SIGNS, READY, CartesianController,
                                        SafetyBox, tilt_from_vertical_deg, yaw_quat)


def test_mirror_signs_map_left_poses_onto_right_poses(scene):
    """One policy serves both arms through Y-mirroring; joint observations rely on this map."""
    ik, rng = scene.ik, np.random.default_rng(0)
    M = np.diag([1.0, -1.0, 1.0])
    for _ in range(20):
        ql = np.clip(READY["left"] + rng.uniform(-0.5, 0.5, 7), ik.lower["left"] + 0.05, ik.upper["left"] - 0.05)
        pl, Rl = ik.site_pose("left", ql)
        pr, Rr = ik.site_pose("right", MIRROR_JOINT_SIGNS * ql)
        np.testing.assert_allclose(pr, M @ pl, atol=1e-9)
        np.testing.assert_allclose(Rr, M @ Rl @ M, atol=1e-9)


def test_ready_pose_is_above_every_table_and_out_of_the_camera_view(scene):
    cloud = scene._build_home_cloud()
    assert cloud[:, 2].min() > 0.33 + 0.10                    # tallest table + margin
    assert scene.camera_visible(cloud, 0).mean() < 0.10       # hands do not block the view


def _controller(scene):
    return CartesianController(scene.ik, scene.actuators, lead_max=0.03, iterations=8,
                               max_joint_step=0.15, max_rot_step=0.08)


def _at_hover(scene, cfg, rng):
    world = scene.reset(rng, stage(cfg, BASIC))
    side = world["side"]
    yaw = scene.grasp_yaw_at(side, world["ball_pos"])
    q = yaw_quat(yaw)
    target = world["ball_pos"] + [0, 0, 0.06]
    r = scene.ik.solve(side, target, q)
    scene.data.qpos[scene.qpos_adr[side]] = r.q_arm
    scene.data.ctrl[scene.actuators[side]] = r.q_arm
    mujoco.mj_forward(scene.model, scene.data)
    return side, q


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_delta_command_moves_the_pinch_point_by_that_much(scene, cfg, rng, axis):
    side, q = _at_hover(scene, cfg, rng)
    ctl = _controller(scene)
    ctl.reset(side, scene.data)
    p0 = scene.pinch_pos(side)
    delta = np.zeros(3)
    delta[axis] = 0.01
    for _ in range(4):
        ctl.command(scene.data, delta=delta, orientation=q)
        scene.step_physics(cfg.control.substeps)
    for _ in range(6):                                         # let the servos settle
        ctl.command(scene.data, delta=np.zeros(3), orientation=q)
        scene.step_physics(cfg.control.substeps)
    moved = scene.pinch_pos(side) - p0
    assert moved[axis] == pytest.approx(0.04, abs=0.008)
    assert np.linalg.norm(np.delete(moved, axis)) < 0.008
    assert scene.hand_tilt_deg(side) < 8                      # fingers kept pointing down


def test_the_safety_box_holds_the_target_and_reports_the_push(scene, cfg, rng):
    side, q = _at_hover(scene, cfg, rng)
    ctl = _controller(scene)
    ctl.reset(side, scene.data)
    p = scene.pinch_pos(side)
    box = SafetyBox(lo=p - 0.02, hi=p + 0.02)
    clipped = ctl.command(scene.data, delta=np.array([0.0, 0.0, -0.5]), box=box, orientation=q)
    assert clipped > 0.4
    assert np.all(ctl.target >= box.lo - 1e-9) and np.all(ctl.target <= box.hi + 1e-9)


def test_joint_commands_never_jump(scene, cfg, rng):
    """A far target must not make the servos lunge (the arm used to whip around)."""
    side, q = _at_hover(scene, cfg, rng)
    ctl = _controller(scene)
    ctl.reset(side, scene.data)
    q_now = scene.data.qpos[scene.qpos_adr[side]].copy()
    ctl.command(scene.data, target=scene.pinch_pos(side) + [0.3, -0.2, 0.2], orientation=yaw_quat(0.0))
    assert np.abs(scene.data.ctrl[scene.actuators[side]] - q_now).max() <= 0.15 + 1e-9


def test_soft_fingers_down_keeps_position_first(scene, cfg, rng):
    """At carry/release heights a vertical hand is not always reachable: the soft
    mode always reaches the position and keeps the hand as vertical as it can."""
    tilts = []
    for _ in range(3):
        world = scene.reset(rng, stage(cfg, BASIC))
        side = world["side"]
        cells = scene.cells[side]["box_xy"]
        for xy in cells[:: max(1, len(cells) // 12)]:
            target = [xy[0], xy[1], scene.sc.box_rim_z + 0.08]
            r = scene.ik.solve(side, target, FINGERS_DOWN_SOFT)
            assert r.pos_error < 0.005
            tilts.append(tilt_from_vertical_deg(scene.ik.data.site_xmat[scene.site_id[side]]))
    assert np.median(tilts) < 10


def test_ready_pose_points_the_fingers_down(scene):
    for side in ("left", "right"):
        _, R = scene.ik.site_pose(side, READY[side])
        assert tilt_from_vertical_deg(R.flatten()) < 1.0


def test_joint_move_is_a_smooth_monotonic_s_curve():
    from dual_arm_rl.env.controller import JointMove
    q0, q1, q2 = np.zeros(7), np.full(7, 0.4), np.full(7, 1.0)
    m = JointMove([q0, q1, q2], dt_tick=0.04, joint_speed=1.0)
    qs = np.array([m.next_setpoint() for _ in range(m.n_ticks)])
    assert m.done and np.allclose(qs[-1], q2)
    steps = np.diff(np.vstack([q0, qs]), axis=0)[:, 0]
    assert np.all(steps >= -1e-12)                              # never reverses
    assert steps.max() / 0.04 <= 1.0 + 1e-6                     # respects the joint speed
    assert steps[0] < steps[len(steps) // 2]                    # starts gently


def test_solve_closest_prefers_the_nearest_solution(scene, cfg, rng):
    world = scene.reset(rng, stage(cfg, BASIC))
    side = world["side"]
    q = yaw_quat(scene.grasp_yaw_at(side, world["ball_pos"]))
    target = world["ball_pos"] + [0, 0, 0.06]
    near = scene.ik.solve(side, target, q).q_arm
    r = scene.ik.solve_closest(side, target, q, near)
    assert r is not None and np.abs(r.q_arm - near).sum() < 0.05


def test_scripted_transit_moves_the_joints_efficiently(cfg):
    """The old Cartesian transit swung joint 1 back and forth (2-3 rad travelled for
    0.1-0.7 rad of net change); the joint-space S-curve moves straight."""
    from dual_arm_rl.env.pick_place_env import PickPlaceEnv
    env = PickPlaceEnv(cfg, stage=BASIC, seed=3)
    wasted = []
    for k in range(4):
        qs, orig_tick, orig_reset = [], env._tick, env.scene.reset

        def tick():
            orig_tick()
            qs.append(env.data.qpos[env.scene.qpos_adr[env._side]].copy())

        def scene_reset(*a, **kw):                     # a re-generated world restarts the record
            qs.clear()
            return orig_reset(*a, **kw)
        env._tick, env.scene.reset = tick, scene_reset
        env.reset(seed=50 + k, options={"start_skill": "grasp", "hover_start": False})
        env._tick, env.scene.reset = orig_tick, orig_reset
        Q = np.array(qs)
        travel, net = np.abs(np.diff(Q, axis=0)).sum(), np.abs(Q[-1] - Q[0]).sum()
        wasted.append(1 - net / travel)
        assert env.scene.hand_tilt_deg(env._side) < 6
    env.close()
    assert np.median(wasted) < 0.25, wasted
