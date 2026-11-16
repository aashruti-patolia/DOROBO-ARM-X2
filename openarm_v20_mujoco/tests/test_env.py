"""Regression tests for the mirroring, reward-payout and success-metric fixes.

Run from the openarm_v20_mujoco directory:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import bimanual_env as be
from scripted_policy import scripted_action as _scripted_action
from reward_calculator import RewardCalculator


def _env():
    return be.BimanualPickPlaceEnv(domain_randomization=False, obs_noise_scale=0.0)


def _place(env, arm, ball_xy, box_xy):
    """Reset, then force a specific arm and object layout."""
    env.reset(seed=0)
    env._active_arm = "right" if arm == "left" else "left"   # reset() flips it
    env.reset(seed=0)
    assert env._active_arm == arm
    sign = 1.0 if arm == "left" else -1.0
    ball = np.array([ball_xy[0], sign * ball_xy[1], be.TABLE_Z + be.BALL_RADIUS])
    box = np.array([box_xy[0], sign * box_xy[1], be.TABLE_Z])
    env._ball_pos, env._box_pos = ball, box
    env._relocate_objects(ball, box)
    env._grasp_quat = be.yaw_quat(be.DEFAULT_GRASP_YAW[arm])   # mirrored yaws for both arms
    for _ in range(20):
        be.mujoco.mj_step(env.model, env.data)
    env.ik.sync_from(env.data)
    return env._get_obs()


def test_mirrored_arms_see_and_do_the_same_thing():
    """Same canonical action sequence on mirrored layouts => same observations.

    Fails if the action isn't un-mirrored for the right arm, or if the
    observation mirror (including the quaternion) is wrong.
    """
    left, right = _env(), _env()
    ol = _place(left, "left", (0.22, 0.15), (0.20, 0.07))
    orr = _place(right, "right", (0.22, 0.15), (0.20, 0.07))
    rng = np.random.default_rng(0)
    for _ in range(25):
        a = rng.uniform(-1, 1, 4).astype(np.float32)
        a[3] = 1.0  # keep the gripper open
        ol, *_ = left.step(a)
        orr, *_ = right.step(a)
    # skip one-hot arm id (idx 3-4) and joint terms, which differ by sign convention
    ee_pos, ee_quat = slice(19, 22), slice(22, 26)
    np.testing.assert_allclose(ol[ee_pos], orr[ee_pos], atol=2e-3)
    np.testing.assert_allclose(ol[ee_quat], orr[ee_quat], atol=5e-3)
    assert np.abs(ol[ee_pos]).max() > 0  # sanity: arm moved


def test_action_shape():
    env = _env()
    assert env.action_space.shape == (4,)
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape == (41,)


def test_release_and_success_bonuses_paid_once():
    rc = RewardCalculator()
    kw = dict(ee_pos=np.array([0.2, 0.1, 0.4]), ball_pos=np.array([0.2, 0.1, 0.3]),
              basket_pos=np.array([0.2, 0.1, 0.25]), ball_vel=np.zeros(6),
              finger_contacts=(False, False), basket_clearance=0.1,
              arm_basket_contact=False, gripper_open_ratio=1.0,
              action=np.zeros(4), prev_action=np.zeros(4), table_z=0.25,
              box_rim_z=0.31, phase=2)
    first = rc.compute(ball_in_box=True, ball_released=True, **kw)[1]
    second = rc.compute(ball_in_box=True, ball_released=True, **kw)[1]
    assert first["r_success"] > 0 and first["drop_r_release"] > 0 and first["drop_r_in_box"] > 0
    assert second["r_success"] == second["drop_r_release"] == second["drop_r_in_box"] == 0
    rc.reset()
    assert rc.compute(ball_in_box=True, ball_released=True, **kw)[1]["r_success"] > 0


def test_holding_ball_in_box_pays_nothing():
    rc = RewardCalculator()
    info = rc.compute(
        phase=2, ee_pos=np.array([0.2, 0.1, 0.4]), ball_pos=np.array([0.2, 0.1, 0.3]),
        basket_pos=np.array([0.2, 0.1, 0.25]), ball_vel=np.zeros(6),
        finger_contacts=(True, True), basket_clearance=0.1, arm_basket_contact=False,
        gripper_open_ratio=0.0, action=np.zeros(4), prev_action=np.zeros(4),
        table_z=0.25, box_rim_z=0.31, ball_in_box=True, ball_released=False)[1]
    assert info["r_success"] == 0 and info["drop_r_in_box"] == 0


def test_no_collision_penalty_outside_margin():
    rc = RewardCalculator()
    cfg = rc.cfg
    assert rc._collision_penalty(cfg.collision_margin, False) == pytest.approx(0.0, abs=1e-9)
    assert rc._collision_penalty(0.08, False) == 0.0
    assert rc._collision_penalty(0.0, False) == pytest.approx(-cfg.collision_exp_lambda)
    assert rc._collision_penalty(0.0, True) < rc._collision_penalty(0.0, False)


def test_timeout_is_not_a_success():
    env = _env()
    env.reset(seed=3)
    env._step_count = be.MAX_EPISODE_STEPS - 1
    _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert truncated and not terminated and info["is_success"] is False


@pytest.mark.parametrize("seed", list(range(6)))
def test_release_happens_clear_of_the_basket(seed):
    """At the moment the ball is released over the basket, no collision penalty applies.

    The old EE-to-box proxy charged ~-1/step at the drop pose, and
    mj_geomDistance reports a bogus 0 for mesh-vs-box; this pins the
    mesh-based clearance at the real release pose.
    """
    env = _env()
    env.reset(seed=seed)
    clearance_at_release = None
    for _ in range(be.MAX_EPISODE_STEPS):
        _, _, term, trunc, info = env.step(_scripted_action(env))
        if info["ball_released"] and clearance_at_release is None:
            clearance_at_release = info["basket_clearance"]
        if term or trunc:
            break
    assert clearance_at_release is not None, "ball was never released"
    assert clearance_at_release > env.reward_calc.cfg.collision_margin
    assert env.reward_calc._collision_penalty(clearance_at_release, False) == 0.0


def test_touching_the_basket_gives_zero_clearance():
    env = _env()
    env.reset(seed=0)
    d = env.data
    ee = env._get_ee_state()[0]
    # move the basket so one of its walls passes through the open fingers
    d.qpos[env._box_qpos_adr:env._box_qpos_adr + 3] = [ee[0], ee[1], ee[2] - 0.03]
    be.mujoco.mj_forward(env.model, d)
    assert env._compute_basket_clearance() < 0.005


# ---------------------------------------------------------------------------
# Physical arm motion, physical grasp, domain randomisation
# ---------------------------------------------------------------------------
def test_arm_is_driven_by_actuators_not_teleported():
    """After a step toward a far target the joints lag their servo targets."""
    env = _env()
    env.reset(seed=0)
    q_before = env.data.qpos[env._left_qpos_adr].copy()
    env.step(np.zeros(4, dtype=np.float32))
    q_after = env.data.qpos[env._left_qpos_adr]
    target = env.data.ctrl[env._left_actuators]
    assert np.abs(target - q_before).max() > 0.05      # the commanded move is large...
    assert np.abs(target - q_after).max() > 0.02       # ...and the arm has not jumped there
    assert np.abs(q_after - q_before).max() > 1e-4     # but it is moving toward it


def test_there_is_no_kinematic_ball_carry():
    assert not hasattr(be.BimanualPickPlaceEnv, "_carry_ball")


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_scripted_pick_and_place_succeeds_through_friction_grasp(seed):
    """Full episode with no teleporting or attachment: only friction holds the ball."""
    env = _env()
    env.reset(seed=seed)
    carried = []
    info = {}
    for _ in range(be.MAX_EPISODE_STEPS):
        _, _, term, trunc, info = env.step(_scripted_action(env))
        if info["ball_grasped"] and info["ball_pos"][2] > be.TABLE_Z + 0.06:
            carried.append(np.linalg.norm(info["ee_pos"] - info["ball_pos"]))
        if term or trunc:
            break
    assert info["is_success"], f"episode ended in phase {info['phase']}"
    assert len(carried) > 20, "ball was never actually lifted in the hand"
    assert max(carried) < be.GRASP_HOLD_DIST   # stayed in the fingers throughout the carry


def test_dropped_ball_sends_phase_back_to_approach():
    env = _env()
    env.reset(seed=0)
    env._phase = be.PHASE_DROP                   # pretend we were carrying it...
    env._ball_grasped = False                    # ...and it slipped out
    far = env.data.xpos[env._ball_body_id] + np.array([0.0, 0.0, 0.2])
    env._update_phase(far, env.data.xpos[env._ball_body_id].copy(), far, (False, False), False)
    assert env._phase == be.PHASE_APPROACH


def test_posture_hold_keeps_the_hand_from_tilting():
    """Position-only IK pulled toward home tips the hand; holding posture does not."""
    env = _env()
    env.reset(seed=0)
    d = env.data
    q0 = d.qpos[env._left_qpos_adr].copy()
    r = env.ik.solve_left(target_pos=[0.22, 0.2, 0.272], target_quat=be.yaw_quat(-90), q_init=q0)
    q_grasp = r.q_arm
    tgt = np.array([0.24, 0.12, 0.40])
    tilts = {}
    for name, posture in (("home", None), ("hold", q_grasp)):
        res = env.ik.solve_left(target_pos=tgt, q_init=q_grasp, posture=posture)
        z = env.ik.data.site_xmat[env.ik._left_site_id].reshape(3, 3)[2, 2]
        tilts[name] = np.degrees(np.arccos(np.clip(z, -1, 1)))
        assert res.pos_error < 2e-3
    assert tilts["hold"] < tilts["home"] - 10


def test_domain_randomization_never_compounds_and_can_be_turned_off():
    env = be.BimanualPickPlaceEnv(domain_randomization=True, obs_noise_scale=0.0)
    env.scenario_level = "off"      # isolate the physics randomisation from scenario geometry
    m, nom = env.model, env._nominal
    for ep in range(8):
        env.reset(seed=ep)
        scale = m.actuator_gainprm[:, 0] / nom["actuator_gainprm"][:, 0]
        arm_acts = env._dr_actuators
        assert scale[arm_acts].min() >= be.DR_ACTUATOR_KP[0] - 1e-9
        assert scale[arm_acts].max() <= be.DR_ACTUATOR_KP[1] + 1e-9
        mass_ratio = m.body_mass[env._ball_body_id] / nom["body_mass"][env._ball_body_id]
        assert be.DR_BALL_MASS[0] - 1e-9 <= mass_ratio <= be.DR_BALL_MASS[1] + 1e-9
        assert be.DR_LATENCY_STEPS[0] <= env._action_delay <= be.DR_LATENCY_STEPS[1]
    env.domain_randomization = False
    env.reset(seed=99)
    for key, attr in (("actuator_gainprm", "actuator_gainprm"), ("actuator_biasprm", "actuator_biasprm"),
                      ("dof_damping", "dof_damping"), ("geom_friction", "geom_friction"),
                      ("body_mass", "body_mass"), ("body_inertia", "body_inertia")):
        np.testing.assert_array_equal(getattr(m, attr), nom[key])
    assert env._action_delay == 0 and not env._perception_bias["ball"].any()


def test_command_latency_delays_the_executed_action():
    env = _env()
    env.reset(seed=0)
    env._action_delay = 2
    env._action_queue = [env._NEUTRAL_ACTION.copy() for _ in range(2)]
    marker = np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float32)
    env.step(marker)                               # step 1: the robot still runs the neutral command
    assert env._gripper_open_ratio == pytest.approx(1.0)
    env.step(np.zeros(4, dtype=np.float32))        # step 2: still neutral
    assert env._gripper_open_ratio == pytest.approx(1.0)
    env.step(np.zeros(4, dtype=np.float32))        # step 3: the marker (close gripper) arrives
    assert env._gripper_open_ratio == pytest.approx(0.0)


def test_perception_is_exact_without_randomization_and_noisy_with_it():
    clean = _env()
    clean.reset(seed=0)
    true = clean.data.xpos[clean._ball_body_id]
    np.testing.assert_array_equal(clean._perceive("ball", true), true)
    noisy = be.BimanualPickPlaceEnv(domain_randomization=True, obs_noise_scale=0.0)
    noisy.reset(seed=0)
    est = np.array([noisy._perceive("ball", true) for _ in range(200)])
    assert 0.0005 < est.std(axis=0).max() < 0.005          # jitters by about DR_PERCEPTION_NOISE
    assert np.linalg.norm(est.mean(axis=0) - true) < 0.03  # and is not wildly off


def test_grasp_holds_only_because_of_friction():
    """With frictionless pads and ball the ball must slip out and the task fail.

    A kinematic attachment (or a weld) would carry the ball regardless of
    friction, so this is what distinguishes a physical grasp from a fake one.
    """
    env = _env()
    env.reset(seed=0)
    m = env.model
    m.geom_friction[env._ball_geom_id, :] = 0.01
    for g in env._finger_pad_geoms:
        m.geom_friction[g, :] = 0.01
    info = {}
    for _ in range(be.MAX_EPISODE_STEPS):
        _, _, term, trunc, info = env.step(_scripted_action(env))
        if term or trunc:
            break
    assert not info["is_success"]


# ---------------------------------------------------------------------------
# Placement over the arms' full reach
# ---------------------------------------------------------------------------
def _placements(n=300):
    env = _env()
    seps, balls, boxes = [], [], []
    tab = env.model.geom("workspace_table").id
    (cx, cy, _), (hx, hy, _) = env.model.geom_pos[tab], env.model.geom_size[tab]
    for ep in range(n):
        env.reset(seed=ep)
        b = env.data.xpos[env._ball_body_id].copy()
        x = env.data.xpos[env._box_body_id].copy()
        # every object rests fully on the table top
        assert abs(b[0] - cx) <= hx - be.BALL_RADIUS and abs(b[1] - cy) <= hy - be.BALL_RADIUS
        assert abs(x[0] - cx) <= hx - be.BOX_HALF_SIZE and abs(x[1] - cy) <= hy - be.BOX_HALF_SIZE
        assert abs(b[2] - (be.TABLE_Z + be.BALL_RADIUS)) < 5e-3 and abs(x[2] - be.TABLE_Z) < 5e-3
        if env._active_arm == "right":          # mirror into the left-arm frame
            b[1], x[1] = -b[1], -x[1]
        seps.append(np.linalg.norm((b - x)[:2])); balls.append(b); boxes.append(x)
    return np.array(seps), np.array(balls), np.array(boxes)


def test_placement_uses_reachability_maps():
    env = _env()
    assert env._placements is not None, "workspace_maps.npz missing: run tools/compute_workspace.py"


def test_ball_and_basket_are_far_apart():
    seps, balls, boxes = _placements()
    assert seps.min() > 0.30                                   # old layout: 0.125-0.22 m
    assert np.median(seps) > 0.38


def test_placements_still_vary_from_episode_to_episode():
    """Maximum distance must not mean the same two spots every time."""
    _, balls, boxes = _placements()
    assert len({tuple(np.round(b[:2], 3)) for b in balls}) > 100
    assert len({tuple(np.round(b[:2], 3)) for b in boxes}) > 100
    assert balls[:, :2].std(axis=0).max() > 0.02 and boxes[:, :2].std(axis=0).max() > 0.02


def test_ball_dropped_out_of_reach_ends_the_episode():
    env = _env()
    env.reset(seed=0)
    assert env._grasp_yaw_at(np.array([0.55, 0.0])) is None     # far front edge: not graspable
    assert env._grasp_yaw_at(env.data.xpos[env._ball_body_id][:2]) is not None
