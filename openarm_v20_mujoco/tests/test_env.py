"""Regression tests for the mirroring, reward-payout and success-metric fixes.

Run from the openarm_v20_mujoco directory:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bimanual_env as be
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


@pytest.mark.parametrize("seed", list(range(12)))
def test_drop_hover_pose_is_not_penalised(seed):
    """The pose the drop phase aims for (DROP_HOVER_HEIGHT above the rim) must be 'clear'.

    The old EE-to-box proxy charged ~-1/step here; mj_geomDistance reports a
    bogus 0 for mesh-vs-box. This pins the mesh-based clearance to a sane value.
    """
    env = _env()
    env.reset(seed=seed)
    d, side = env.data, env._active_arm
    tgt = d.xpos[env._box_body_id].copy()
    tgt[2] = be.BOX_RIM_Z + be.DROP_HOVER_HEIGHT
    adr = env._left_qpos_adr if side == "left" else env._right_qpos_adr
    solve = env.ik.solve_left if side == "left" else env.ik.solve_right
    res = solve(target_pos=tgt, q_init=d.qpos[adr])
    assert res.pos_error < 2e-3
    d.qpos[adr] = res.q_arm
    be.mujoco.mj_forward(env.model, d)
    c = env._compute_basket_clearance()
    assert c > env.reward_calc.cfg.collision_margin
    assert env.reward_calc._collision_penalty(c, False) == 0.0


def test_touching_the_basket_gives_zero_clearance():
    env = _env()
    env.reset(seed=0)
    d, side = env.data, env._active_arm
    tgt = d.xpos[env._box_body_id].copy()
    tgt[2] = be.BOX_RIM_Z + 0.005          # push the gripper down into the walls
    adr = env._left_qpos_adr if side == "left" else env._right_qpos_adr
    solve = env.ik.solve_left if side == "left" else env.ik.solve_right
    d.qpos[adr] = solve(target_pos=tgt, q_init=d.qpos[adr]).q_arm
    be.mujoco.mj_forward(env.model, d)
    assert env._compute_basket_clearance() < 0.005
