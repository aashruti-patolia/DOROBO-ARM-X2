"""What counts as success, what counts as failure, and how far apart things are placed.

1. Success is shown only when the released ball rests inside the upright basket.
   A flipped basket, or a ball beside it, is never a success.
2. The gripper must not touch the basket: touching it fails the episode and is
   penalised in full from the first training step.
3. Ball and basket are placed as far apart as the active arm can reach.

Run from the openarm_v20_mujoco directory:  python -m pytest tests/test_success_rules.py -q
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import bimanual_env as be  # noqa: E402
from reward_calculator import RewardCalculator  # noqa: E402
from scripted_policy import scripted_action  # noqa: E402

OPEN = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _env():
    return be.BimanualPickPlaceEnv(randomization_level="off", obs_noise_scale=0.0)


def _released_ball_at(env, offset, box_quat=(1, 0, 0, 0), box_lift=0.0):
    """Put the ball at box_centre + offset as if it had just been dropped."""
    d = env.data
    box = d.xpos[env._box_body_id].copy()
    d.qpos[env._box_qpos_adr + 2] += box_lift
    d.qpos[env._box_qpos_adr + 3:env._box_qpos_adr + 7] = box_quat
    d.qpos[env._ball_qpos_adr:env._ball_qpos_adr + 3] = box + np.asarray(offset, dtype=float)
    d.qvel[:] = 0
    mujoco.mj_forward(env.model, d)
    env._phase = be.PHASE_DROP
    env._ball_grasped = False
    env._ball_released = True


def _run(env, steps):
    """Step with the gripper open; returns (ever_success, last_info, total_reward, steps_taken)."""
    ever, total, info = False, 0.0, {}
    for k in range(steps):
        _, r, term, trunc, info = env.step(OPEN)
        total += r
        ever |= bool(info["is_success"])
        if term or trunc:
            return ever, info, total, k + 1
    return ever, info, total, steps


# ---------------------------------------------------------------------------
# 1. Success is only a ball at rest inside the upright basket
# ---------------------------------------------------------------------------
def test_ball_resting_in_the_upright_basket_is_a_success():
    env = _env()
    env.reset(seed=0)
    _released_ball_at(env, [0, 0, 0.035])
    ever, info, _, n = _run(env, 60)
    assert ever and info["is_success"] and info["failure"] is None
    assert n >= be.SUCCESS_HOLD_STEPS            # had to stay there, not just pass through


def test_ball_beside_the_basket_is_not_a_success():
    env = _env()
    env.reset(seed=0)
    sc = env._sc
    _released_ball_at(env, [sc.box_half + sc.ball_radius + 0.01, 0, sc.ball_radius])
    ever, info, _, _ = _run(env, 60)
    assert not ever and not info["is_success"]


def test_ball_balanced_above_the_rim_is_not_a_success():
    env = _env()
    env.reset(seed=0)
    env._ball_released = True
    env._phase = be.PHASE_DROP
    d = env.data
    box = d.xpos[env._box_body_id]
    above_rim = box + [0, 0, env._sc.box_depth + 0.03]
    assert not env._check_ball_in_box(above_rim, box)


def test_flipped_basket_is_a_failure_even_with_the_ball_at_its_centre():
    env = _env()
    env.reset(seed=0)
    # basket turned upside down over the ball (the ball is "at the box centre")
    _released_ball_at(env, [0, 0, 0.0], box_quat=(0, 1, 0, 0), box_lift=env._sc.box_depth)
    d = env.data
    assert env._box_tilt_deg() > 150
    assert not env._check_ball_in_box(d.xpos[env._ball_body_id], d.xpos[env._box_body_id])
    ever, info, total, n = _run(env, 30)
    assert not ever and not info["is_success"]
    assert info["failure"] == "box_flipped"
    assert n == 1                                 # the episode ends there


@pytest.mark.parametrize("tilt_deg", [35, 60, 90])
def test_tipped_basket_never_counts(tilt_deg):
    env = _env()
    env.reset(seed=0)
    half = np.radians(tilt_deg) / 2
    _released_ball_at(env, [0, 0, 0.03], box_quat=(np.cos(half), np.sin(half), 0, 0))
    _, r, term, _, info = env.step(OPEN)
    assert term and not info["is_success"] and info["failure"] == "box_flipped"
    assert info["r_failure"] < 0 and info["r_success"] == 0


def test_ball_in_basket_while_still_held_is_not_a_success():
    env = _env()
    env.reset(seed=0)
    _released_ball_at(env, [0, 0, 0.035])
    env._ball_released = False                    # never let go
    ever, info, _, _ = _run(env, 30)
    assert not ever


def test_success_pays_the_bonus_and_failure_pays_the_penalty_once():
    rc = RewardCalculator()
    kw = dict(phase=2, ee_pos=np.array([0.2, 0.1, 0.45]), ball_pos=np.array([0.2, 0.1, 0.28]),
              basket_pos=np.array([0.2, 0.1, 0.25]), ball_vel=np.zeros(6),
              finger_contacts=(False, False), basket_clearance=0.1, gripper_open_ratio=1.0,
              action=np.zeros(4), prev_action=np.zeros(4), table_z=0.25, box_rim_z=0.31)
    ok = rc.compute(arm_basket_contact=False, ball_in_box=True, ball_released=True, **kw)[1]
    assert ok["r_success"] > 0 and ok["r_failure"] == 0
    rc.reset()
    bad = rc.compute(arm_basket_contact=True, ball_in_box=True, ball_released=True,
                     task_failed=True, **kw)[1]
    assert bad["r_success"] == 0                  # no success bonus on a failed episode
    assert bad["r_failure"] == rc.cfg.failure_penalty < 0


# ---------------------------------------------------------------------------
# 2. The gripper must not touch the basket
# ---------------------------------------------------------------------------
def test_gripper_touching_the_basket_fails_the_episode():
    env = _env()
    env.reset(seed=0)
    d = env.data
    ee = env._get_ee_state()[0]
    # put a basket wall through the open fingers
    d.qpos[env._box_qpos_adr:env._box_qpos_adr + 3] = [ee[0], ee[1], ee[2] - 0.03]
    d.qvel[:] = 0
    mujoco.mj_forward(env.model, d)
    assert env._check_arm_box_contact()
    _, r, term, trunc, info = env.step(OPEN)
    assert term and not info["is_success"]
    assert info["failure"] == "gripper_touched_box"
    assert info["r_failure"] == env.reward_calc.cfg.failure_penalty
    assert r < -20                                 # failure penalty plus the contact penalty


def test_contact_penalty_is_full_strength_from_the_first_step():
    """The curriculum used to scale the contact penalty from 0, so early in training
    hitting the basket cost nothing."""
    rc = RewardCalculator()
    rc.set_curriculum(0.0)
    assert rc._collision_penalty(0.10, True) == pytest.approx(rc.cfg.collision_contact_penalty)
    assert rc._collision_penalty(0.0, False) == pytest.approx(0.0)      # only the soft term is eased in
    rc.set_curriculum(1.0)
    assert rc._collision_penalty(0.0, True) == pytest.approx(
        rc.cfg.collision_contact_penalty - rc.cfg.collision_exp_lambda)


@pytest.mark.parametrize("seed", range(6))
def test_scripted_policy_never_touches_the_basket_and_really_succeeds(seed):
    env = _env()
    env.reset(seed=seed)
    touched, info = False, {}
    for _ in range(be.MAX_EPISODE_STEPS):
        _, _, term, trunc, info = env.step(scripted_action(env))
        touched |= bool(info["arm_basket_contact"])
        if term or trunc:
            break
    assert info["is_success"], f"ended with failure={info['failure']} in phase {info['phase']}"
    assert not touched
    d = env.data
    # and the claim is true in the simulator: ball inside the walls, below the rim, basket upright
    local = d.xmat[env._box_body_id].reshape(3, 3).T @ (d.xpos[env._ball_body_id] - d.xpos[env._box_body_id])
    assert abs(local[0]) < env._sc.box_half and abs(local[1]) < env._sc.box_half
    assert 0 < local[2] < env._sc.box_depth
    assert env._box_tilt_deg() < 5


# ---------------------------------------------------------------------------
# 3. Ball and basket are as far apart as the active arm can reach
# ---------------------------------------------------------------------------
def test_ball_and_basket_are_placed_at_the_arms_maximum_distance():
    env = _env()
    res = env._placements["res"]
    seps, ratios = [], []
    for seed in range(60):
        env.reset(seed=seed)
        cells = env._cells[env._active_arm]
        farthest = np.linalg.norm(cells["ball_xy"][:, None] - cells["box_xy"][None], axis=2).max()
        sep = np.linalg.norm(env._ball_pos[:2] - env._box_pos[:2])
        seps.append(sep)
        ratios.append(sep / farthest)
        # within the top band of what this arm can reach (cell jitter allowed for)
        assert sep >= be.SEPARATION_TOP_FRACTION * farthest - res * np.sqrt(2)
    assert np.mean(ratios) > 0.9
    assert min(seps) > 0.30                       # the original layout topped out at 0.22 m


def test_both_objects_are_reachable_at_that_distance():
    """Maximum distance must not mean unreachable: the reference policy completes the task."""
    env = _env()
    wins = 0
    for seed in range(40, 52):
        env.reset(seed=seed)
        for _ in range(be.MAX_EPISODE_STEPS):
            _, _, term, trunc, info = env.step(scripted_action(env))
            if term or trunc:
                break
        wins += bool(info["is_success"])
    assert wins >= 11, f"only {wins}/12 at maximum separation"
