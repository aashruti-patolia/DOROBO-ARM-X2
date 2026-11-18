"""What counts as success and failure.

Success is only a released ball resting inside the upright basket. A ball beside
it, balanced on the rim, still held, or in a tipped basket is never a success.
"""
import mujoco
import numpy as np
import pytest

from conftest import BASIC, stage
from dual_arm_rl.env.pick_place_env import PickPlaceEnv

OPEN = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
HOLD = np.array([0.0, 0.0, 0.0, -1.0], np.float32)


def _ball_at(scene, offset, box_quat=(1, 0, 0, 0), box_lift=0.0):
    d = scene.data
    box = d.xpos[scene.box_body].copy()
    d.qpos[scene.box_qpos + 2] += box_lift
    d.qpos[scene.box_qpos + 3:scene.box_qpos + 7] = box_quat
    d.qpos[scene.ball_qpos:scene.ball_qpos + 3] = box + np.asarray(offset, float)
    d.qvel[:] = 0
    mujoco.mj_forward(scene.model, d)


def test_ball_inside_the_upright_basket_counts(scene, cfg, rng):
    scene.reset(rng, stage(cfg, BASIC))
    _ball_at(scene, [0, 0, 0.03])
    assert scene.ball_in_box()


def test_ball_beside_the_basket_does_not_count(scene, cfg, rng):
    scene.reset(rng, stage(cfg, BASIC))
    _ball_at(scene, [scene.sc.box_half + scene.sc.ball_radius + 0.01, 0, scene.sc.ball_radius])
    assert not scene.ball_in_box()


def test_ball_above_the_rim_does_not_count(scene, cfg, rng):
    scene.reset(rng, stage(cfg, BASIC))
    _ball_at(scene, [0, 0, scene.sc.box_depth + 0.03])
    assert not scene.ball_in_box()


@pytest.mark.parametrize("tilt_deg", [35, 60, 120, 180])
def test_a_tipped_basket_never_counts(cfg, tilt_deg):
    env = PickPlaceEnv(cfg, stage=BASIC, seed=0)
    env.reset(seed=0, options={"start_skill": "place"})
    half = np.radians(tilt_deg) / 2
    _ball_at(env.scene, [0, 0, 0.03], box_quat=(np.cos(half), np.sin(half), 0, 0))
    assert not env.scene.ball_in_box()
    assert env._check_failure() == "failure" and env._episode["end_reason"] == "box_tipped"
    env.close()


@pytest.mark.parametrize("tilt_deg", [90, 180])
def test_a_knocked_over_basket_ends_the_episode_as_a_failure(cfg, tilt_deg):
    env = PickPlaceEnv(cfg, stage=BASIC, seed=0)
    env.reset(seed=0, options={"start_skill": "place"})
    half = np.radians(tilt_deg) / 2
    _ball_at(env.scene, [0, 0, 0.03], box_quat=(np.cos(half), np.sin(half), 0, 0),
             box_lift=env.scene.sc.box_half + 0.01)
    _, r, term, _, info = env.step(HOLD)
    assert term and not info["is_success"] and info["episode"]["end"] == "box_tipped"
    assert info["r_event"] == -cfg.reward.failure_penalty
    env.close()


def test_releasing_over_the_basket_succeeds(cfg):
    env = PickPlaceEnv(cfg, stage=BASIC, seed=1)
    wins = 0
    for seed in range(5):
        env.reset(seed=seed, options={"start_skill": "place"})
        _, r, term, trunc, info = env.step(OPEN)        # the hold-over-the-basket pose is right above it
        assert term
        wins += info["is_success"]
        if info["is_success"]:
            assert r > 0 and env.scene.ball_in_box()
    assert wins >= 4
    env.close()


def test_a_ball_still_in_the_hand_is_not_a_success(cfg):
    env = PickPlaceEnv(cfg, stage=BASIC, seed=2)
    env.reset(seed=2, options={"start_skill": "place"})
    for _ in range(10):
        _, _, term, trunc, info = env.step(HOLD)
        assert not info["is_success"] and not term
    env.close()


def test_releasing_away_from_the_basket_is_a_miss(cfg):
    env = PickPlaceEnv(cfg, stage=BASIC, seed=3)
    env.reset(seed=3, options={"start_skill": "place"})
    d, s = env.data, env.scene
    d.qpos[s.box_qpos] += 0.15                         # the basket is not under the hand any more
    mujoco.mj_forward(env.model, d)
    _, r, term, _, info = env.step(OPEN)
    assert term and not info["is_success"] and info["r_event"] == -cfg.reward.miss_penalty
    env.close()


def test_touching_the_basket_is_penalised(cfg):
    env = PickPlaceEnv(cfg, stage=BASIC, seed=4)
    env.reset(seed=4, options={"start_skill": "place"})
    d, s = env.data, env.scene
    pinch = s.pinch_pos(env._side)
    d.qpos[s.box_qpos:s.box_qpos + 3] = [pinch[0], pinch[1], pinch[2] - 0.005]   # floor into the fingertips
    d.qvel[:] = 0
    mujoco.mj_forward(env.model, d)
    assert s.arm_box_contact(env._side)
    _, r, _, _, info = env.step(HOLD)
    assert info["r_penalty"] <= -cfg.reward.box_contact_penalty
    env.close()
