"""The Gymnasium environment: interface, camera-only observations, arm choice, skills."""
from collections import deque

import mujoco
import numpy as np
import pytest

from conftest import BASIC, FULL
from dual_arm_rl.env.controller import MIRROR_JOINT_SIGNS
from dual_arm_rl.env.pick_place_env import OBS_DIM, OBS_SLICES, PickPlaceEnv
from dual_arm_rl.env.reward import GRASP, PLACE


@pytest.fixture(scope="module")
def env(cfg):
    e = PickPlaceEnv(cfg, stage=BASIC, seed=0)
    yield e
    e.close()


def test_spaces_and_observation(env):
    obs, info = env.reset(seed=1)
    assert env.action_space.shape == (4,) and env.observation_space.shape == (OBS_DIM,)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32 and np.isfinite(obs).all()
    assert info["skill"] in ("grasp", "place") and "scenario" in info


def test_the_camera_picks_the_arm_on_the_balls_side(env):
    for seed in range(8):
        env.reset(seed=seed)
        true_y = env._world["ball_pos"][1]
        assert env._side == ("left" if true_y > 0 else "right")


def test_the_policy_never_sees_the_true_ball_position(env):
    """Ball features come from the camera belief only."""
    env.reset(seed=3, options={"start_skill": "grasp", "hover_start": True})
    belief = env.perception.state
    belief.ball_pos = np.array([0.31, 0.12 if env._side == "left" else -0.12, 0.29])
    pinch = env.scene.pinch_pos(env._side)
    mir = np.array([1, -1, 1]) if env._side == "right" else np.ones(3)
    obs1 = env._obs()
    np.testing.assert_allclose(obs1[OBS_SLICES["ball_rel"]], mir * (belief.ball_pos - pinch), atol=1e-5)
    d = env.data                                            # move the real ball; the camera has not looked
    d.qpos[env.scene.ball_qpos:env.scene.ball_qpos + 3] += [0.05, 0.05, 0.0]
    mujoco.mj_forward(env.model, d)
    obs2 = env._obs()
    # (1e-4: MuJoCo reports site positions from the start of the last substep)
    np.testing.assert_allclose(obs2[OBS_SLICES["ball_rel"]], obs1[OBS_SLICES["ball_rel"]], atol=1e-4)


def test_mirrored_arms_produce_the_same_observation(env):
    env.reset(seed=4, options={"start_skill": "grasp", "hover_start": True})
    s, d = env.scene, env.data
    s.sc.encoder_pos_noise = s.sc.encoder_vel_noise = 0.0
    q_left = s.ik.solve("left", [0.25, 0.2, 0.35], None).q_arm
    out = {}
    for side, q in (("left", q_left), ("right", MIRROR_JOINT_SIGNS * q_left)):
        env._side = side
        d.qpos[s.qpos_adr[side]] = q
        d.qvel[:] = 0
        mujoco.mj_forward(env.model, d)
        out[side] = env._obs()
    for name in ("joint_pos", "joint_vel", "pinch_quat"):
        np.testing.assert_allclose(out["left"][OBS_SLICES[name]], out["right"][OBS_SLICES[name]],
                                   atol=1e-6, err_msg=name)


def test_grasp_start_puts_the_open_hand_above_the_ball(env):
    for seed, hover in ((5, True), (6, False)):
        env.reset(seed=seed, options={"start_skill": "grasp", "hover_start": hover})
        assert env._skill == GRASP
        pinch, ball = env.scene.pinch_pos(env._side), env.scene.ball_pos()
        assert np.linalg.norm(pinch[:2] - ball[:2]) < 0.025
        assert 0.03 < pinch[2] - ball[2] < 0.09
        assert env.scene.hand_tilt_deg(env._side) < 10
        assert env.scene.finger_angle(env._side) > 0.5                 # open


def test_place_start_holds_the_ball_above_the_basket(env):
    env.reset(seed=7, options={"start_skill": "place"})
    s = env.scene
    assert env._skill == PLACE and env._held_sensor()
    assert np.linalg.norm(s.ball_pos() - s.pinch_pos(env._side)) < 0.01
    assert np.linalg.norm(s.ball_pos()[:2] - s.box_pos()[:2]) < 0.03
    assert s.ball_pos()[2] > s.sc.box_rim_z


def test_command_latency_delays_the_executed_action(env):
    env.reset(seed=8, options={"start_skill": "grasp", "hover_start": True})
    env._queue = deque([np.zeros(4, np.float32), np.zeros(4, np.float32)])
    target0 = env.controller.target.copy()
    env.step(np.array([1.0, 0, 0, 1.0], np.float32))
    env.step(np.array([1.0, 0, 0, 1.0], np.float32))
    assert np.linalg.norm(env.controller.target - target0) < 1e-6        # still the delayed zeros
    env.step(np.array([1.0, 0, 0, 1.0], np.float32))
    assert np.linalg.norm(env.controller.target - target0) > 0.005


def test_the_policy_cannot_leave_its_safety_box(env):
    env.reset(seed=9, options={"start_skill": "grasp", "hover_start": True})
    for _ in range(25):
        _, r, term, trunc, info = env.step(np.array([0, 0, -1.0, 1.0], np.float32))
        if term or trunc:
            break
    assert env.controller.target[2] >= env._box.lo[2] - 1e-9
    fingertip = env.scene.pinch_pos(env._side)[2] - env.cfg.control.fingertip_below_pinch
    assert fingertip > env.scene.sc.table_z - 0.004                    # never jams into the table


def test_episodes_end_with_a_summary(env):
    env.reset(seed=10)
    rng = np.random.default_rng(0)
    for _ in range(300):
        _, r, term, trunc, info = env.step(rng.uniform(-1, 1, 4).astype(np.float32))
        assert np.isfinite(r)
        if term or trunc:
            break
    assert term or trunc
    e = info["episode"]
    for k in ("success", "return", "length", "end", "max_stage", "table_z", "ball_radius", "box_half",
              "distance", "rolling", "side", "start_skill", "ball_err"):
        assert k in e


def test_curriculum_stage_can_be_switched(cfg):
    e = PickPlaceEnv(cfg, stage=BASIC, seed=0)
    e.set_curriculum_stage(99)
    assert e.get_curriculum_stage() == FULL
    e.reset(seed=0)
    assert e.scene.sc.physics_level == cfg.curriculum.stages[FULL].physics_level
    e.close()
