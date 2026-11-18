"""The scripted teacher (base knowledge): camera + gripper sensors only, and it works."""
import numpy as np
import pytest

from conftest import BASIC, FULL
from dual_arm_rl.env.pick_place_env import PickPlaceEnv
from dual_arm_rl.rl.evaluate import evaluate_teacher


def test_teacher_labels_come_with_every_step(cfg):
    env = PickPlaceEnv(cfg, stage=BASIC, seed=0)
    _, info = env.reset(seed=0, options={"start_skill": "grasp", "hover_start": True})
    for _ in range(5):
        a = info["teacher_action"]
        assert a.shape == (4,) and np.all(np.isfinite(a)) and np.all(np.abs(a) <= 1.0)
        _, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    env.close()


def test_teacher_never_reads_the_true_ball_position(cfg):
    """Moving the real ball without a new camera frame does not change its action."""
    import mujoco
    env = PickPlaceEnv(cfg, stage=BASIC, seed=1)
    env.reset(seed=1, options={"start_skill": "grasp", "hover_start": True})
    a1 = env.teacher.act(env)
    d = env.data
    d.qpos[env.scene.ball_qpos:env.scene.ball_qpos + 3] += [0.04, -0.03, 0.0]
    mujoco.mj_forward(env.model, d)
    # (a leak would shift the action by ~1; 1e-2 allows MuJoCo's site-position refresh)
    np.testing.assert_allclose(env.teacher.act(env), a1, atol=1e-2)
    env.close()


@pytest.mark.slow
def test_teacher_succeeds_at_the_easiest_stage(cfg):
    eps = evaluate_teacher(cfg, 10, BASIC, seed=300)
    assert sum(e["success"] for e in eps) >= 8, [e["end"] for e in eps]
    assert {e["side"] for e in eps} == {"left", "right"}


@pytest.mark.slow
def test_teacher_copes_with_full_difficulty(cfg):
    eps = evaluate_teacher(cfg, 10, FULL, seed=400)
    assert sum(e["success"] for e in eps) >= 5, [e["end"] for e in eps]
