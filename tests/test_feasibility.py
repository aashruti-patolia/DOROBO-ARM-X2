"""The task is solvable: a privileged oracle (test only) completes it through the
real pipeline - scripted transit/transport, safety limits, friction grasp."""
import pytest

from conftest import BASIC, FULL
from dual_arm_rl.env.pick_place_env import PickPlaceEnv
from oracle import run_oracle


@pytest.mark.slow
def test_the_full_pipeline_succeeds_at_the_easiest_stage(cfg):
    env = PickPlaceEnv(cfg, stage=BASIC, seed=0)
    eps = run_oracle(env, 10, seed=100, hover_start=False)       # including the scripted transit
    env.close()
    wins = sum(e["success"] for e in eps)
    assert wins >= 8, [e["end"] for e in eps]


@pytest.mark.slow
def test_the_task_stays_solvable_at_full_difficulty(cfg):
    """Physics/visual randomisation, rolling ball, pushes, distractors, latency."""
    env = PickPlaceEnv(cfg, stage=FULL, seed=0)
    eps = run_oracle(env, 12, seed=200)
    env.close()
    wins = sum(e["success"] for e in eps)
    assert wins >= 6, [e["end"] for e in eps]          # the naive oracle struggles with latency
