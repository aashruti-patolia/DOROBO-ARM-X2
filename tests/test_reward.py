"""Potential-based reward: progress pays, lingering never does, finishing is what is optimal."""
import numpy as np
import pytest

from dual_arm_rl.config import ControlConfig, RewardConfig
from dual_arm_rl.env.reward import GRASP, PLACE, RewardFunction, TaskSnapshot

GAMMA = 0.99


@pytest.fixture
def rf():
    return RewardFunction(RewardConfig(), ControlConfig(), GAMMA)


def snap(skill=GRASP, pinch=(0.3, 0.2, 0.35), ball=(0.3, 0.2, 0.27), box=(0.3, -0.1, 0.25), held=False):
    return TaskSnapshot(skill=skill, pinch=np.array(pinch), ball=np.array(ball), box=np.array(box),
                        table_z=0.25, ball_radius=0.02, held=held)


def test_stages_are_strictly_ordered(rf):
    far = rf.potential(snap(pinch=(0.0, 0.0, 0.6)))
    touching = rf.potential(snap(pinch=(0.3, 0.2, 0.27)))
    held = rf.potential(snap(held=True))
    lifted = rf.potential(snap(held=True, ball=(0.3, 0.2, 0.30)))
    placing_off = rf.potential(snap(skill=PLACE, ball=(0.4, 0.0, 0.4)))
    placing_on = rf.potential(snap(skill=PLACE, ball=(0.3, -0.1, 0.4)))
    assert 0 <= far < touching <= 1.0 < 2.0 <= held < lifted <= 3.0 <= placing_off < placing_on <= 4.0


def test_staying_put_never_pays(rf):
    """The old reward paid +5 per step for touching the ball; the policy learned to hover there."""
    for s in (snap(pinch=(0.3, 0.2, 0.28)), snap(held=True), snap(skill=PLACE, ball=(0.3, -0.1, 0.4))):
        phi = rf.potential(s)
        r, _ = rf(phi, phi, terminal=False, outcome=None, penalties={})
        assert r <= 0


def test_the_shaped_return_adds_up_to_the_outcome(rf):
    """Sum of gamma^t r_t over any trajectory = gamma^(T-1) * outcome - Phi(s_0)."""
    rng = np.random.default_rng(0)
    phis = rng.uniform(0, 4, size=30)
    total = 0.0
    for t in range(len(phis) - 1):
        terminal = t == len(phis) - 2
        r, _ = rf(phis[t], phis[t + 1], terminal, "success" if terminal else None, {})
        total += GAMMA ** t * r
    expected = GAMMA ** (len(phis) - 2) * rf.cfg.success_bonus - phis[0]
    assert total == pytest.approx(expected, rel=1e-9)


def test_outcomes_and_penalties(rf):
    r_ok, info = rf(3.5, 0.0, True, "success", {})
    r_miss, _ = rf(3.5, 0.0, True, "miss", {})
    r_fail, _ = rf(3.5, 0.0, True, "failure", {})
    r_timeout, _ = rf(3.5, 0.0, True, "timeout", {})
    assert r_ok > 0 > r_miss > r_fail
    assert r_timeout == pytest.approx(r_miss)              # running out the clock is no safer than missing
    assert info["r_event"] == rf.cfg.success_bonus
    pen = rf.penalties(box_contact=True, distractor_contact=False, safety_clip=0.01,
                       action=np.ones(4), prev_action=-np.ones(4))
    assert pen["box_contact"] == rf.cfg.box_contact_penalty and pen["safety"] > 0
    assert pen["action_rate"] == pytest.approx(rf.cfg.action_rate_penalty * 16)
    r_pen, _ = rf(1.0, 1.0, False, None, pen)
    assert r_pen < -rf.cfg.box_contact_penalty


def test_releasing_over_the_basket_is_not_punished_before_it_lands(rf):
    """PLACE progress is defined by where the ball is, held or falling."""
    held = rf.potential(snap(skill=PLACE, ball=(0.3, -0.1, 0.40), held=True))
    falling = rf.potential(snap(skill=PLACE, ball=(0.3, -0.1, 0.36), held=False))
    assert falling == pytest.approx(held)
