"""Tests for per-episode scenarios: table height, object sizes, friction, sensing,
distractors and disturbances.

Run from the openarm_v20_mujoco directory:  python -m pytest tests/test_scenarios.py -q
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
from scripted_policy import run_episode  # noqa: E402

HEIGHTS = [0.13, 0.17, 0.21, 0.25, 0.29, 0.33, 0.37]


def _env(level="off", **kw):
    return be.BimanualPickPlaceEnv(randomization_level=level, obs_noise_scale=0.0, **kw)


def _settle(env, steps=500):
    for _ in range(steps):
        mujoco.mj_step(env.model, env.data)


# ---------------------------------------------------------------------------
# The scenario is really applied to the model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table_z", HEIGHTS)
def test_table_top_is_at_the_scenario_height(table_z):
    env = _env()
    env.reset(seed=0, options={"scenario": {"table_z": table_z}})
    m, tab = env.model, env._table_geom_id
    assert m.geom_pos[tab][2] + m.geom_size[tab][2] == pytest.approx(table_z)   # top surface
    assert m.geom_pos[tab][2] - m.geom_size[tab][2] == pytest.approx(0.0)       # stands on the floor
    assert env._sc.table_z == table_z
    assert env._ball_pos[2] == pytest.approx(table_z + env._sc.ball_radius)
    assert env._box_pos[2] == pytest.approx(table_z)


def test_ball_and_basket_dimensions_follow_the_scenario():
    env = _env()
    env.reset(seed=0, options={"scenario": {"ball_radius": 0.03, "box_half": 0.055, "box_depth": 0.08}})
    m = env.model
    assert m.geom_size[env._ball_geom_id][0] == pytest.approx(0.03)
    assert m.geom_rbound[env._ball_geom_id] == pytest.approx(0.03)          # bounds refreshed too
    ratio = 0.03 / be.BALL_RADIUS
    assert m.body_mass[env._ball_body_id] == pytest.approx(
        env._nominal["body_mass"][env._ball_body_id] * ratio ** 3)
    wall = m.geom_size[env._box_part_ids["box_wall_f"]]
    assert wall[2] * 2 == pytest.approx(0.08)                                # wall height
    assert wall[1] == pytest.approx(0.055)                                   # half-width
    assert env._box_half_sizes.max() >= 0.05                                 # clearance cache refreshed


def test_unknown_scenario_field_is_rejected():
    env = _env()
    with pytest.raises(ValueError, match="Unknown scenario"):
        env.reset(seed=0, options={"scenario": {"tabel_z": 0.3}})


# ---------------------------------------------------------------------------
# Collisions stay correct however the world is resized
# (the bug that made this necessary: a raised table silently stopped colliding)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table_z", HEIGHTS)
def test_ball_rests_on_the_resized_table(table_z):
    env = _env()
    env.reset(seed=0, options={"scenario": {"table_z": table_z}})
    d, sc = env.data, env._sc
    d.qpos[env._ball_qpos_adr:env._ball_qpos_adr + 3] = [0.45, 0.3, table_z + 0.1]
    d.qvel[:] = 0
    mujoco.mj_forward(env.model, d)
    _settle(env)
    assert d.xpos[env._ball_body_id][2] - table_z == pytest.approx(sc.ball_radius, abs=0.004)


@pytest.mark.parametrize("table_z", [0.17, 0.29, 0.33, 0.37])
@pytest.mark.parametrize("half,depth", [(0.038, 0.045), (0.052, 0.075)])
def test_ball_lands_on_the_floor_of_the_resized_basket(table_z, half, depth):
    env = _env()
    env.reset(seed=1, options={"scenario": {"table_z": table_z, "box_half": half, "box_depth": depth}})
    d, sc = env.data, env._sc
    box = d.xpos[env._box_body_id].copy()
    d.qpos[env._ball_qpos_adr:env._ball_qpos_adr + 3] = [box[0], box[1], box[2] + depth + 0.1]
    d.qvel[:] = 0
    mujoco.mj_forward(env.model, d)
    _settle(env)
    assert d.xpos[env._box_body_id][2] - table_z == pytest.approx(0.0, abs=0.004)   # basket sits on the table
    rest = d.xpos[env._ball_body_id][2] - table_z
    assert rest == pytest.approx(2 * be.BOX_BOTTOM_T + sc.ball_radius, abs=0.005)


@pytest.mark.parametrize("dx,dy", [(-0.04, 0.0), (0.04, 0.04), (0.0, -0.04)])
@pytest.mark.parametrize("table_z", [0.17, 0.25, 0.33])
def test_arms_do_not_touch_the_table_at_the_home_pose(table_z, dx, dy):
    env = _env()
    env.reset(seed=0, options={"scenario": {"table_z": table_z, "table_dx": dx, "table_dy": dy}})
    d = env.data
    arm = env._arm_geom_mask["left"] | env._arm_geom_mask["right"]
    tab = env._table_geom_id
    touching = [i for i in range(d.ncon)
                if tab in (d.contact.geom1[i], d.contact.geom2[i])
                and (arm[d.contact.geom1[i]] or arm[d.contact.geom2[i]])]
    assert not touching


def test_a_table_raised_into_the_hands_is_detected():
    env = _env()
    env.reset(seed=0)                                    # arms at home, table at nominal height
    tab, m, d = env._table_geom_id, env.model, env.data
    assert not any(tab in (d.contact.geom1[i], d.contact.geom2[i]) for i in range(d.ncon)
                   if (env._arm_geom_mask["left"] | env._arm_geom_mask["right"])[
                       d.contact.geom1[i] if d.contact.geom2[i] == tab else d.contact.geom2[i]])
    env._set_geom_box(tab, pos=[*m.geom_pos[tab][:2], 0.225], half=[*m.geom_size[tab][:2], 0.225])   # top at 0.45
    mujoco.mj_forward(m, d)
    arm = env._arm_geom_mask["left"] | env._arm_geom_mask["right"]
    assert any(tab in (d.contact.geom1[i], d.contact.geom2[i])
               and (arm[d.contact.geom1[i]] or arm[d.contact.geom2[i]]) for i in range(d.ncon))


@pytest.mark.parametrize("table_z", [0.17, 0.29, 0.33, 0.37])
def test_hands_start_clear_of_the_table_at_every_height(table_z):
    env = _env()
    env.reset(seed=0, options={"scenario": {"table_z": table_z}})
    lowest = env._home_cloud[:, 2].min()
    assert lowest - table_z >= be.START_CLEARANCE - 0.003


# ---------------------------------------------------------------------------
# Randomisation: deterministic, bounded, never compounding, switchable
# ---------------------------------------------------------------------------
def test_same_seed_gives_the_same_scenario_and_first_observation():
    runs = []
    for _ in range(2):
        env = be.BimanualPickPlaceEnv(randomization_level="full")
        obs, info = env.reset(seed=7)
        runs.append((info["scenario"], obs))
    assert runs[0][0] == runs[1][0]
    np.testing.assert_array_equal(runs[0][1], runs[1][1])


@pytest.mark.parametrize("level", ["mild", "full"])
def test_sampled_scenarios_stay_inside_their_ranges(level):
    env = _env(level)
    k, R = be.LEVEL_SCALE[level], be.SCENARIO_RANGES
    z_lo, z_hi = env._table_z_range
    seen = []
    for ep in range(60):
        _, info = env.reset(seed=ep)
        sc = info["scenario"]
        seen.append(sc)
        assert z_lo - 1e-9 <= sc["table_z"] <= z_hi + 1e-9
        assert abs(sc["table_z"] - be.TABLE_Z) <= k * R["table_z_halfrange"] + 1e-9
        assert abs(sc["table_dx"]) <= k * R["table_shift"] + 1e-9
        r = sc["ball_radius"] / be.BALL_RADIUS
        assert 1 + k * (R["ball_radius_scale"][0] - 1) - 1e-9 <= r <= 1 + k * (R["ball_radius_scale"][1] - 1) + 1e-9
        assert sc["encoder_pos_noise"] == pytest.approx(k * R["encoder_pos_noise"])
        assert 0 <= sc["n_distractors"] <= round(k * R["max_distractors"])
    assert len({s["ball_radius"] for s in seen}) > 30       # genuinely varies


def test_off_means_exactly_nominal_even_after_randomised_episodes():
    env = be.BimanualPickPlaceEnv(randomization_level="full", obs_noise_scale=0.0)
    for ep in range(20):
        env.reset(seed=ep)                                   # 20 randomised worlds in a row
    env.scenario_level = "off"
    env.domain_randomization = False
    env.reset(seed=99)

    fresh = _env()
    fresh.reset(seed=99)
    for attr in ("geom_pos", "geom_size", "geom_rbound", "geom_aabb", "geom_solref", "geom_friction",
                 "body_pos", "body_mass", "body_inertia", "actuator_gainprm", "dof_damping"):
        np.testing.assert_allclose(getattr(env.model, attr), getattr(fresh.model, attr),
                                   err_msg=f"{attr} drifted from nominal")
    assert env._sc == be.Scenario()


def test_levels_are_ordered_by_strength():
    spread = {}
    for level in ("mild", "full"):
        env = _env(level)
        radii = [env.reset(seed=s)[1]["scenario"]["ball_radius"] for s in range(40)]
        spread[level] = np.ptp(radii)
    assert 0 < spread["mild"] < spread["full"]


# ---------------------------------------------------------------------------
# Placement follows the scenario
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(10))
def test_objects_sit_on_this_episodes_table_and_distractors_stay_clear(seed):
    env = _env("full")
    env.reset(seed=seed)
    sc, m, d = env._sc, env.model, env.data
    tab = env._table_geom_id
    cx, cy = m.geom_pos[tab][:2]
    hx, hy = m.geom_size[tab][:2]
    ball, box = d.xpos[env._ball_body_id], d.xpos[env._box_body_id]
    assert abs(ball[0] - cx) <= hx - sc.ball_radius and abs(ball[1] - cy) <= hy - sc.ball_radius
    assert abs(box[0] - cx) <= hx - sc.box_half and abs(box[1] - cy) <= hy - sc.box_half
    assert ball[2] == pytest.approx(sc.table_z + sc.ball_radius, abs=0.005)
    assert np.linalg.norm(ball[:2] - box[:2]) >= be.MIN_SEPARATION - 0.03

    placed = [b for b in env._distractor_bodies if d.xpos[b][2] > 0]
    assert len(placed) == sc.n_distractors
    seg = box[:2] - ball[:2]
    for b in placed:
        p = d.xpos[b]
        assert p[2] > sc.table_z                                           # on top of the table
        assert np.linalg.norm(p[:2] - ball[:2]) > 0.10
        assert np.linalg.norm(p[:2] - box[:2]) > 0.12
        t = np.clip((p[:2] - ball[:2]) @ seg / (seg @ seg), 0, 1)
        assert np.linalg.norm(p[:2] - (ball[:2] + t * seg)) > 0.09          # off the carry path


@pytest.mark.parametrize("table_z", [0.25, 0.29, 0.33, 0.37])
def test_no_episode_starts_with_the_hand_on_an_object(table_z):
    """With a tall table the home-pose hands hang just above it: objects must not be
    placed underneath them (9 of 40 episodes used to start interpenetrating)."""
    env = _env()
    for seed in range(25):
        env.reset(seed=seed, options={"scenario": {"table_z": table_z, "n_distractors": 3}})
        d = env.data
        arm = env._arm_geom_mask["left"] | env._arm_geom_mask["right"]
        for i in range(d.ncon):
            g1, g2 = d.contact.geom1[i], d.contact.geom2[i]
            if arm[g1] or arm[g2]:
                other = g2 if arm[g1] else g1
                assert other not in (env._ball_geom_id, *env._distractor_geoms) and not env._box_geom_mask[other],                     f"seed {seed}: the arm starts in contact with {env.model.geom(other).name}"
        # nothing pushed the arm while it settled: it is still in its start pose
        q_left, q_right, _ = env._start_pose(table_z)
        np.testing.assert_allclose(d.qpos[env._left_qpos_adr], q_left, atol=2e-3)
        np.testing.assert_allclose(d.qpos[env._right_qpos_adr], q_right, atol=2e-3)


def test_touching_a_distractor_counts_as_a_collision():
    env = _env()
    env.reset(seed=0, options={"scenario": {"n_distractors": 1}})
    assert not env._check_arm_distractor_contact()
    d = env.data
    ee = env._get_ee_state()[0]
    body = env._distractor_bodies[0]
    env.model.body_pos[body] = [ee[0], ee[1], ee[2]]                        # on top of the fingers
    mujoco.mj_forward(env.model, d)
    assert env._check_arm_distractor_contact()


# ---------------------------------------------------------------------------
# Sensing and disturbances
# ---------------------------------------------------------------------------
def test_observation_height_is_relative_to_the_table():
    obs_by_z = {}
    for z in (0.17, 0.25):                                  # both start from the same home pose
        env = _env()
        env.reset(seed=0, options={"scenario": {"table_z": z}})
        ee_abs = env._get_ee_state()[0][2]
        obs_by_z[z] = env._get_obs()[21]                    # ee_pos z in the observation
        assert obs_by_z[z] == pytest.approx(ee_abs - z, abs=1e-5)
    assert obs_by_z[0.17] - obs_by_z[0.25] == pytest.approx(0.08, abs=1e-3)   # same pose, table 8 cm lower


def test_encoder_noise_matches_its_configured_std():
    env = _env()
    env.reset(seed=0, options={"scenario": {"encoder_pos_noise": 0.01, "encoder_vel_noise": 0.1}})
    obs = np.array([env._get_obs() for _ in range(600)])
    assert obs[:, 5:12].std(axis=0).mean() == pytest.approx(0.01, rel=0.2)    # joint positions
    assert obs[:, 12:19].std(axis=0).mean() == pytest.approx(0.1, rel=0.2)    # joint velocities
    clean = _env()
    clean.reset(seed=0)
    assert np.array([clean._get_obs() for _ in range(50)])[:, 5:12].std(axis=0).max() < 1e-7


def test_observation_dropout_repeats_the_last_frame():
    env = _env()
    env.reset(seed=0, options={"scenario": {"dropout_prob": 0.3}})
    frozen = total = 0
    prev = None
    for _ in range(300):
        obs, *_ = env.step(np.array([0, 0, 0, 1.0], dtype=np.float32))
        frozen += prev is not None and np.array_equal(obs, prev)
        total += 1
        prev = obs
    assert 0.2 < frozen / total < 0.8
    clean = _env()
    clean.reset(seed=0)
    prev, repeats = None, 0
    for _ in range(100):
        obs, *_ = clean.step(np.array([0, 0, 0, 1.0], dtype=np.float32))
        repeats += prev is not None and np.array_equal(obs, prev)
        prev = obs
    assert repeats == 0


def test_pushes_only_move_a_resting_ball_before_it_is_grasped():
    env = _env()
    env.reset(seed=0, options={"scenario": {"push_prob": 1.0}})
    d, adr = env.data, env._ball_dof_adr
    d.qvel[adr:adr + 3] = 0
    env._maybe_push_ball()
    assert np.linalg.norm(d.qvel[adr:adr + 2]) >= env._sc.push_speed_min - 1e-9    # nudged

    d.qvel[adr:adr + 3] = 0
    env._phase = be.PHASE_GRASP
    env._maybe_push_ball()
    assert not d.qvel[adr:adr + 2].any()                                           # never during the grasp

    env._phase = be.PHASE_APPROACH
    env._ball_grasped = True
    env._maybe_push_ball()
    assert not d.qvel[adr:adr + 2].any()                                           # never once held

    env._ball_grasped = False
    d.qvel[adr:adr + 3] = [0.5, 0, 0]                                               # already rolling
    env._maybe_push_ball()
    assert d.qvel[adr] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The task still works across the scenario ranges (scripted reference policy)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("param,value", [
    ("table_z", 0.17), ("table_z", 0.21), ("table_z", 0.25), ("table_z", 0.29), ("table_z", 0.33),
    ("table_dx", 0.02), ("table_dy", -0.02),
    ("ball_radius", 0.019), ("ball_radius", 0.0253),
    ("box_half", 0.038), ("box_half", 0.0518),
    ("box_depth", 0.045), ("box_depth", 0.075),
    ("table_friction_scale", 0.7), ("box_friction_scale", 1.3),
    ("ball_solref_scale", 0.6), ("ball_solref_scale", 1.6),
    ("encoder_pos_noise", 0.002),
])
def test_scripted_policy_succeeds_across_the_range(param, value):
    env = _env()
    wins = sum(run_episode(env, seed=100 + i, scenario={param: value})["success"] for i in range(4))
    assert wins >= 3, f"scripted policy succeeded in only {wins}/4 episodes with {param}={value}"


def test_full_randomisation_is_survivable():
    env = be.BimanualPickPlaceEnv(randomization_level="full")
    wins = sum(run_episode(env, seed=300 + i)["success"] for i in range(12))
    assert wins >= 6, f"{wins}/12 under full randomisation"
