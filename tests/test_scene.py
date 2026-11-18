"""World generation: every episode's table, ball, basket and distance really change,
collisions stay correct however things are resized, and randomisation never compounds."""
import mujoco
import numpy as np
import pytest

from conftest import BASIC, FULL, stage
from dual_arm_rl.config import Config
from dual_arm_rl.env.scene import BALL_RADIUS, BOX_BOTTOM_T, START_CLEARANCE, Scenario, Scene

HEIGHTS = [0.17, 0.21, 0.25, 0.29, 0.33]


def _settle(scene, steps=500):
    for _ in range(steps):
        mujoco.mj_step(scene.model, scene.data)


@pytest.mark.parametrize("table_z", HEIGHTS)
def test_table_top_is_at_the_scenario_height(scene, cfg, rng, table_z):
    world = scene.reset(rng, stage(cfg, BASIC), {"table_z": table_z})
    m, tab = scene.model, scene.table_geom
    assert m.geom_pos[tab][2] + m.geom_size[tab][2] == pytest.approx(table_z)
    assert m.geom_pos[tab][2] - m.geom_size[tab][2] == pytest.approx(0.0)
    assert world["ball_pos"][2] == pytest.approx(table_z + scene.sc.ball_radius)
    assert world["box_pos"][2] == pytest.approx(table_z)


def test_ball_and_basket_dimensions_follow_the_scenario(scene, cfg, rng):
    scene.reset(rng, stage(cfg, BASIC), {"ball_radius": 0.025, "box_half": 0.05, "box_depth": 0.07})
    m = scene.model
    assert m.geom_size[scene.ball_geom][0] == pytest.approx(0.025)
    assert m.geom_rbound[scene.ball_geom] == pytest.approx(0.025)          # bounds refreshed too
    assert m.body_mass[scene.ball_body] == pytest.approx(
        scene.nominal["body_mass"][scene.ball_body] * (0.025 / BALL_RADIUS) ** 3)
    wall = m.geom_size[scene.box_parts["box_wall_f"]]
    assert wall[2] * 2 == pytest.approx(0.07) and wall[1] == pytest.approx(0.05)


def test_unknown_scenario_field_is_rejected(scene, cfg, rng):
    with pytest.raises(ValueError, match="Unknown scenario"):
        scene.reset(rng, stage(cfg, BASIC), {"tabel_z": 0.3})


@pytest.mark.parametrize("table_z", HEIGHTS)
def test_ball_rests_on_the_resized_table(scene, cfg, rng, table_z):
    scene.reset(rng, stage(cfg, BASIC), {"table_z": table_z})
    d = scene.data
    d.qpos[scene.ball_qpos:scene.ball_qpos + 3] = [0.45, 0.3, table_z + 0.1]
    d.qvel[:] = 0
    mujoco.mj_forward(scene.model, d)
    _settle(scene)
    assert d.xpos[scene.ball_body][2] - table_z == pytest.approx(scene.sc.ball_radius, abs=0.004)


@pytest.mark.parametrize("table_z,half,depth", [(0.17, 0.038, 0.045), (0.33, 0.052, 0.075)])
def test_ball_lands_on_the_floor_of_the_resized_basket(scene, cfg, rng, table_z, half, depth):
    scene.reset(rng, stage(cfg, BASIC), {"table_z": table_z, "box_half": half, "box_depth": depth})
    d = scene.data
    box = d.xpos[scene.box_body].copy()
    d.qpos[scene.ball_qpos:scene.ball_qpos + 3] = [box[0], box[1], box[2] + depth + 0.1]
    d.qvel[:] = 0
    mujoco.mj_forward(scene.model, d)
    _settle(scene)
    assert d.xpos[scene.box_body][2] - table_z == pytest.approx(0.0, abs=0.004)
    assert d.xpos[scene.ball_body][2] - table_z == pytest.approx(2 * BOX_BOTTOM_T + scene.sc.ball_radius, abs=0.005)
    assert scene.ball_in_box()


@pytest.mark.parametrize("table_z", HEIGHTS)
def test_hands_start_clear_of_the_table(scene, cfg, rng, table_z):
    scene.reset(rng, stage(cfg, BASIC), {"table_z": table_z, "n_distractors": 2})
    assert scene.home_cloud[:, 2].min() - table_z >= START_CLEARANCE
    d = scene.data
    arms = scene.arm_geom_mask["left"] | scene.arm_geom_mask["right"]
    for i in range(d.ncon):
        g1, g2 = d.contact.geom1[i], d.contact.geom2[i]
        assert not (arms[g1] ^ arms[g2]), "an arm starts touching something"


def test_geometry_and_distance_vary_every_episode_even_at_the_easiest_stage(scene, cfg, rng):
    """Ball size, basket size, table height and ball-basket distance change every episode."""
    seen = {"table_z": [], "ball_radius": [], "box_half": [], "box_depth": [], "distance": []}
    for _ in range(40):
        world = scene.reset(rng, stage(cfg, BASIC))
        for k in ("table_z", "ball_radius", "box_half", "box_depth"):
            seen[k].append(world["scenario"][k])
        seen["distance"].append(world["distance"])
    s = cfg.scene
    assert np.ptp(seen["table_z"]) > 0.08
    assert np.ptp(seen["ball_radius"]) > 0.3 * BALL_RADIUS * (s.ball_radius_scale[1] - s.ball_radius_scale[0])
    assert np.ptp(seen["box_half"]) > 0.005
    assert np.ptp(seen["distance"]) > 0.04
    assert len(set(np.round(seen["ball_radius"], 6))) == 40


def test_distance_covers_the_whole_range_at_full_difficulty(scene, cfg, rng):
    d = [scene.reset(rng, stage(cfg, FULL))["distance"] for _ in range(60)]
    assert min(d) < 0.17 and max(d) > 0.33
    assert min(d) >= cfg.scene.min_distance - 0.03


def test_distance_follows_the_curriculum_stage(scene, cfg, rng):
    d = [scene.reset(rng, stage(cfg, BASIC))["distance"] for _ in range(40)]
    assert max(d) <= stage(cfg, BASIC).max_distance + cfg.scene.distance_tolerance + 0.03


@pytest.mark.parametrize("seed", range(6))
def test_objects_are_on_the_table_in_view_and_on_the_chosen_arms_side(scene, cfg, seed):
    world = scene.reset(np.random.default_rng(seed), stage(cfg, FULL))
    m, sc = scene.model, scene.sc
    cx, cy = m.geom_pos[scene.table_geom][:2]
    hx, hy = m.geom_size[scene.table_geom][:2]
    ball, box = scene.ball_pos(), scene.box_pos()
    assert abs(ball[0] - cx) <= hx - sc.ball_radius and abs(box[0] - cx) <= hx - sc.box_half
    assert scene.camera_visible(ball[None], 0)[0]
    sign = 1 if world["side"] == "left" else -1
    assert sign * world["ball_pos"][1] >= cfg.scene.midline_band - 0.011      # cell jitter
    placed = [b for b in scene.distractor_bodies if scene.data.xpos[b][2] > 0]
    assert len(placed) == sc.n_distractors
    for b in placed:
        p = scene.data.xpos[b]
        assert np.linalg.norm(p[:2] - world["ball_pos"][:2]) > 0.10
        assert np.linalg.norm(p[:2] - world["box_pos"][:2]) > 0.12


def test_a_rolling_ball_slows_down_and_stays_on_the_table(scene, cfg, rng):
    dist = []
    for _ in range(15):
        world = scene.reset(rng, stage(cfg, FULL), {"rolling": True})
        if not scene.sc.rolling:
            continue
        assert np.linalg.norm(scene.ball_vel()[:2]) > 0.1
        _settle(scene, 1500)
        assert not scene.ball_off_table()
        assert np.linalg.norm(scene.ball_vel()) < 0.01                 # it stops (condim 6)
        dist.append(np.linalg.norm(scene.ball_pos()[:2] - world["ball_pos"][:2]))
    assert len(dist) >= 8 and 0.02 < np.median(dist) < 0.15


def test_randomisation_never_compounds(cfg, rng):
    a = Scene(cfg)
    for _ in range(15):
        a.reset(rng, stage(cfg, FULL))
    a.restore_nominal()
    b = Scene(cfg)
    for attr in ("geom_pos", "geom_size", "geom_rbound", "geom_friction", "geom_rgba", "body_mass",
                 "body_inertia", "actuator_gainprm", "actuator_biasprm", "dof_damping", "light_diffuse",
                 "cam_pos", "cam_quat"):
        np.testing.assert_allclose(getattr(a.model, attr), getattr(b.model, attr), err_msg=attr)


def test_same_seed_gives_the_same_world(cfg):
    out = []
    for _ in range(2):
        s = Scene(cfg)
        out.append(s.reset(np.random.default_rng(7), stage(cfg, FULL)))
    assert out[0]["scenario"] == out[1]["scenario"]
    np.testing.assert_allclose(out[0]["ball_pos"], out[1]["ball_pos"])


def test_physics_randomisation_stays_in_range(scene, cfg, rng):
    acts = scene.dr_actuators
    for _ in range(10):
        world = scene.reset(rng, stage(cfg, FULL))
        kp = scene.model.actuator_gainprm[acts, 0] / scene.nominal["actuator_gainprm"][acts, 0]
        assert np.all((kp > 0.79) & (kp < 1.21))
        assert 0 <= world["latency"] <= 2
    world = scene.reset(rng, stage(cfg, BASIC))
    np.testing.assert_allclose(scene.model.actuator_gainprm, scene.nominal["actuator_gainprm"])
    assert world["latency"] == 0
