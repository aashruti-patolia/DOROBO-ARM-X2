"""Camera perception: the ball and basket are located from the head camera only."""
import numpy as np
import pytest

from conftest import BASIC, FULL, stage
from dual_arm_rl.perception.camera import Detection, HeadCamera, ObjectTracker


@pytest.fixture(scope="module")
def cam(scene, cfg):
    c = HeadCamera(scene.model, cfg.camera)
    yield c
    c.close()


def test_ray_cast_depth_equals_the_rendered_depth_buffer(scene, cfg, cam, rng):
    scene.reset(rng, stage(cfg, BASIC))
    red, blue, neutral = cam.segment(cam.render_rgb(scene.data))
    idx = np.flatnonzero((red | blue).ravel())
    assert idx.size > 50
    ray = cam.ray_depth(scene.data, idx)
    gpu = cam.render_depth(scene.data).ravel()[idx]
    assert np.median(np.abs(ray - gpu)) < 1e-3


def test_back_projection_recovers_the_hit_points(scene, cfg, cam, rng):
    scene.reset(rng, stage(cfg, BASIC))                       # no visual randomisation: nominal = real
    idx = np.random.default_rng(0).choice(cfg.camera.width * cfg.camera.height, 100, replace=False)
    depth = cam.ray_depth(scene.data, idx)
    ok = depth > 0
    pts = cam.backproject(idx[ok], depth[ok], *cam.nominal_pose(scene.data))
    cam_pos = scene.data.cam_xpos[cam.cam_id]
    dirs = (pts - cam_pos) / np.linalg.norm(pts - cam_pos, axis=1, keepdims=True)
    # every recovered point lies on the floor, the table or an object: below the camera, in front of it
    assert np.all(pts[:, 2] > -1e-3) and np.all(dirs @ scene.data.cam_xmat[cam.cam_id].reshape(3, 3)[:, 2] < 0)


@pytest.mark.parametrize("k,max_err_mm", [(BASIC, 6.0), (FULL, 12.0)])
def test_the_ball_and_basket_are_located_accurately(scene, cfg, cam, k, max_err_mm):
    rng = np.random.default_rng(5)
    st = stage(cfg, k)
    seen, eb, ex, et = 0, [], [], []
    for _ in range(40):
        scene.reset(rng, st)
        det = cam.capture(scene.data, rng, st.visual_level)
        if det.ball_pos is None or det.box_pos is None:
            continue
        seen += 1
        eb.append(np.linalg.norm(det.ball_pos - scene.ball_pos()))
        ex.append(np.linalg.norm(det.box_pos[:2] - scene.box_pos()[:2]))
        assert det.box_pos[2] == pytest.approx(scene.sc.box_rim_z, abs=0.012)       # rim height
        et.append(abs(det.table_z - scene.sc.table_z))
    assert seen >= 38
    assert 1000 * np.mean(eb) < max_err_mm and 1000 * np.mean(ex) < max_err_mm
    assert 1000 * np.mean(et) < 4.0


def test_the_ball_size_and_basket_size_are_estimated(scene, cfg, cam):
    rng = np.random.default_rng(9)
    er, eh = [], []
    for _ in range(20):
        scene.reset(rng, stage(cfg, BASIC))
        det = cam.capture(scene.data)
        er.append(det.ball_radius - scene.sc.ball_radius)
        eh.append(det.box_half - scene.sc.box_half)
    assert abs(np.mean(er)) < 0.006 and abs(np.mean(eh)) < 0.006


def test_an_unseen_ball_keeps_its_last_estimate_and_ages(cfg):
    tr = ObjectTracker(cfg.camera, dt_tick=0.04)
    tr.update(Detection(ball_pos=np.array([0.3, 0.2, 0.27]), ball_radius=0.02, ball_pixels=40,
                        box_pos=np.array([0.3, -0.1, 0.31]), box_half=0.045, table_z=0.25))
    for _ in range(2):
        tr.tick()
        tr.update(Detection())                       # occluded by the hand
    s = tr.state
    assert not s.ball_visible and s.ball_age == 2 and s.ball_ever_seen
    np.testing.assert_allclose(s.ball_pos, [0.3, 0.2, 0.27], atol=1e-9)
    np.testing.assert_allclose(s.box_pos, [0.3, -0.1, 0.31])


def test_velocity_is_estimated_from_successive_frames(cfg):
    tr = ObjectTracker(cfg.camera, dt_tick=0.04)
    for k in range(8):
        tr.tick()
        tr.update(Detection(ball_pos=np.array([0.3 + 0.004 * k, 0.2, 0.27]), ball_pixels=40))
    assert tr.state.ball_vel[0] == pytest.approx(0.1, rel=0.1)          # 4 mm per 40 ms


def test_the_camera_choice_of_arm_matches_the_side_of_the_ball(scene, cfg, cam):
    rng = np.random.default_rng(11)
    for _ in range(20):
        world = scene.reset(rng, stage(cfg, FULL))
        det = cam.capture(scene.data, rng, 1.0)
        if det.ball_pos is not None:
            assert (det.ball_pos[1] > 0) == (world["side"] == "left")
