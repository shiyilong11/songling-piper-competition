import ast
from pathlib import Path

import numpy as np
import pytest

from piper_sim.competition_controller import placement_candidates
from piper_sim.competition_sensor import PlacementCheck, SensorScene, ZoneEstimate, backproject
from piper_sim.control import load_interface
from piper_sim.demo_skill import TargetEstimate
from piper_sim.observation import Observation
from piper_sim.sensor_robot import SensorRobot


def target(x=0.3, y=0, yaw=0, top=0.111):
    return TargetEstimate(
        np.array([x, y, 0.086]),
        0.05,
        yaw,
        np.ones((5, 5), bool),
        np.tile([x, y, top], (25, 1)),
        [2, 2],
        0,
    )


def checker():
    return PlacementCheck(
        ZoneEstimate("red", np.array([[0.2, -0.1], [0.4, -0.1], [0.4, 0.1], [0.2, 0.1]]), 0.061, 0)
    )


def test_requires_two_seconds_of_fresh_stable_observations():
    c = checker()
    assert not any(c.update(target(), t, 0.07) for t in [0, 0.4, 0.8, 1.2, 1.6])
    assert c.update(target(), 2.0, 0.07)


@pytest.mark.parametrize("bad", [None, target(x=0.38), target(y=0.12), target(top=0.135)])
def test_missing_outside_overhanging_or_floating_never_pass(bad):
    c = checker()
    assert not any(c.update(bad, t, 0.07) for t in np.arange(0, 3, 0.4))


def test_occlusion_stale_frame_closed_gripper_or_rotation_reset_timer():
    for interruption in ("occlusion", "duplicate", "closed", "rotation"):
        c = checker()
        for t in [0, 0.4, 0.8, 1.2, 1.6]:
            c.update(target(), t, 0.07)
        if interruption == "occlusion":
            assert not c.update(None, 2.0, 0.07)
        elif interruption == "duplicate":
            assert not c.update(target(), 1.6, 0.07)
        elif interruption == "closed":
            assert not c.update(target(), 2.0, 0.05)
        else:
            assert not c.update(target(yaw=0.2), 2.0, 0.07)
        assert not c.update(target(), 2.4, 0.07)


def test_nan_gripper_width_is_not_valid_release_evidence():
    c = checker()
    assert not any(c.update(target(), t, np.nan) for t in np.arange(0, 3, 0.4))


def test_fruit_rotation_with_fixed_center_resets_stability():
    c = checker()
    c.cube = False
    t = target()
    t.points = np.column_stack((np.linspace(0.26, 0.34, 100), np.zeros(100), np.full(100, 0.1)))
    for stamp in [0, 0.4, 0.8, 1.2, 1.6]:
        assert not c.update(t, stamp, 0.07)
    t.points = np.column_stack(
        (np.full(100, 0.3), np.linspace(-0.04, 0.04, 100), np.full(100, 0.1))
    )
    assert not c.update(t, 2.0, 0.07)


def test_robot_shadow_has_no_scene_and_fk_depends_only_on_encoders():
    state = np.array([0, 0.4, -0.8, 0, 1.2, 0, 0.07])
    robot = SensorRobot(
        load_interface(), lambda: (1.0, state), lambda action: None, lambda target: None
    )
    assert robot.model.nq == 8
    assert robot.model.body("link6").id > 0
    with pytest.raises(KeyError):
        robot.model.geom("target_red")
    old = robot.end_pose()
    state[0] = 0.3
    robot.sync()
    assert not np.allclose(old, robot.end_pose())


def test_perception_does_not_accept_world_argument_and_invalid_depth_rejected():
    obs = Observation(
        0,
        np.zeros(6),
        np.zeros(6),
        0.07,
        np.eye(4),
        np.zeros((2, 2, 3), np.uint8),
        np.ones((2, 2), np.uint16),
        {"fx": 100, "fy": 100, "cx": 1, "cy": 1},
    )
    with pytest.raises(ValueError):
        backproject(obs, np.eye(4))
    obs.depth_m = np.full((2, 2), np.nan)
    cloud, valid = backproject(obs, np.eye(4))
    assert not valid.any() and np.isnan(cloud).all()


@pytest.mark.parametrize("has_card", [True, False])
def test_separates_same_hue_table_cube_and_goal_card(has_card):
    tabletop = [170, 145, 100]
    rgb = np.full((400, 400, 3), tabletop, dtype=np.uint8)
    depth = np.full((400, 400), 0.8)
    cube = np.s_[172:226, 48:102]
    depth[cube] = 0.75
    if has_card:
        rgb[100:300, 150:350] = [240, 210, 20]

    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    transform[:3, 3] = [0.45, 0, 0.8]
    obs = Observation(
        0,
        np.zeros(6),
        np.zeros(6),
        0.07,
        transform,
        rgb,
        depth,
        {"fx": 800, "fy": 800, "cx": 200, "cy": 200},
    )
    scene = SensorScene(np.eye(4), {}, target_edge_m=0.20)
    scene.table = 0.0
    blocks, zones = scene.observe(obs)

    assert len(blocks) == 1
    color, block = blocks[0]
    assert color == "yellow" and block.height == pytest.approx(0.05)
    assert block.mask.sum() == 54 * 54 and block.mask[cube].all()
    assert ("yellow" in zones) is has_card


def test_placement_candidates_cover_inset_corners_of_rotated_target_zone():
    center = np.array([0.58, 0.34])
    angle = np.deg2rad(30)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = (
        center + np.array([[-0.10, -0.10], [0.10, -0.10], [0.10, 0.10], [-0.10, 0.10]]) @ rotation.T
    )
    zone = ZoneEstimate("red", corners, 0.0, 0.0)
    points = list(placement_candidates(zone))
    assert len(points) == 9
    np.testing.assert_allclose(points[0], center)
    assert all(zone.contains(point, 0.045) for point in points)
    assert np.linalg.norm(points[1]) < np.linalg.norm(center) - 0.05


def test_controller_source_has_no_oracle_dependencies():
    root = Path(__file__).resolve().parents[1] / "src/piper_sim"
    forbidden = {
        "world",
        "objects",
        "events_truth",
        "_judge",
        "bounds",
        "evaluation",
        "contact",
        "xpos",
    }
    for name in [
        "competition_controller.py",
        "competition_sensor.py",
        "l2_sensor_controller.py",
        "l3_competition.py",
        "competition_segmentation.py",
    ]:
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden
            if isinstance(node, ast.ImportFrom):
                assert "competition_world" not in (node.module or "")
