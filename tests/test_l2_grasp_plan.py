"""Planner contract with a deliberately off-centre pickup and rotated release."""

from types import SimpleNamespace

import numpy as np
import pytest

from piper_sim.competition_sensor import ZoneEstimate
from piper_sim.demo_skill import TargetEstimate
from piper_sim.l2_sensor_controller import L2SensorController


def fixture():
    center = np.array([0.3, 0.0, 0.09])
    candidate = {
        "position_base_m": (center + [0.02, 0, 0]).tolist(),
        "yaw_rad": 0.0,
        "closing_axis_base": [0, 1, 0],
    }
    target = TargetEstimate(
        center,
        0.04,
        0,
        np.ones((2, 2), bool),
        center + np.array([[-0.03, -0.01, 0], [-0.03, 0.01, 0], [0.03, -0.01, 0], [0.03, 0.01, 0]]),
        [1, 1],
        0,
        grasp_geometry={"candidates": [candidate]},
    )
    zone = ZoneEstimate(
        "red", np.array([[0.4, -0.1], [0.6, -0.1], [0.6, 0.1], [0.4, 0.1]]), 0.07, 1.0
    )
    controller = object.__new__(L2SensorController)
    controller.scene = SimpleNamespace(zones={"red": zone})
    controller.placements = []
    controller.cfg = {}
    controller.motion = SimpleNamespace(cfg={})
    return controller, target


def test_rotated_off_center_grasp_compensates_object_destination():
    controller, target = fixture()
    calls = []

    def plan(grasp, xy, floor):
        calls.append(xy.copy())
        return {
            "center": grasp.center.copy(),
            "hover": grasp.center + [0, 0, 0.1],
            "place": np.r_[xy, floor + 0.02],
            "yaw": grasp.yaw,
            "tilt": 0.0,
            "place_yaw": np.pi / 2,
            "place_tilt": 0.0,
        }

    controller.motion.plan = plan
    result = controller.make_plan(target, "red")
    desired_object_center = np.array([0.5, 0.0])
    assert len(calls) == 2
    assert np.allclose(result["object_destination_estimate_m"][:2], desired_object_center)
    assert not np.allclose(result["place"][:2], desired_object_center)
    assert np.allclose(result["grasp_geometry"]["actual_approach_direction_base"], [0, 0, -1])
    assert abs(
        np.dot(result["grasp_geometry"]["actual_closing_axis_base"], [0, 1, 0])
    ) == pytest.approx(1)
    assert controller.motion.cfg["grasp_yaw_candidates_rad"] == [np.pi, 0]


def test_no_fallback_to_center_grasp_when_geometry_rejects_all_candidates():
    controller, target = fixture()
    target.grasp_geometry["candidates"] = []
    with pytest.raises(ValueError, match="No mask-derived"):
        controller.make_plan(target, "red")


def test_rejects_large_wrist_rotation_and_tries_another_slot():
    controller, target = fixture()
    calls = 0

    def plan(grasp, xy, floor):
        nonlocal calls
        calls += 1
        return {
            "center": grasp.center.copy(),
            "hover": grasp.center + [0, 0, 0.1],
            "place": np.r_[xy, floor + 0.02],
            "yaw": grasp.yaw,
            "tilt": 0.0,
            "place_yaw": 0.8 * np.pi if calls <= 2 else grasp.yaw,
        }

    controller.motion.plan = plan
    result = controller.make_plan(target, "red")
    assert controller.last_plan_diagnostics["rotation_rejections"] == 1
    assert result["grasp_geometry"]["transport_rotation_rad"] == pytest.approx(0)


def test_edge_alignment_fallback_centres_long_object_along_free_slot():
    controller, target = fixture()
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_euler("z", np.pi / 4).as_matrix()
    target.points = (
        target.center
        + np.array(
            [[-0.065, -0.0125, 0], [-0.065, 0.0125, 0], [0.065, -0.0125, 0], [0.065, 0.0125, 0]]
        )
        @ rotation.T
    )
    target.grasp_geometry.update(major_axis_reliable=True, metric_major_axis_base=rotation[:, 0])
    target.grasp_geometry["candidates"][0]["yaw_rad"] = np.pi / 4
    controller.placements = [
        {
            "destination": "red",
            "center": np.array([0.53, 0.0, 0.09]),
            "span": np.array([0.05, 0.08]),
        }
    ]

    def plan(grasp, xy, floor):
        return {
            "center": grasp.center.copy(),
            "hover": grasp.center + [0, 0, 0.1],
            "place": np.r_[xy, floor + 0.02],
            "yaw": grasp.yaw,
            "tilt": 0.0,
            "place_yaw": controller.motion.cfg["place_yaw_rad"],
        }

    controller.motion.plan = plan
    result = controller.make_plan(target, "red")
    center = result["object_destination_estimate_m"][:2]
    assert center[0] < 0.5
    assert center[1] == pytest.approx(0.0)


def test_replanning_moved_object_excludes_only_its_own_reservation():
    controller, target = fixture()
    controller.placements = [
        {
            "id": "retry_01",
            "destination": "red",
            "center": np.array([0.5, 0, 0.09]),
            "span": np.array([0.2, 0.2]),
        }
    ]
    controller.motion.plan = lambda grasp, xy, floor: {
        "center": grasp.center.copy(),
        "hover": grasp.center + [0, 0, 0.1],
        "place": np.r_[xy, floor + 0.02],
        "yaw": grasp.yaw,
        "tilt": 0.0,
    }
    with pytest.raises(ValueError, match="No reachable"):
        controller.make_plan(target, "red")
    plan = controller.make_plan(target, "red", ignore_id="retry_01")
    assert np.allclose(plan["object_destination_estimate_m"][:2], [0.5, 0])
    assert len(controller.placements) == 1  # reservation is not erased before execution


@pytest.mark.parametrize("category,reserve_strip", [("potato", False), ("cube_yellow", True)])
def test_compact_object_releases_to_side_of_shared_zone_to_leave_room_for_banana(
    category, reserve_strip
):
    controller, target = fixture()
    controller.cfg.update(
        l2_precise_packing=True, l2_color_mapping={"potato": "red", "banana": "red"}
    )
    controller.motion.plan = lambda grasp, xy, floor: {
        "center": grasp.center.copy(),
        "hover": grasp.center + [0, 0, 0.1],
        "place": np.r_[xy, floor + 0.02],
        "yaw": grasp.yaw,
        "tilt": 0.0,
        "place_yaw": controller.motion.cfg["place_yaw_rad"],
    }
    plan = controller.make_plan(
        target, "red", category=category, reserve_lateral_strip=reserve_strip
    )
    assert np.allclose(plan["object_destination_estimate_m"][:2], [0.5, 0.045])
