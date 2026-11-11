from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from piper_sim.competition_sensor import ZoneEstimate
from piper_sim.demo_skill import TargetEstimate
from piper_sim.grasp_feedback import GraspEmptyError, GraspFeedbackError
from piper_sim.l2_progress import L2Progress
from piper_sim.l2_sensor_controller import L2SensorController


def target(x=0.3, y=0):
    center = np.array([x, y, 0.085])
    return TargetEstimate(
        center,
        0.05,
        0,
        np.ones((3, 3), bool),
        center + np.array([[-0.02, 0, 0], [0.02, 0, 0], [0, 0.02, 0]]),
        [1, 1],
        0,
        np.array([200.0, 30.0, 20.0]),
    )


def test_retries_keep_id_and_do_not_count_release_or_repeat_verified_object():
    ledger = L2Progress()
    item = ledger.start("red", "red", target())
    ledger.expect_release(item, np.array([0.6, 0, 0.085]))
    assert ledger.completed == []
    misplaced = target(0.65)
    ledger.fail(item, "outside", misplaced)
    retried = ledger.start("red", "red", misplaced)
    assert retried is item and retried["attempts"] == 2
    ledger.verify(item, target(0.6), 10.0)
    assert ledger.completed == [item["id"]]
    with pytest.raises(ValueError, match="verified"):
        ledger.start("red", "red", target(0.6))
    ledger.fail(item, "moved_during_other_pick", target(0.65))
    assert ledger.completed == []


def test_ambiguous_identity_does_not_silently_bind_or_create_new_object():
    ledger = L2Progress()
    ledger.start("red", "red", target(0.3))
    ledger.start("red", "red", target(0.45))
    with pytest.raises(ValueError, match="Ambiguous"):
        ledger.start("red", "red", target(0.375))
    assert len(ledger.items) == 2


def test_color_boundary_change_with_same_task_destination_does_not_duplicate_identity():
    ledger = L2Progress(
        color_mapping={"green": "green", "yellow_green": "green", "yellow": "yellow"}
    )
    item = ledger.start("yellow_green", "green", target())
    ledger.verify(item, target(), 10.0)
    assert ledger.associate("green", target(0.301)) is item
    assert ledger.associate("yellow", target(0.301)) is None


def test_failed_placement_reserves_observed_space_without_counting_success(tmp_path):
    c = controller(tmp_path)
    t = target()
    item = c.progress.start("red", "red", t)
    c.progress.fail(item, "outside", t)
    c.reserve_observed_footprint(item, t)
    assert c.progress.completed == []
    assert item["status"] == "retry"
    assert len(c.placements) == 1
    assert np.allclose(c.placements[0]["center"], t.center)
    moved = target(0.32)
    c.reserve_observed_footprint(item, moved)
    assert len(c.placements) == 1
    assert np.allclose(c.placements[0]["center"], moved.center)


@pytest.mark.parametrize(
    "observed_color,appearance_ok", [("green", True), ("yellow", True), ("green", False)]
)
def test_postrelease_color_alias_still_requires_identity_and_stability(
    tmp_path, observed_color, appearance_ok
):
    c = controller(tmp_path)
    mapping = {"green": "green", "yellow_green": "green", "yellow": "yellow"}
    c.cfg["l2_color_mapping"] = mapping
    c.progress = L2Progress(color_mapping=mapping)
    t = target()
    t.mask = np.zeros((20, 20), bool)
    t.mask[8:12, 8:12] = True
    item = c.progress.start("yellow_green", "green", t)
    c.progress.expect_release(item, t.center)
    if not appearance_ok:
        t.appearance_rgb = np.array([10.0, 200.0, 10.0])
    c.zone_views = {"green": np.zeros(6)}
    c.scene = SimpleNamespace(
        zones={
            "green": ZoneEstimate(
                "green", np.array([[0.1, -0.1], [0.5, -0.1], [0.5, 0.1], [0.1, 0.1]]), 0.06, 0.0
            )
        }
    )
    c.motion_cfg = {"grasp_feedback": {}}
    c.motion.move = Mock()
    c.motion.hold = lambda s: setattr(c.robot.data, "time", c.robot.data.time + s)
    c.observe = lambda: (None, None, None)
    c.detect = lambda obs, **_: [SimpleNamespace(color=observed_color, target=t)]
    c.visible_geometry = lambda **_: (
        SimpleNamespace(
            timestamp_s=c.robot.data.time, gripper_width_m=0.07, joint_position_rad=np.zeros(6)
        ),
        [t],
    )
    result = L2SensorController.verify_fruit(c, "yellow_green", "green", t.center, item=item)
    assert (result is not None) == (observed_color == "green" and appearance_ok)
    if result is not None:
        assert c.robot.data.time >= 2.0
    assert c.progress.completed == []  # verification observation does not mutate progress


def controller(tmp_path):
    c = object.__new__(L2SensorController)
    c.output = tmp_path
    c.cfg = {"l2_color_mapping": {"red": "red"}, "l2_mapping_status": "test_fixture"}
    c.progress = L2Progress()
    c.robot = SimpleNamespace(data=SimpleNamespace(time=0.0))
    c.motion = SimpleNamespace(
        pick=Mock(return_value={"holding_estimate": True}),
        place=Mock(),
        recover_empty=Mock(),
        hold_current=Mock(),
        hold=Mock(),
    )
    c.motion.pick_closed_loop = c.motion.pick
    c.motion.place_closed_loop = c.motion.place
    c.segmenter = SimpleNamespace(close=Mock())
    c.events, c.placements, c.completed = [], [], []
    c.verified_zone_views = {}
    c.inference_s = 0.0
    c.callback = Mock()
    c.scan = Mock()
    c.last_verification = {"reason": "outside"}
    c.last_seen_target = None
    c.find_count = 0

    def find():
        c.find_count += 1
        retry = next((i for i in c.progress.items if i["status"] == "retry"), None)
        t = target(retry["center"][0] if retry else 0.3 + len(c.progress.items) * 0.3)
        c.robot.data.time += 1.0
        return (
            SimpleNamespace(color="red", key=f"frame_{c.find_count}"),
            t,
            {"place": t.center.copy()},
        )

    c.find_target = find
    c.verify_fruit = lambda color, dest, expected, item=None: target(expected[0])

    def final_group(destination, items):
        return {
            item["id"]: c.verify_fruit(
                item["color"], destination, np.array(item["center"]), item=item
            )
            for item in items
        }

    c.verify_final_group = final_group
    return c


def test_empty_pick_and_bad_placement_reobserve_retry_then_final_verify(tmp_path):
    c = controller(tmp_path)
    c.motion.pick.side_effect = [GraspEmptyError()] + [{"holding_estimate": True}] * 7
    count = 0

    def verify(color, dest, expected, item=None):
        nonlocal count
        count += 1
        return None if count == 1 else target(expected[0])

    c.verify_fruit = verify
    result = c.run()
    assert result["placement_success"] and result["final_rechecked"]
    assert len(set(result["completed_instances"])) == 6
    assert c.progress.items[0]["attempts"] == 3
    assert c.find_count == 8
    c.motion.recover_empty.assert_called_once()
    assert count == 13  # failed placement + 6 successes + 6 final rechecks
    assert {e["event"] for e in result["events"]} >= {
        "grasp_failure",
        "placement_retry",
        "object_verified",
    }
    c.segmenter.close.assert_called_once()


def test_timeout_preserves_partial_progress_but_never_claims_final_success(tmp_path):
    c = controller(tmp_path)
    original = c.find_target

    def find():
        if len(c.progress.completed) == 1:
            raise TimeoutError("round limit")
        return original()

    c.find_target = find
    result = c.run()
    assert len(result["completed_instances"]) == 1
    assert not result["placement_success"] and not result["final_rechecked"]
    c.motion.hold_current.assert_called_once()


def test_missing_target_starts_an_extended_scan(tmp_path):
    c = controller(tmp_path)
    original = c.find_target
    scan_modes = []

    def find(*, extended_scan=False):
        scan_modes.append(extended_scan)
        if len(scan_modes) == 1:
            raise ValueError("No target in the initial views")
        return original()

    c.find_target = find
    result = c.run()
    assert result["placement_success"]
    assert scan_modes[:2] == [False, True]
    assert not any(scan_modes[2:])


def test_timeout_during_final_recheck_never_marks_recheck_complete(tmp_path):
    c = controller(tmp_path)
    c.verify_final_group = Mock(side_effect=TimeoutError("round limit during final view"))
    result = c.run()
    assert len(result["completed_instances"]) == 6
    assert not result["final_rechecked"]
    assert not result["placement_success"]


def test_uncertain_grip_is_not_treated_as_empty_and_blindly_retried(tmp_path):
    c = controller(tmp_path)
    c.motion.pick.side_effect = GraspFeedbackError("invalid encoder", "grip_feedback_invalid")
    result = c.run()
    assert not result["placement_success"] and result["completed_instances"] == []
    c.motion.place.assert_not_called()
    c.motion.recover_empty.assert_not_called()
    c.motion.hold_current.assert_called_once()


def test_final_recheck_failure_revokes_count_and_replans_same_object(tmp_path):
    c = controller(tmp_path)
    count = 0

    def verify(color, dest, expected, item=None):
        nonlocal count
        count += 1
        return None if count == 7 else target(expected[0])

    c.verify_fruit = verify
    result = c.run()
    assert result["placement_success"] and len(result["objects"]) == 6
    assert result["objects"][0]["attempts"] == 2
    assert any(e["event"] == "completion_revoked" for e in result["events"])


@pytest.mark.parametrize("second_visible", [True, False])
@pytest.mark.parametrize("reuse_geometry", [True, False])
def test_actual_group_verifier_needs_each_objects_own_stable_evidence(
    tmp_path, second_visible, reuse_geometry
):
    c = controller(tmp_path)
    a, b = target(0.2), target(0.4)
    for t in [a, b]:
        t.mask = np.zeros((20, 20), bool)
        t.mask[8:12, 8:12] = True
    items = [c.progress.start("red", "red", a), c.progress.start("red", "red", b)]
    for item, t in zip(items, [a, b], strict=True):
        c.progress.verify(item, t, 0.0)
        if reuse_geometry:
            c.reserve_observed_footprint(item, t)
            c.placements[-1]["verified_geometry"] = t
    c.zone_views = {"red": np.zeros(6)}
    c.scene = SimpleNamespace(
        zones={
            "red": ZoneEstimate(
                "red", np.array([[0.1, -0.1], [0.5, -0.1], [0.5, 0.1], [0.1, 0.1]]), 0.06, 0.0
            )
        }
    )
    c.motion_cfg = {"grasp_feedback": {}}
    c.motion.move = Mock()
    c.motion.hold = lambda seconds: setattr(c.robot.data, "time", c.robot.data.time + seconds)
    c.observe = lambda: (None, None, None)
    c.detect = Mock(return_value=[SimpleNamespace(color="red", target=t) for t in [a, b]])
    c.visible_geometry = lambda **_: (
        SimpleNamespace(timestamp_s=c.robot.data.time, gripper_width_m=0.07),
        [a, b] if second_visible else [a],
    )
    result = L2SensorController.verify_final_group(c, "red", items)
    assert set(result) == {item["id"] for item in (items if second_visible else items[:1])}
    assert c.detect.called == (not reuse_geometry or not second_visible)
