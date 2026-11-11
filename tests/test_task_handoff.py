import json

import numpy as np
import pytest

from piper_sim.multiframe_demo import make_demo_frame
from piper_sim.multiframe_task import MultiframeTaskSession
from piper_sim.qwen_adapter import InvalidPrediction
from piper_sim.scene_memory import SceneMemory, load_memory_config
from piper_sim.scene_observation import Provenance
from piper_sim.task_handoff import ExecutionFeedback, TaskHandoffSession, load_handoff

INSTRUCTION = "将两只柿子依次放入蓝色托盘。"


def decide(brain, targets, *, now_s, progress=None):
    class FixtureQwen:
        def predict_messages(self, images, messages):
            if progress is not None:
                assert "task_progress" in messages[1]["content"][-1]["text"]
            return json.dumps(
                {
                    "status": "ready",
                    "memory_version": brain.memory.version,
                    "spatial_intent": "none",
                    "relation_checks": [],
                    "actions": [
                        {"target_instance_id": target, "destination_region_id": "blue_tray"}
                        for target in targets
                    ],
                    "observe_region": None,
                    "message": "仅为离线接口夹具",
                }
            ), {"mock_only": True}

    return brain.decide(FixtureQwen(), INSTRUCTION, now_s=now_s, task_progress=progress)[0]


def setup(targets=("obj_002",)):
    cfg = load_memory_config()
    cfg["required_views"] = ["left", "right"]
    brain = MultiframeTaskSession(SceneMemory(cfg))
    brain.observe(**make_demo_frame(0.14, 0.14, 0.0, view_id="left"))
    brain.observe(**make_demo_frame(-0.14, -0.14, 1.0, view_id="right"))
    decision = decide(brain, targets, now_s=1.0)
    ledger = TaskHandoffSession(brain)
    ledger.accept_plan(decision, now_s=1.0)
    return brain, ledger


def feedback(packet, status, stamp, *, event_id=None, source=None, evidence=None, error_code=None):
    meta = packet.metadata
    if evidence is None:
        evidence = {
            "observation_timestamp_s": stamp,
            "target_instance_id": meta["target_instance_id"],
            "destination_region_id": meta["destination_region_id"],
        }
        keys = {
            "picked": "grasp_confirmed",
            "placed": "release_confirmed",
            "verified": "placement_verified",
        }
        if status in keys:
            evidence[keys[status]] = True
    return ExecutionFeedback(
        event_id or status + str(stamp),
        meta["task_id"],
        meta["step_id"],
        meta["handoff_id"],
        status,
        stamp,
        source or Provenance("sensor_estimate", "Synthetic feedback fixture; no robot executed"),
        evidence,
        error_code,
    )


def finish(ledger, packet, start=1.02):
    for i, status in enumerate(("started", "picked", "placed", "verified")):
        stamp = start + i * 0.1
        ledger.apply_feedback(feedback(packet, status, stamp), now_s=stamp)


def test_export_contains_aligned_depth_mask_state_and_policy_features_without_rgb(tmp_path):
    brain, ledger = setup()
    result = ledger.prepare_current(now_s=1.01)
    assert result.status == "prepared"
    packet = result.packet
    assert packet.arrays["target_points"].shape == (512, 3)
    assert packet.arrays["scene_points"].shape == (1024, 3)
    assert packet.arrays["target_valid_mask"].sum() > 12
    assert packet.arrays["target_mask"].shape == packet.arrays["aligned_depth_m"].shape
    assert packet.arrays["joint_position_rad"].shape == (6,)
    np.testing.assert_allclose(
        packet.arrays["T_end_goal"],
        np.linalg.inv(packet.arrays["T_base_end"]) @ packet.arrays["T_base_goal"],
        atol=1e-6,
    )
    assert not packet.metadata["rgb_included"]
    assert not packet.metadata["robot_control_enabled"]
    assert packet.metadata["grasp_pose_status"] == "pending_grasp_planner"
    path = packet.export(tmp_path / "handoff")
    manifest = json.loads(path.read_text())
    loaded = load_handoff(path, now_s=1.02)
    assert loaded.metadata["handoff_id"] == packet.metadata["handoff_id"]
    np.testing.assert_array_equal(loaded.arrays["target_mask"], packet.arrays["target_mask"])
    with pytest.raises(ValueError, match="stale"):
        load_handoff(path, now_s=2.0)
    with np.load(path.parent / manifest["array_file"], allow_pickle=False) as arrays:
        assert "rgb" not in arrays and arrays["depth_valid_mask"].dtype == bool
        assert arrays["target_points"].shape == (512, 3)
    again = ledger.prepare_current(now_s=1.02).packet
    assert again.metadata["handoff_id"] == packet.metadata["handoff_id"]
    assert brain.last_inference["metrics"]["mock_only"]


def test_packet_array_integrity_is_checked_before_consumer_access(tmp_path):
    _, ledger = setup()
    packet = ledger.prepare_current(now_s=1.01).packet
    path = packet.export(tmp_path / "handoff")
    arrays = path.parent / "observation.npz"
    arrays.write_bytes(arrays.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_handoff(path, now_s=1.02)


def test_historical_target_requests_observation_instead_of_returning_old_mask():
    _, ledger = setup(("obj_001",))
    result = ledger.prepare_current(now_s=1.01)
    assert result.status == "need_observation" and result.packet is None
    assert result.observe_region == "object:obj_001"


def test_only_verified_feedback_completes_a_step_and_duplicate_feedback_is_idempotent():
    _, ledger = setup()
    packet = ledger.prepare_current(now_s=1.01).packet
    for i, status in enumerate(("started", "picked", "placed")):
        stamp = 1.02 + i * 0.1
        ledger.apply_feedback(feedback(packet, status, stamp), now_s=stamp)
    assert ledger.state()["phase"] == "awaiting_verification"
    assert ledger.progress_context()["completed_target_ids"] == []
    event = feedback(packet, "verified", 1.32)
    assert ledger.apply_feedback(event.as_dict(), now_s=1.32)["phase"] == "completed"
    assert ledger.apply_feedback(event, now_s=10.0)["duplicate"]
    assert ledger.progress_context()["completed_target_ids"] == ["obj_002"]
    changed = ExecutionFeedback(**{**event.__dict__, "error_code": "different-content"})
    with pytest.raises(ValueError, match="reused"):
        ledger.apply_feedback(changed, now_s=10.0)


def test_progress_feedback_is_ordered_and_cannot_claim_unmeasured_success():
    _, ledger = setup()
    packet = ledger.prepare_current(now_s=1.01).packet
    with pytest.raises(ValueError, match="transition"):
        ledger.apply_feedback(feedback(packet, "verified", 1.02), now_s=1.02)
    ledger.apply_feedback(feedback(packet, "started", 1.02), now_s=1.02)
    with pytest.raises(ValueError, match="sensor evidence"):
        ledger.apply_feedback(feedback(packet, "picked", 1.1, evidence={}), now_s=1.1)
    false = feedback(packet, "picked", 1.1)
    false.evidence["grasp_confirmed"] = False
    with pytest.raises(ValueError, match="does not confirm"):
        ledger.apply_feedback(false, now_s=1.1)
    with pytest.raises(ValueError, match="oracle forbidden"):
        ledger.apply_feedback(
            feedback(packet, "picked", 1.1, source=Provenance("oracle", "simulator body truth")),
            now_s=1.1,
        )


def test_new_observation_requires_new_handoff_without_repeating_qwen():
    brain, ledger = setup()
    old = ledger.prepare_current(now_s=1.01).packet
    brain.observe(**make_demo_frame(-0.14, -0.14, 1.05, view_id="right"))
    with pytest.raises(ValueError, match="expired or changed"):
        ledger.apply_feedback(feedback(old, "started", 1.06), now_s=1.06)
    fresh = ledger.prepare_current(now_s=1.06).packet
    assert fresh.metadata["handoff_id"] != old.metadata["handoff_id"]
    assert fresh.metadata["decision_memory_version"] == 2
    assert fresh.metadata["binding_memory_version"] == 3
    with pytest.raises(ValueError, match="current handoff"):
        ledger.apply_feedback(feedback(old, "started", 1.07), now_s=1.07)
    assert (
        ledger.apply_feedback(feedback(fresh, "started", 1.07), now_s=1.07)["phase"] == "executing"
    )


def test_expired_packets_cannot_start_and_stale_feedback_cannot_change_state():
    _, ledger = setup()
    packet = ledger.prepare_current(now_s=1.01).packet
    with pytest.raises(ValueError, match="expired"):
        ledger.apply_feedback(feedback(packet, "started", 1.3), now_s=1.3)
    with pytest.raises(ValueError, match="stale"):
        ledger.apply_feedback(feedback(packet, "started", 1.02), now_s=5.0)
    with pytest.raises(ValueError, match="precedes"):
        ledger.apply_feedback(feedback(packet, "started", 1.0), now_s=1.02)
    assert ledger.state()["phase"] == "prepared"


def test_multi_step_feedback_requires_fresh_replanning_and_carries_completed_ids():
    brain, ledger = setup(("obj_002", "obj_001"))
    packet = ledger.prepare_current(now_s=1.01).packet
    finish(ledger, packet)
    assert ledger.state()["phase"] == "needs_replan"
    assert ledger.prepare_current(now_s=1.4).status == "needs_replan"
    with pytest.raises(ValueError, match="Reobserve"):
        ledger.refresh_remaining(brain.last_decision, now_s=1.4)
    brain.observe(**make_demo_frame(0.14, 0.14, 2.0, view_id="left"))
    context = ledger.progress_context()
    assert context["completed_target_ids"] == ["obj_002"]
    next_plan = decide(brain, ("obj_001",), now_s=2.0, progress=context)
    ledger.refresh_remaining(next_plan, now_s=2.0)
    next_packet = ledger.prepare_current(now_s=2.01).packet
    assert next_packet.metadata["target_instance_id"] == "obj_001"
    assert next_packet.metadata["plan_revision"] == 2
    assert next_packet.metadata["step_id"] != packet.metadata["step_id"]
    finish(ledger, next_packet, start=2.02)
    assert ledger.state()["phase"] == "completed"
    assert set(ledger.progress_context()["completed_target_ids"]) == {"obj_001", "obj_002"}


def test_failure_is_not_completion_and_recovery_does_not_dispatch_automatically():
    _, ledger = setup()
    packet = ledger.prepare_current(now_s=1.01).packet
    ledger.apply_feedback(
        feedback(packet, "failed", 1.02, error_code="planning_failed"), now_s=1.02
    )
    assert ledger.state()["phase"] == "needs_replan"
    assert ledger.progress_context()["completed_target_ids"] == []
    assert ledger.progress_context()["failed_target_ids"] == ["obj_002"]
    assert ledger.prepare_current(now_s=1.03).packet is None


def test_replanning_cannot_repeat_an_already_verified_completed_target():
    brain, ledger = setup(("obj_002", "obj_001"))
    finish(ledger, ledger.prepare_current(now_s=1.01).packet)
    brain.observe(**make_demo_frame(0.14, 0.14, 2.0, view_id="left"))
    with pytest.raises(InvalidPrediction, match="already verified"):
        decide(brain, ("obj_002",), now_s=2.0, progress=ledger.progress_context())


def test_verified_feedback_must_confirm_the_assigned_destination():
    _, ledger = setup()
    packet = ledger.prepare_current(now_s=1.01).packet
    for i, status in enumerate(("started", "picked", "placed")):
        stamp = 1.02 + i * 0.1
        ledger.apply_feedback(feedback(packet, status, stamp), now_s=stamp)
    wrong = feedback(packet, "verified", 1.32)
    wrong.evidence["destination_region_id"] = "another_tray"
    with pytest.raises(ValueError, match="destination"):
        ledger.apply_feedback(wrong, now_s=1.32)
    assert ledger.state()["phase"] == "awaiting_verification"


def test_remaining_plan_must_explicitly_include_this_tasks_progress():
    brain, ledger = setup(("obj_002", "obj_001"))
    finish(ledger, ledger.prepare_current(now_s=1.01).packet)
    brain.observe(**make_demo_frame(0.14, 0.14, 2.0, view_id="left"))
    next_plan = decide(brain, ("obj_001",), now_s=2.0)
    with pytest.raises(ValueError, match="feedback progress"):
        ledger.refresh_remaining(next_plan, now_s=2.0)


def test_replan_budget_is_bounded_and_new_brain_decision_blocks_old_task():
    brain, ledger = setup()
    decide(brain, ("obj_001",), now_s=1.0)
    assert ledger.prepare_current(now_s=1.01).code == "understanding_superseded"
    ledger.max_plan_revisions = 1
    with pytest.raises(ValueError, match="budget exhausted"):
        ledger.refresh_remaining(brain.last_decision, now_s=1.01)
