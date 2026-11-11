import copy
import json

import numpy as np
import pytest

from piper_sim.multiframe_demo import create_demo_bundle, make_demo_frame
from piper_sim.multiframe_io import read_frame
from piper_sim.multiframe_task import MultiframeTaskSession, parse_multiframe_decision
from piper_sim.qwen_adapter import InvalidPrediction
from piper_sim.scene_memory import SceneMemory, load_memory_config
from piper_sim.scene_observation import SceneObserver, policy_scene_features


def session():
    cfg = load_memory_config()
    cfg["required_views"] = ["left", "right"]
    return MultiframeTaskSession(SceneMemory(cfg))


def two_views():
    task = session()
    task.observe(**make_demo_frame(0.14, 0.14, 0.0, view_id="left"))
    task.observe(**make_demo_frame(-0.14, -0.14, 1.0, view_id="right"))
    return task


def decision(task, target="obj_001"):
    return {
        "status": "ready",
        "memory_version": task.memory.version,
        "actions": [{"target_instance_id": target, "destination_region_id": "blue_tray"}],
        "observe_region": None,
        "message": "根据固定坐标选择目标",
        "relation_checks": [],
        "spatial_intent": "none",
    }


def test_disjoint_views_keep_two_objects_and_global_geometry():
    task = two_views()
    snapshot = task.memory.summary()
    assert len(snapshot["objects"]) == 2
    assert [len(frame.instances) for frame in task.memory.frames] == [1, 1]
    first, second = snapshot["objects"]
    assert first["last_observed_center_base_m"][1] > second["last_observed_center_base_m"][1]
    assert not first["currently_visible"] and second["currently_visible"]
    assert snapshot["coverage"]["survey_complete"]
    assert all(not row["geometry_complete"] for row in snapshot["objects"])
    # Both images centre their only object. Image-local position cannot establish global order.
    for frame in task.memory.frames:
        ys, xs = np.where(frame.scene.objects[0].observed_mask)
        np.testing.assert_allclose([xs.mean(), ys.mean()], [160, 120])


def test_camera_motion_does_not_create_duplicate_identity():
    task = session()
    task.observe(**make_demo_frame(0.14, 0.14, 0.0, view_id="left"))
    task.observe(**make_demo_frame(0.14, 0.15, 1.0, view_id="left"))
    assert list(task.memory.objects) == ["obj_001"]
    assert task.memory.objects["obj_001"].last_frame_id == "frame_000002"
    assert task.memory.summary()["coverage"]["missing_views"] == ["right"]


def test_ambiguous_association_is_not_silently_merged():
    task = session()
    task.observe(**make_demo_frame(0.10, 0.10, 0.0, view_id="left"))
    # Different colour prevents association while introducing the second nearby object.
    other = make_demo_frame(0.13, 0.13, 1.0, view_id="right")
    other["observation"].rgb = other["observation"].rgb.copy()
    other["observation"].rgb[other["detections"][0].mask] = [20, 220, 30]
    task.observe(**other)
    # Remove appearance discrimination for this test; geometry alone is ambiguous.
    task.memory.objects["obj_002"].color = task.memory.objects["obj_001"].color.copy()
    task.observe(**make_demo_frame(0.115, 0.115, 2.0, view_id="left"))
    newest = task.memory.objects["obj_003"]
    assert set(newest.unresolved_candidates) == {"obj_001", "obj_002"}
    with pytest.raises(InvalidPrediction, match="ambiguous"):
        parse_multiframe_decision(
            json.dumps(decision(task, "obj_003")), task.memory.summary(), task.memory.cfg
        )


def test_keyframe_budget_is_bounded_and_retains_current_view():
    task = session()
    task.memory.cfg["max_keyframes"] = 2
    for i in range(8):
        view = "left" if i % 2 == 0 else "right"
        y = 0.14 if view == "left" else -0.14
        task.observe(**make_demo_frame(y, y, float(i), view_id=view))
    assert len(task.memory.frames) == 2
    assert task.memory.latest_frame.frame_id in {f.frame_id for f in task.memory.frames}
    assert len(task.memory.objects) == 2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x.update(memory_version=0),
        lambda x: x["actions"][0].update(target_instance_id="invented"),
        lambda x: x["actions"][0].update(destination_region_id="invented"),
        lambda x: x["actions"][0].update(joint_command=[0] * 7),
        lambda x: x["actions"].append(copy.deepcopy(x["actions"][0])),
    ],
)
def test_unsafe_or_stale_qwen_decisions_are_rejected(mutate):
    task = two_views()
    value = decision(task)
    mutate(value)
    with pytest.raises(InvalidPrediction):
        parse_multiframe_decision(json.dumps(value), task.memory.summary(), task.memory.cfg)


def test_incomplete_survey_requires_observation_not_a_guess():
    task = session()
    task.observe(**make_demo_frame(0.14, 0.14, 0.0, view_id="left"))
    with pytest.raises(InvalidPrediction, match="Survey"):
        parse_multiframe_decision(
            json.dumps(decision(task)), task.memory.summary(), task.memory.cfg
        )
    value = {
        **decision(task),
        "status": "need_observation",
        "actions": [],
        "observe_region": "right",
    }
    assert (
        parse_multiframe_decision(json.dumps(value), task.memory.summary(), task.memory.cfg)
        == value
    )


def test_qwen_receives_both_images_and_memory_but_mock_is_not_accuracy_evidence():
    task = two_views()
    calls = []

    class FakeQwen:
        def predict_messages(self, images, messages):
            calls.append((images, messages))
            return json.dumps(decision(task)), {"mock_only": True}

    value, details = task.decide(FakeQwen(), "将最左边的柿子放入蓝色托盘")
    assert value["actions"][0]["target_instance_id"] == "obj_001"
    images, messages = calls[0]
    assert len(images) == 2
    content = messages[1]["content"]
    assert sum(part["type"] == "image" for part in content) == 2
    text = " ".join(part["text"] for part in content if part["type"] == "text")
    assert "obj_001" in text and "obj_002" in text and "left_axis_base" in text
    assert details["metrics"]["mock_only"] and not details["robot_control_enabled"]


def test_scene_change_during_model_call_invalidates_the_decision():
    task = two_views()
    original = decision(task)

    class ChangingQwen:
        def predict_messages(self, images, messages):
            task.observe(**make_demo_frame(0.14, 0.14, 2.0, view_id="left"))
            return json.dumps(original), {"mock_only": True}

    with pytest.raises(InvalidPrediction, match="changed during"):
        task.decide(ChangingQwen(), "把最左边的柿子放到蓝色托盘")
    assert task.last_inference["inference_executed"]


def test_unknown_destination_is_rejected_then_model_can_request_observation():
    task = two_views()
    task.memory.regions.clear()
    calls = []

    class RepairingQwen:
        def predict_messages(self, images, messages):
            calls.append(messages)
            if len(calls) == 1:
                return json.dumps(decision(task)), {"mock_only": True}
            assert "available_region_ids" in messages[-1]["content"]
            return json.dumps(
                {
                    **decision(task),
                    "status": "need_observation",
                    "actions": [],
                    "observe_region": "left",
                    "message": "需要观察目标区域",
                }
            ), {"mock_only": True}

    result, details = task.decide(RepairingQwen(), "把柿子放到蓝色区域")
    assert len(calls) == 2 and result["status"] == "need_observation"
    assert len(details["attempts"]) == 2
    assert "Unknown destination" in details["attempts"][0]["validation_error"]
    assert not details["robot_control_enabled"]


def test_unknown_destination_is_never_accepted_after_failed_retry():
    task = two_views()

    class WrongQwen:
        count = 0

        def predict_messages(self, images, messages):
            self.count += 1
            value = decision(task)
            value["actions"][0]["destination_region_id"] = "invented"
            return json.dumps(value), {"mock_only": True}

    model = WrongQwen()
    with pytest.raises(InvalidPrediction, match="Unknown destination"):
        task.decide(model, "把柿子放到指定区域")
    assert model.count == 2 and task.last_decision is None


def test_expired_object_ids_and_frames_do_not_pollute_a_new_survey():
    task = two_views()
    task.observe(**make_demo_frame(0.14, 0.14, 60.0, view_id="left"))
    snapshot = task.memory.summary()
    assert [o["instance_id"] for o in snapshot["objects"]] == ["obj_003"]
    assert [f["frame_id"] for f in snapshot["keyframes"]] == ["frame_000003"]
    assert not snapshot["coverage"]["survey_complete"]


def test_qwen_plan_can_bind_fresh_unchanged_geometry_without_repeat_inference():
    task = two_views()

    class FakeQwen:
        def predict_messages(self, images, messages):
            return json.dumps(decision(task)), {"mock_only": True}

    plan, _ = task.decide(FakeQwen(), "将最左边的柿子放入蓝色托盘")
    task.observe(**make_demo_frame(0.14, 0.14, 2.0, view_id="left"))
    mask, goal = task.selection_for_execution(plan, now_s=2.01)
    assert mask.instance_key == "obj_001" and goal.region_id == "blue_tray"
    assert plan["memory_version"] == 2  # Preserve the model's original output.
    assert task.last_binding_validation == {
        "decision_memory_version": 2,
        "binding_memory_version": 3,
        "bounded_geometry_refresh": True,
    }


@pytest.mark.parametrize("target_y", [0.16, 0.30])
def test_changed_geometry_or_new_identity_requires_new_qwen_decision(target_y):
    task = two_views()

    class FakeQwen:
        def predict_messages(self, images, messages):
            return json.dumps(decision(task)), {"mock_only": True}

    plan, _ = task.decide(FakeQwen(), "将最左边的柿子放入蓝色托盘")
    task.observe(**make_demo_frame(target_y, target_y, 2.0, view_id="left"))
    with pytest.raises(InvalidPrediction, match="changed"):
        task.selection_for_execution(plan, now_s=2.0)


def test_execution_requires_reobservation_and_fresh_qwen_version():
    task = two_views()
    old = decision(task)
    with pytest.raises(ValueError, match="reobserved"):
        task.selection_for_execution(old, now_s=1.0)
    frame = make_demo_frame(0.14, 0.14, 2.0, view_id="left")
    task.observe(**frame)
    with pytest.raises(InvalidPrediction, match="stale memory"):
        task.selection_for_execution(old, now_s=2.0)
    mask, goal = task.selection_for_execution(decision(task), now_s=2.0)
    scene = SceneObserver().build(
        frame["observation"], frame["calibration"], now_s=2.0, detections=[mask], regions=[goal]
    )
    scene.select_target(mask.instance_key, mask.source)
    features = policy_scene_features(scene, goal_region_id=goal.region_id)
    assert features["target_valid_mask"].sum() > 12 and features["goal_valid"]
    with pytest.raises(ValueError, match="stale"):
        task.selection_for_execution(decision(task), now_s=2.5)


def test_sensor_timestamp_and_provenance_checks_are_kept():
    task = two_views()
    with pytest.raises(ValueError, match="monotonically"):
        task.observe(**make_demo_frame(0.14, 0.14, 0.5, view_id="left"))
    bad = make_demo_frame(0.14, 0.14, 3.0, view_id="left")
    bad["detections"][0].timestamp_s = 1.0
    with pytest.raises(ValueError, match="timestamp"):
        task.observe(**bad)
    assert task.memory.version == 2


def test_public_recording_round_trip_and_two_view_request(tmp_path):
    bundle = create_demo_bundle(tmp_path / "bundle")
    manifest = json.loads(bundle.read_text())
    assert manifest["fixture_only"]
    task = session()
    for path in manifest["frame_files"]:
        task.observe(**read_frame(bundle.parent / path))
    request = task.save_request(tmp_path / "prepared", manifest["instruction"])
    assert len(request["annotated_image_files"]) == 2
    assert not request["inference_executed"]
    assert len(request["snapshot"]["objects"]) == 2
