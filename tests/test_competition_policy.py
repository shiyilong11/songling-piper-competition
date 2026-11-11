from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from piper_sim.observation import Observation


@pytest.fixture
def preview(monkeypatch):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from piper_sim.competition_policy import MultiRatePolicyPreview
    from piper_sim.multirate_skill_policy import MultiRateSkillEncoder

    shapes = {
        "visual_images": (4, 4, 72, 96),
        "visual_state": (4, 37),
        "current_robot": (37,),
        "vision_token": (2,),
    }

    def load(path, *, device):
        return SimpleNamespace(
            observation_space=SimpleNamespace(
                spaces={key: SimpleNamespace(shape=shape) for key, shape in shapes.items()}
            ),
            action_space=SimpleNamespace(shape=(7,), low=-np.ones(7), high=np.ones(7)),
            policy=SimpleNamespace(features_extractor=object.__new__(MultiRateSkillEncoder)),
            bc_training_metadata={
                "skill": path.stem, "policy_hz": 20, "vision_hz": 10, "joint_execution_hz": 200,
            },
            predict=Mock(return_value=(np.zeros(7, np.float32), None)),
        )

    monkeypatch.setattr(PPO, "load", load)
    return MultiRatePolicyPreview(
        SimpleNamespace(T_end_camera=np.eye(4)),
        pick_checkpoint="pick.zip",
        place_checkpoint="place.zip",
    )


def camera_frame():
    depth = np.full((20, 24), 0.4, np.float32)
    target = np.zeros_like(depth, dtype=bool)
    target[:5, :8] = True
    depth[target] = 0.45
    return Observation(
        1.0, np.zeros(6), np.zeros(6), 0.07, np.eye(4),
        depth_m=depth, intrinsics={"fx": 24, "fy": 24, "cx": 12, "cy": 10},
    ), target, ~target


def test_policy_reuses_capture_with_fresh_feedback_without_advancing_visual_history(preview):
    capture, target, goal = camera_frame()
    preview.push_frame(capture, target, goal, mask_timestamp_s=1.0)
    pick = preview.models["pick"].predict
    preview.predict("pick", capture, now_s=1.0)
    first = pick.call_args.args[0]
    newer = replace(capture, timestamp_s=1.05, joint_position_rad=np.ones(6))
    preview.predict("pick", newer, now_s=1.05, previous_action=np.full(7, 0.1))
    second = pick.call_args.args[0]

    for key in ("visual_images", "visual_state", "vision_token"):
        np.testing.assert_array_equal(first[key], second[key])
    assert not np.allclose(first["current_robot"], second["current_robot"])
    assert second["visual_state"][:, -1].tolist() == [0.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="Stale"):
        preview.predict("place", capture, now_s=1.05)
    with pytest.raises(ValueError, match="Stale"):
        preview.predict("place", replace(newer, timestamp_s=1.30), now_s=1.30)
    with pytest.raises(ValueError, match="future-dated"):
        preview.predict("place", newer, now_s=1.04)
    preview.models["place"].predict.assert_not_called()


def test_bad_mask_sync_and_reset_require_new_frame(preview):
    capture, target, goal = camera_frame()
    preview.push_frame(capture, target, goal, mask_timestamp_s=1.0)
    with pytest.raises(ValueError, match="same source timestamp"):
        preview.push_frame(capture, target, goal, mask_timestamp_s=0.9)
    with pytest.raises(ValueError, match="new synchronized"):
        preview.predict("pick", capture, now_s=1.0)
    with pytest.raises(ValueError, match="new camera capture"):
        preview.push_frame(capture, target, goal, mask_timestamp_s=1.0)
    preview.reset()
    with pytest.raises(ValueError, match="new synchronized"):
        preview.predict("pick", capture, now_s=1.0)
    preview.push_frame(capture, target, goal, mask_timestamp_s=1.0)
    assert preview.predict("place", capture, now_s=1.0).shape == (7,)
