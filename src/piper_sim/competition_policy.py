"""Read-only inference for the competition's multirate pick/place candidates."""

from collections import deque
from pathlib import Path

import numpy as np

from piper_sim.calibration import validate_transform
from piper_sim.mask_depth_features import MaskDepthFeatures, current_robot_features
from piper_sim.multirate_control import load_rate_profile
from piper_sim.observation import Observation

_OBSERVATION_SHAPES = {
    "visual_images": (4, 4, 72, 96),
    "visual_state": (4, 37),
    "current_robot": (37,),
    "vision_token": (2,),
}
_EPS_S = 1e-8


def _previous_action(action):
    if action is None:
        return np.zeros(7, np.float32), False
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (7,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
        raise ValueError("Expected the last issued normalized action (seven values in [-1, 1])")
    return action, True


class MultiRatePolicyPreview:
    """Build the training-time inputs and predict without a command transport.

    Use trusted local SB3 archives. Reset when a new target or tracker identity
    is bound. Timestamps must share one clock. A single visual capture can be
    reused at several policy ticks with separate current encoder samples.
    """

    def __init__(self, calibration, *, pick_checkpoint, place_checkpoint, device="cpu"):
        from stable_baselines3 import PPO

        from piper_sim.multirate_skill_policy import MultiRateSkillEncoder

        validate_transform(calibration.T_end_camera)
        self.profile = load_rate_profile()
        self.models = {}
        for skill, checkpoint in (("pick", pick_checkpoint), ("place", place_checkpoint)):
            model = PPO.load(Path(checkpoint), device=device)
            spaces = getattr(model.observation_space, "spaces", {})
            shapes = {key: space.shape for key, space in spaces.items()}
            metadata = getattr(model, "bc_training_metadata", None)
            expected_metadata = {
                "skill": skill,
                "policy_hz": self.profile.policy_hz,
                "vision_hz": self.profile.depth_hz,
                "joint_execution_hz": self.profile.joint_command_hz,
            }
            if (
                shapes != _OBSERVATION_SHAPES
                or model.action_space.shape != (7,)
                or not np.all(model.action_space.low == -1)
                or not np.all(model.action_space.high == 1)
                or not isinstance(model.policy.features_extractor, MultiRateSkillEncoder)
                or (
                    metadata is not None
                    and any(metadata.get(key) != value for key, value in expected_metadata.items())
                )
            ):
                raise ValueError(f"{skill} checkpoint has an incompatible multirate contract")
            self.models[skill] = model
        self.features = MaskDepthFeatures(calibration)
        self.history = deque(maxlen=4)
        self.episode = 0
        self.reset()

    def reset(self):
        self.episode += 1
        self.frame_index = 0
        self.frame_accepted = False
        self.features.reset()
        self.history.clear()

    def push_frame(
        self, observation: Observation, target_mask, goal_mask, *,
        mask_timestamp_s, previous_action=None,
    ):
        """Accept two masks aligned to one wrist-depth frame at capture time."""
        self.frame_accepted = False
        timestamp = observation.timestamp_s
        if (
            not np.isfinite([timestamp, mask_timestamp_s]).all()
            or abs(timestamp - mask_timestamp_s) > _EPS_S
        ):
            raise ValueError("Depth and masks must carry the same source timestamp")
        if observation.T_base_camera is not None:
            raise ValueError("The multirate feature contract uses the wrist-camera transform")
        depth = observation.depth_m
        if (
            depth is None
            or depth.ndim != 2
            or depth.dtype.kind != "f"
            or not observation.depth_valid_mask.any()
        ):
            raise ValueError("A valid aligned metric depth frame is required")
        intrinsics = observation.intrinsics or {}
        k = np.array([intrinsics.get(key, np.nan) for key in ("fx", "fy", "cx", "cy")])
        if not np.isfinite(k).all() or (k[:2] <= 0).any():
            raise ValueError("Valid rectified intrinsics are required")
        validate_transform(observation.T_base_end)
        target_mask, goal_mask = np.asarray(target_mask), np.asarray(goal_mask)
        if any(mask.dtype != bool or mask.shape != depth.shape for mask in (target_mask, goal_mask)):
            raise ValueError("Both masks must be boolean arrays aligned to depth")
        previous, valid = _previous_action(previous_action)
        image, state = self.features.process(observation, target_mask, goal_mask, previous, valid)
        self.history.append((image, np.r_[state, 1.0].astype(np.float32)))
        self.frame_index += 1
        self.frame_accepted = True

    def predict(self, skill, robot: Observation, *, now_s, previous_action=None):
        """Return six normalized joint deltas and one normalized gripper width target."""
        if skill not in self.models:
            raise ValueError("Select the pick or place checkpoint")
        if not self.frame_accepted:
            raise ValueError("A new synchronized visual frame is required")
        ages = np.array([now_s - self.features.last_stamp, now_s - robot.timestamp_s])
        if not np.isfinite(ages).all() or np.any(ages < -_EPS_S):
            raise ValueError("Invalid or future-dated policy input")
        vision_age, joint_age = np.maximum(ages, 0.0)
        profile = self.profile
        if vision_age > profile.max_visual_age_s or joint_age > profile.max_joint_age_s:
            raise ValueError("Stale visual or joint feedback")
        if (
            np.shape(robot.joint_position_rad) != (6,)
            or np.shape(robot.joint_velocity_rad_s) != (6,)
        ):
            raise ValueError("Six positions and six velocities are required")
        previous, valid = _previous_action(previous_action)
        current = current_robot_features(
            joint_position=robot.joint_position_rad,
            joint_velocity=robot.joint_velocity_rad_s,
            gripper_width=robot.gripper_width_m,
            previous_action=previous,
            previous_valid=valid,
            end_pose=validate_transform(robot.T_base_end),
            vision_age=vision_age,
            mask_age=vision_age,
            joint_age=joint_age,
            max_visual_age=profile.max_visual_age_s,
            max_joint_age=profile.max_joint_age_s,
        )
        pad = (np.zeros((4, 72, 96), np.float32), np.zeros(37, np.float32))
        frames = [pad] * (4 - len(self.history)) + list(self.history)
        observation = {
            "visual_images": np.stack([frame[0] for frame in frames]),
            "visual_state": np.stack([frame[1] for frame in frames]),
            "current_robot": current,
            "vision_token": np.array([self.episode, self.frame_index], np.float32),
        }
        action, _ = self.models[skill].predict(observation, deterministic=True)
        return _previous_action(action)[0]
