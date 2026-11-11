"""Shared causal BC/runtime features derived only from masks, depth and robot data.

Mask origin is supplied/audited by the caller. This builder does not relabel
renderer masks as sensor predictions or read object poses/teacher plans.
"""

import numpy as np

from piper_sim.sensor_skill_env import (
    PublicEstimate,
    backproject_mask,
    public_actor_frame,
    tcp_from_encoders,
)


class MaskDepthFeatures:
    def __init__(self, calibration):
        self.calibration = calibration
        self.reset()

    def reset(self):
        self.last_stamp = self.start_stamp = None
        self.last_target = None
        self.last_target_stamp = 0.
        self.goal = self.goal_half = self.support_height = None
        self.goal_stamp = None

    def process(self, observation, target_mask, goal_mask, previous_action, previous_valid):
        stamp = float(observation.timestamp_s)
        if self.last_stamp is not None and stamp <= self.last_stamp+1e-9:
            raise ValueError("Only a new camera capture may advance visual features")
        self.last_stamp = stamp
        if self.start_stamp is None:
            self.start_stamp = stamp
        shape = observation.depth_m.shape
        target_mask, goal_mask = np.asarray(target_mask, bool), np.asarray(goal_mask, bool)
        if target_mask.shape != shape or goal_mask.shape != shape:
            raise ValueError("Masks must match the original aligned depth frame")
        # Known horizontal support/gravity assumption; height is measured from
        # visible destination depth, never loaded from the scene/teacher metadata.
        floor_points = backproject_mask(observation, goal_mask & ~target_mask, self.calibration)
        if self.goal is None and len(floor_points) >= 12:
            height = float(np.median(floor_points[:, 2]))
            floor_points = floor_points[np.abs(floor_points[:, 2]-height) < .009]
            if len(floor_points) >= 12:
                self.goal = np.median(floor_points, axis=0)
                self.goal_half = np.maximum(np.minimum(np.ptp(floor_points, axis=0)[:2]/4, .04), .001)
                self.support_height, self.goal_stamp = height, stamp
        points = backproject_mask(observation, target_mask, self.calibration)
        if len(points) and self.support_height is not None:
            points = points[points[:, 2] > self.support_height+.008]
        target = np.median(points, axis=0) if len(points) >= 12 else None
        if target is not None:
            self.last_target, self.last_target_stamp = target.copy(), stamp
        e = PublicEstimate(
            stamp, tcp_from_encoders(observation), target, target_mask, goal_mask, self.goal,
            self.goal_half, None if self.support_height is None else np.array([0., 0., 1.]),
            None if self.support_height is None else -self.support_height,
            observation.gripper_width_m)
        return public_actor_frame(
            observation, e, previous_action, previous_valid, last_target=self.last_target,
            target_age=stamp-self.last_target_stamp,
            goal_age=0. if self.goal_stamp is None else stamp-self.goal_stamp)


def current_robot_features(*, joint_position, joint_velocity, gripper_width, previous_action,
                           previous_valid, end_pose, vision_age, mask_age, joint_age,
                           max_visual_age=.25, max_joint_age=.015):
    valid = vision_age <= max_visual_age and mask_age <= max_visual_age
    state = np.r_[np.asarray(joint_position)/np.pi, joint_velocity, gripper_width/.07,
                  previous_action, float(previous_valid), np.asarray(end_pose)[:3].ravel(),
                  vision_age/max_visual_age, joint_age/max_joint_age, mask_age/max_visual_age,
                  float(valid)].astype(np.float32)
    if state.shape != (37,) or not np.isfinite(state).all():
        raise ValueError("Invalid independent current robot features")
    return state
