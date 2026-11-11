"""Calibration gates, causal live features and independent skill outcome evidence.

No IK, ROS, simulator truth or autonomous command publishing. Callers own actual
sensor association, motion limits, a driver watchdog and measured stop handling.
"""

import copy
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from piper_sim.bc_data import actor_frame
from piper_sim.calibration import validate_transform
from piper_sim.grasp_target_memory import GraspTargetMemory
from piper_sim.scene_observation import Provenance, policy_scene_features
from piper_sim.task_handoff import ExecutionFeedback, finite_time


def check_calibration(path, interface):
    """Require actual parameters and a referenced accepted measurement report.

    This checks the recorded contract, not the truth of a human acceptance claim.
    An identity/default matrix is never inserted in place of missing calibration.
    """
    path = Path(path)
    if not path.is_file():
        return {"ready": False, "blockers": ["calibration_file_missing"]}
    data = json.loads(path.read_text())
    blockers = []
    if data.get("schema_version") != 1 or data.get("status") != "validated":
        blockers.append("calibration_not_validated")
    for key in ("calibration_id", "camera_serial", "robot_serial"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            blockers.append(key + "_missing")
    if data.get("frames") != interface["frames"]:
        blockers.append("coordinate_frames_mismatch")
    if data.get("joint_names") != interface["joints"]["names"]:
        blockers.append("joint_order_not_confirmed")
    if data.get("gripper_width_unit") != "metres_full_opening":
        blockers.append("gripper_unit_not_confirmed")
    if data.get("depth_aligned_to_rgb") is not True:
        blockers.append("depth_rgb_alignment_not_confirmed")
    try:
        size = np.asarray(data["image_size_wh"], float)
        k = np.asarray([data["intrinsics"][n] for n in ("fx", "fy", "cx", "cy")], float)
        scale = float(data["depth_scale_m_per_unit"])
        if (size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any()
                or not np.equal(size, size.astype(int)).all() or not np.isfinite(k).all()
                or (k[:2] <= 0).any() or not np.isfinite(scale) or scale <= 0):
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        blockers.append("camera_intrinsics_or_depth_scale_missing")
    mount = data.get("camera_mount")
    transform_key = {"eye_in_hand": "T_end_camera", "eye_to_hand": "T_base_camera"}.get(mount)
    try:
        if transform_key is None:
            raise ValueError()
        validate_transform(data[transform_key])
    except (KeyError, TypeError, ValueError):
        blockers.append("camera_mount_or_extrinsics_missing")
    validation = data.get("validation") or {}
    try:
        skew = finite_time(validation.get("measured_max_sensor_skew_s"))
        if skew < 0 or skew > interface["observation"]["max_sensor_skew_s"]:
            raise ValueError()
    except ValueError:
        blockers.append("sensor_synchronization_not_validated")
    if validation.get("accepted") is not True:
        blockers.append("calibration_measurement_not_accepted")
    try:
        report = path.parent / validation["report_file"]
        if (not report.is_file()
                or hashlib.sha256(report.read_bytes()).hexdigest() != validation["report_sha256"]):
            raise ValueError()
    except (KeyError, TypeError, ValueError, OSError):
        blockers.append("calibration_measurement_report_missing_or_changed")
    return {"ready": not blockers, "blockers": blockers,
            "calibration_id": data.get("calibration_id"),
            "calibration_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def calibration_for_pose(bundle, T_base_end):
    """Convert a validated fixed-camera estimate to the observer's per-frame interface.

    The robot end pose must come from the synchronized driver state. No inverse
    kinematics is needed. This is a transform composition, not calibration itself.
    """
    if bundle.get("status") != "validated":
        raise ValueError("Measured calibration required")
    if bundle["camera_mount"] == "eye_in_hand":
        return validate_transform(bundle["T_end_camera"]).copy()
    if bundle["camera_mount"] == "eye_to_hand":
        return np.linalg.inv(validate_transform(T_base_end)) @ validate_transform(bundle["T_base_camera"])
    raise ValueError("Unknown camera mount")


def public_skill_observation(scene, observation, *, destination_region_id, now_s,
                             previous_action, previous_action_valid):
    """Adapt a fresh public SceneEstimate + synchronized joints, never an old packet."""
    now_s = finite_time(now_s)
    if (abs(scene.timestamp_s - observation.timestamp_s) > 1e-6
            or not np.allclose(scene.T_base_end, observation.T_base_end, atol=1e-6)):
        raise ValueError("Scene and robot state must belong to the same sensor frame")
    features = policy_scene_features(scene, goal_region_id=destination_region_id)
    return {**{k: v for k, v in features.items() if k not in ("point_frame", "T_base_goal", "target_visible")},
            "joint_position_rad": observation.joint_position_rad.copy(),
            "joint_velocity_rad_s": observation.joint_velocity_rad_s.copy(),
            "gripper_width_m": observation.gripper_width_m,
            "T_base_end": scene.T_base_end.copy(), "decision_time_s": now_s,
            "previous_action": np.asarray(previous_action, np.float32).copy(),
            "previous_action_valid": bool(previous_action_valid)}


class OnlineSkillHistory:
    def __init__(self, memory_config, *, history_steps=4):
        if type(history_steps) is not int or history_steps < 1:
            raise ValueError("Positive history size required")
        self.memory = GraspTargetMemory(**memory_config)
        self.history = deque(maxlen=history_steps)
        self.identity = None
        self.previous_closing = False
        self.last_sensor_s = None

    def reset(self):
        self.memory.reset()
        self.history.clear()
        self.identity = None
        self.previous_closing = False
        self.last_sensor_s = None

    def push(self, observation, *, target_id, destination_id, calibration_id,
             previous_command_target, previous_command_valid):
        """Only ACKed, previously issued action/command may be supplied by the adapter.

        Call continuously across pick->place. Reset on target/destination or
        calibration change; no history may cross an independent manipulation.
        """
        if not all(isinstance(v, str) and v for v in (target_id, destination_id, calibration_id)):
            raise ValueError("Target, destination and measured calibration IDs required")
        identity = (target_id, destination_id, calibration_id)
        if self.identity is not None and identity != self.identity:
            self.reset()
        self.identity = identity
        stamp, now = finite_time(observation["timestamp_s"]), finite_time(observation["decision_time_s"])
        if now < stamp or now - stamp > self.memory.max_sensor_age_s:
            raise ValueError("Fresh synchronized observation required")
        if self.last_sensor_s is not None and stamp <= self.last_sensor_s:
            raise ValueError("Repeated/out-of-order frame cannot advance policy history")
        if (np.shape(observation["joint_position_rad"]) != (6,)
                or np.shape(observation["joint_velocity_rad_s"]) != (6,)
                or np.shape(previous_command_target) != (7,)):
            raise ValueError("Expected six arm joints and seven-dimensional command")
        closing = bool(observation["previous_action_valid"] and observation["previous_action"][6] < -0.5)
        if closing != self.previous_closing:
            self.memory.reset()
        self.previous_closing = closing
        result = self.memory.update(observation["target_points"], observation["target_valid_mask"],
            observation["T_base_end"], target_id=target_id, timestamp_s=stamp, now_s=now,
            allow_static_memory=not closing)
        result = self.memory.express(result, observation["T_base_end"])
        frame = actor_frame(observation, result, previous_command_target, previous_command_valid)
        self.history.append(frame)
        self.last_sensor_s = stamp
        padding = self.history.maxlen - len(self.history)
        items = [self.history[0]] * padding + list(self.history)
        return ({k: np.stack([row[k] for row in items]) for k in frame},
                np.array([False] * padding + [True] * len(self.history)),
                {"target_geometry_status": result["status"], "target_usable": result["usable"]})


class SkillOutcomeMonitor:
    """Measured stage outcomes, separate from the action network's predictions.

    Complete footprint/support/holding claims must be estimated by a real sensor
    verifier. Visible points alone cannot prove a full footprint is inside.
    """

    def __init__(self, packet, settings):
        self.metadata = copy.deepcopy(packet.metadata)
        self.settings = copy.deepcopy(settings)
        self.phase = "picking"
        self.last_sensor_s = None
        self.since_s = None
        self.started_s = finite_time(self.metadata["prepared_at_s"])
        self.sequence = 0

    def _feedback(self, status, stamp, source, evidence):
        self.sequence += 1
        meta = self.metadata
        return ExecutionFeedback(
            f"{meta['handoff_id']}:{self.sequence}:{status}", meta["task_id"], meta["step_id"],
            meta["handoff_id"], status, stamp, source, copy.deepcopy(evidence))

    def update(self, evidence, *, source, now_s):
        now_s = finite_time(now_s)
        if self.phase == "completed":
            return {"phase": self.phase, "feedback": None, "stop_required": True}
        source.require_public()
        if source.kind != "sensor_estimate":
            raise ValueError("Measured sensor evidence required")
        stamp = finite_time(evidence.get("observation_timestamp_s"))
        meta, cfg = self.metadata, self.settings
        if (evidence.get("target_instance_id") != meta["target_instance_id"]
                or evidence.get("destination_region_id") != meta["destination_region_id"]):
            raise ValueError("Evidence target/destination mismatch")
        if (stamp > now_s or stamp < self.started_s
                or now_s - stamp > cfg["max_observation_age_s"]):
            self.since_s = None
            return {"phase": self.phase, "feedback": None, "stop_required": True,
                    "code": "fresh_verification_required"}
        if now_s - self.started_s > cfg["max_skill_duration_s"]:
            self.since_s = None
            return {"phase": self.phase, "feedback": None, "stop_required": True,
                    "code": "skill_timeout"}
        if self.last_sensor_s is not None and stamp <= self.last_sensor_s:
            raise ValueError("Only a new measured frame can advance verification")
        if self.last_sensor_s is not None and stamp - self.last_sensor_s > cfg["max_evidence_gap_s"]:
            self.since_s = None
        self.last_sensor_s = stamp
        event = None
        if self.phase == "picking":
            clearance = evidence.get("support_clearance_m")
            condition = (evidence.get("holding_confirmed") is True
                and evidence.get("target_identity_confirmed") is True
                and type(clearance) in (int, float) and np.isfinite(clearance)
                and clearance >= cfg["grasp_clearance_m"])
            self.since_s = stamp if condition and self.since_s is None else self.since_s if condition else None
            if self.since_s is not None and stamp - self.since_s + 1e-9 >= cfg["grasp_hold_s"]:
                event = self._feedback("picked", stamp, source, {**evidence, "grasp_confirmed": True})
                self.phase, self.since_s = "placing", None
        elif self.phase == "placing":
            if evidence.get("release_confirmed") is True and evidence.get("holding_confirmed") is False:
                event = self._feedback("placed", stamp, source, evidence)
                self.phase, self.since_s = "verifying", None
        elif self.phase == "verifying":
            condition = (evidence.get("release_confirmed") is True
                and evidence.get("target_identity_confirmed") is True
                and evidence.get("fully_inside_region") is True
                and evidence.get("footprint_complete") is True
                and evidence.get("stable") is True
                and evidence.get("holding_confirmed") is False)
            self.since_s = stamp if condition and self.since_s is None else self.since_s if condition else None
            if self.since_s is not None and stamp - self.since_s + 1e-9 >= cfg["placement_stable_s"]:
                event = self._feedback("verified", stamp, source, {**evidence, "placement_verified": True})
                self.phase = "completed"
        return {"phase": self.phase, "feedback": event, "stop_required": self.phase == "completed"}
