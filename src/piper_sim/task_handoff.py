"""Versioned brain-to-policy handoff and feedback ledger. No motion or model calls."""

import copy
import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from piper_sim.multiframe_task import parse_multiframe_decision
from piper_sim.qwen_adapter import InvalidPrediction
from piper_sim.scene_observation import Provenance, policy_scene_features

HANDOFF_ARRAY_FIELDS = {
    "target_points",
    "target_valid_mask",
    "scene_points",
    "scene_valid_mask",
    "target_visible",
    "table_plane_base",
    "table_valid",
    "table_confidence",
    "T_base_goal",
    "T_end_goal",
    "goal_half_size_xy_m",
    "goal_valid",
    "target_mask",
    "aligned_depth_m",
    "depth_valid_mask",
    "target_points_base",
    "visible_scene_points_base",
    "T_base_end",
    "T_end_camera_estimated",
    "T_base_camera_estimated",
    "joint_position_rad",
    "joint_velocity_rad_s",
    "gripper_width_m",
}


def finite_time(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError("Expected a finite timestamp in the shared sensor clock")
    return float(value)


@dataclass
class HandoffPacket:
    metadata: dict
    arrays: dict

    def export(self, folder):
        """Portable JSON + no-pickle NPZ for an execution adapter; never a command."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / "observation.npz"
        np.savez_compressed(path, **self.arrays)
        manifest = copy.deepcopy(self.metadata)
        manifest["array_file"] = path.name
        manifest["array_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["arrays"] = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in self.arrays.items()
        }
        (folder / "handoff.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
        )
        return folder / "handoff.json"


def load_handoff(path, *, now_s):
    """Validate a portable packet before an adapter inspects it; never execute it."""
    now_s = finite_time(now_s)
    path = Path(path).resolve()
    metadata = json.loads(path.read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported handoff schema")
    if (
        metadata.get("robot_control_enabled") is not False
        or metadata.get("execution_dispatch_enabled") is not False
    ):
        raise ValueError("Handoff preparation cannot enable robot execution")
    if not finite_time(metadata["prepared_at_s"]) <= now_s <= finite_time(metadata["expires_at_s"]):
        raise ValueError("Handoff packet is stale or from the future")
    for key in ("target_source", "destination_source", "calibration_source"):
        Provenance(**metadata[key]).require_public()
    if "perception_binding" in metadata:
        binding = metadata["perception_binding"]
        if (not isinstance(binding, dict)
                or set(binding) != {"stable_target_id", "tracker_epoch", "track_id", "sensor_frame_id",
                                    "tracker_frame_index", "mask_timestamp_s", "depth_timestamp_s",
                                    "identity_status", "tracking_status"}
                or binding["stable_target_id"] != metadata["target_instance_id"]
                or binding["identity_status"] != "estimated" or binding["tracking_status"] != "tracking"
                or not isinstance(binding["tracker_epoch"], str) or not binding["tracker_epoch"]
                or not isinstance(binding["sensor_frame_id"], str) or not binding["sensor_frame_id"]
                or type(binding["track_id"]) is not int or binding["track_id"] < 1
                or type(binding["tracker_frame_index"]) is not int or binding["tracker_frame_index"] < 0
                or binding["mask_timestamp_s"] != metadata["observation_timestamp_s"]
                or binding["depth_timestamp_s"] != metadata["observation_timestamp_s"]):
            raise ValueError("Invalid tracker identity or mask/depth frame binding")
    array_file = (path.parent / metadata["array_file"]).resolve()
    if not array_file.is_relative_to(path.parent):
        raise ValueError("Handoff arrays must be inside the packet folder")
    if hashlib.sha256(array_file.read_bytes()).hexdigest() != metadata["array_sha256"]:
        raise ValueError("Handoff array checksum mismatch")
    with np.load(array_file, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    if set(arrays) != set(metadata["arrays"]) or set(arrays) != HANDOFF_ARRAY_FIELDS:
        raise ValueError("Unexpected handoff array fields")
    for key, value in arrays.items():
        if (
            list(value.shape) != metadata["arrays"][key]["shape"]
            or str(value.dtype) != metadata["arrays"][key]["dtype"]
        ):
            raise ValueError("Handoff array schema mismatch")
    if (
        arrays["target_mask"].shape != arrays["aligned_depth_m"].shape
        or arrays["depth_valid_mask"].shape != arrays["aligned_depth_m"].shape
        or arrays["target_mask"].dtype != bool
        or arrays["depth_valid_mask"].dtype != bool
    ):
        raise ValueError("Handoff depth/mask alignment mismatch")
    return HandoffPacket(metadata, arrays)


@dataclass(frozen=True)
class ExecutionFeedback:
    event_id: str
    task_id: str
    step_id: str
    handoff_id: str
    status: str
    timestamp_s: float
    source: Provenance
    evidence: dict = field(default_factory=dict)
    error_code: str | None = None

    def as_dict(self):
        return {
            "schema_version": 1,
            "event_id": self.event_id,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "handoff_id": self.handoff_id,
            "status": self.status,
            "timestamp_s": self.timestamp_s,
            "source": self.source.as_dict(),
            "evidence": copy.deepcopy(self.evidence),
            "error_code": self.error_code,
        }

    @classmethod
    def from_dict(cls, data):
        required = {
            "schema_version",
            "event_id",
            "task_id",
            "step_id",
            "handoff_id",
            "status",
            "timestamp_s",
            "source",
            "evidence",
            "error_code",
        }
        if not isinstance(data, dict) or set(data) != required or data["schema_version"] != 1:
            raise ValueError("Invalid execution feedback schema")
        values = {key: value for key, value in data.items() if key != "schema_version"}
        values["source"] = Provenance(**values["source"])
        return cls(**values)


@dataclass
class Preparation:
    status: str
    packet: HandoffPacket | None = None
    code: str | None = None
    observe_region: str | None = None
    message: str = ""


class TaskHandoffSession:
    """Serial step dispatch preparation; success requires independent verification feedback.

    The caller owns a common clock and serialises brain.observe/decide with this
    ledger. This adapter does not import the legacy simulator skill coordinator.
    """

    def __init__(self, brain, *, feedback_max_age_s=2.0, max_plan_revisions=8):
        if not np.isfinite(feedback_max_age_s) or feedback_max_age_s <= 0:
            raise ValueError("Invalid feedback age threshold")
        self.brain = brain
        self.feedback_max_age_s = float(feedback_max_age_s)
        if type(max_plan_revisions) is not int or max_plan_revisions < 1:
            raise ValueError("Invalid plan revision budget")
        self.max_plan_revisions = max_plan_revisions
        self.lock = threading.RLock()
        self.task_id = None
        self.instruction = None
        self.revision = 0
        self.phase = "empty"
        self.steps = []
        self.decision = None
        self.packet = None
        self.events = {}
        self.last_feedback_time = None
        self.replan_after_version = -1

    def _accepted_decision(self, decision, now_s):
        if decision != self.brain.last_decision or self.brain.last_decision_snapshot is None:
            raise ValueError("Only the brain session's accepted final decision may be handed off")
        snapshot = self.brain.last_decision_snapshot
        value = parse_multiframe_decision(json.dumps(decision), snapshot, self.brain.memory.cfg)
        if value["status"] != "ready":
            raise ValueError("A non-ready understanding cannot produce manipulation steps")
        if snapshot["memory_version"] != self.brain.memory.version:
            raise ValueError("Refresh the plan before accepting a different memory version")
        if now_s < snapshot["timestamp_s"]:
            raise ValueError("Task admission timestamp precedes its observation context")
        return copy.deepcopy(value)

    def accept_plan(self, decision, *, now_s):
        now_s = finite_time(now_s)
        with self.lock:
            if self.phase != "empty":
                raise ValueError("Create a new handoff session for a new instruction")
            value = self._accepted_decision(decision, now_s)
            self.task_id = uuid.uuid4().hex
            self.instruction = self.brain.last_request["instruction"]
            self._append_revision(value)
            return self.state()

    def _append_revision(self, decision):
        self.revision += 1
        self.decision = decision
        self.packet = None
        for i, action in enumerate(decision["actions"]):
            self.steps.append(
                {
                    "step_id": uuid.uuid4().hex,
                    "revision": self.revision,
                    "action_index": i,
                    "skill": "pick_place",
                    **action,
                    "status": "pending",
                    "binding_generation": 0,
                }
            )
        self.phase = "pending"

    def _current_step(self):
        return next(
            (
                step
                for step in self.steps
                if step["revision"] == self.revision
                and step["status"] not in ("succeeded", "failed", "cancelled", "superseded")
            ),
            None,
        )

    def prepare_current(self, *, now_s):
        now_s = finite_time(now_s)
        with self.lock:
            if self.phase == "needs_replan":
                return Preparation(
                    "needs_replan",
                    code="fresh_plan_required",
                    message="上一步已改变场景，请重新观察并理解剩余任务。",
                )
            if self.phase not in ("pending", "waiting_observation", "prepared"):
                raise ValueError("No step available for preparation")
            step = self._current_step()
            if (
                self.decision != self.brain.last_decision
                or self.brain.last_request["instruction"] != self.instruction
            ):
                self.phase = "needs_replan"
                return Preparation(
                    "needs_replan",
                    code="understanding_superseded",
                    message="大脑已有新的理解结果，旧任务需要重新确认。",
                )
            memory = self.brain.memory
            snapshot = memory.summary(now_s)
            if (
                self.packet is not None
                and self.phase == "prepared"
                and self.packet.metadata["binding_memory_version"] == memory.version
                and now_s <= self.packet.metadata["expires_at_s"]
            ):
                return Preparation("prepared", packet=copy.deepcopy(self.packet))
            self.packet = None
            objects = {row["instance_id"]: row for row in snapshot["objects"]}
            obj = objects.get(step["target_instance_id"])
            if obj is None or obj["identity_status"] != "estimated":
                self.phase = "needs_replan"
                return Preparation(
                    "needs_replan", code="identity_unresolved", message="目标身份缺失或有歧义。"
                )
            max_age = memory.cfg["observation_max_age_s"]
            if not obj["currently_visible"] or obj["age_s"] > max_age:
                self.phase, step["status"] = "waiting_observation", "waiting_observation"
                return Preparation(
                    "need_observation",
                    code="target_not_fresh",
                    observe_region="object:" + step["target_instance_id"],
                    message="执行前需要重新看到并确认目标。",
                )
            goal = memory.regions.get(step["destination_region_id"])
            if goal is None:
                self.phase = "needs_replan"
                return Preparation(
                    "needs_replan", code="destination_missing", message="目的区域已失效。"
                )
            if goal.timestamp_s is not None and now_s - goal.timestamp_s > max_age:
                self.phase, step["status"] = "waiting_observation", "waiting_observation"
                return Preparation(
                    "need_observation",
                    code="destination_not_fresh",
                    observe_region="region:" + goal.region_id,
                    message="执行前需要重新确认目的区域。",
                )
            try:
                mask, goal = self.brain.selection_for_execution(
                    self.decision, now_s=now_s, action_index=step["action_index"]
                )
            except (InvalidPrediction, ValueError) as exc:
                self.phase = "needs_replan"
                return Preparation("needs_replan", code="scene_changed", message=str(exc))
            frame = memory.latest_frame
            if frame.depth_m is None or frame.depth_m.shape != mask.mask.shape:
                raise ValueError("Handoff depth must be aligned to its target mask")
            scene = copy.deepcopy(frame.scene)
            for entity in scene.objects:
                entity.instance_key = frame.instances.get(entity.instance_key, entity.instance_key)
            scene.regions = [r for r in scene.regions if r.region_id != goal.region_id] + [goal]
            scene.select_target(mask.instance_key, mask.source)
            features = policy_scene_features(scene, goal_region_id=goal.region_id)
            q, dq = np.asarray(frame.joint_position_rad), np.asarray(frame.joint_velocity_rad_s)
            if (
                q.shape != (6,)
                or dq.shape != (6,)
                or not np.isfinite(q).all()
                or not np.isfinite(dq).all()
                or not np.isfinite(frame.gripper_width_m)
            ):
                raise ValueError("Policy handoff requires finite six-joint sensor state")
            arrays = {
                key: np.asarray(value).copy()
                for key, value in features.items()
                if key not in ("timestamp_s", "point_frame")
            }
            arrays.update(
                target_mask=mask.mask.copy(),
                aligned_depth_m=frame.depth_m.copy(),
                depth_valid_mask=np.isfinite(frame.depth_m) & (frame.depth_m > 0),
                target_points_base=scene.target.points_base.copy(),
                visible_scene_points_base=scene.points_base.copy(),
                T_base_end=scene.T_base_end.copy(),
                T_end_camera_estimated=scene.T_end_camera_estimated.copy(),
                T_base_camera_estimated=(scene.T_base_end @ scene.T_end_camera_estimated),
                joint_position_rad=q.copy(),
                joint_velocity_rad_s=dq.copy(),
                gripper_width_m=np.asarray(frame.gripper_width_m),
            )
            if not features["target_visible"] or not features["goal_valid"]:
                raise ValueError("Invalid target/goal policy binding")
            expires = min(
                frame.timestamp_s + max_age,
                float("inf") if goal.timestamp_s is None else goal.timestamp_s + max_age,
            )
            step["binding_generation"] += 1
            metadata = {
                "schema_version": 1,
                "task_id": self.task_id,
                "step_id": step["step_id"],
                "handoff_id": uuid.uuid4().hex,
                "plan_revision": self.revision,
                "action_index": step["action_index"],
                "revision_action_count": len(self.decision["actions"]),
                "skill": "pick_place",
                "instruction": self.instruction,
                "accepted_decision_sha256": hashlib.sha256(
                    json.dumps(self.decision, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest(),
                "target_instance_id": mask.instance_key,
                "destination_region_id": goal.region_id,
                "decision_memory_version": self.decision["memory_version"],
                "binding_memory_version": memory.version,
                "frame_id": frame.frame_id,
                "observation_timestamp_s": frame.timestamp_s,
                "prepared_at_s": now_s,
                "expires_at_s": expires,
                "binding_generation": step["binding_generation"],
                "base_frame": scene.frame,
                "point_frame": features["point_frame"],
                "intrinsics": {
                    key: float(scene.intrinsics[key]) for key in ("fx", "fy", "cx", "cy")
                },
                "depth_frame": "estimated_optical_frame",
                "depth_encoding": "aligned optical-axis Z in metres; invalid pixels marked by depth_valid_mask",
                "units": {
                    "time": "seconds in shared sensor clock",
                    "length": "metres",
                    "joint_position": "radians",
                    "joint_velocity": "radians_per_second",
                },
                "target_source": mask.source.as_dict(),
                "target_confidence": None
                if scene.target.confidence is None
                else float(scene.target.confidence),
                "target_confidence_is_calibrated": False,
                "destination_source": goal.source.as_dict(),
                "calibration_source": scene.calibration_source.as_dict(),
                "spatial_reference": copy.deepcopy(snapshot["spatial_reference"]),
                "grasp_pose_status": "pending_grasp_planner",
                "rgb_included": False,
                "geometry_complete": False,
                "unobserved_space": "unknown",
                "collision_free_certified": False,
                "robot_control_enabled": False,
                "execution_dispatch_enabled": False,
            }
            self.packet = HandoffPacket(metadata, arrays)
            self.phase, step["status"] = "prepared", "prepared"
            return Preparation("prepared", packet=copy.deepcopy(self.packet))

    def apply_feedback(self, feedback, *, now_s):
        now_s = finite_time(now_s)
        if isinstance(feedback, dict):
            feedback = ExecutionFeedback.from_dict(feedback)
        stamp = finite_time(feedback.timestamp_s)
        if not isinstance(feedback.status, str) or feedback.status not in (
            "started",
            "picked",
            "placed",
            "verified",
            "failed",
            "cancelled",
        ):
            raise ValueError("Invalid execution feedback status")
        if not all(
            isinstance(value, str) and 0 < len(value) <= 128
            for value in (
                feedback.event_id,
                feedback.task_id,
                feedback.step_id,
                feedback.handoff_id,
            )
        ):
            raise ValueError("Feedback identifiers must be nonempty strings")
        feedback.source.require_public()
        encoded = json.dumps(feedback.as_dict(), sort_keys=True, allow_nan=False)
        with self.lock:
            if feedback.event_id in self.events:
                if self.events[feedback.event_id] != encoded:
                    raise ValueError("Feedback event ID reused with different content")
                return {"duplicate": True, **self.state()}
            if stamp > now_s or now_s - stamp > self.feedback_max_age_s:
                raise ValueError("Feedback timestamp is stale or from the future")
            if self.last_feedback_time is not None and stamp < self.last_feedback_time:
                raise ValueError("Out-of-order execution feedback")
            step = self._current_step()
            if (
                step is None
                or self.packet is None
                or feedback.task_id != self.task_id
                or feedback.step_id != step["step_id"]
                or feedback.handoff_id != self.packet.metadata["handoff_id"]
            ):
                raise ValueError("Feedback does not match the current handoff")
            if stamp < self.packet.metadata["prepared_at_s"]:
                raise ValueError("Feedback precedes the prepared handoff")
            status = feedback.status
            allowed = {
                "started": "prepared",
                "picked": "executing",
                "placed": "picked",
                "verified": "awaiting_verification",
            }
            if status in allowed:
                if step["status"] != allowed[status]:
                    raise ValueError("Invalid execution feedback transition")
            elif status not in ("failed", "cancelled") or step["status"] not in (
                "prepared",
                "executing",
                "picked",
                "awaiting_verification",
            ):
                raise ValueError("Invalid execution feedback status")
            if status == "started" and (
                now_s > self.packet.metadata["expires_at_s"]
                or self.brain.memory.version != self.packet.metadata["binding_memory_version"]
                or self.brain.last_decision != self.decision
                or self.brain.last_request["instruction"] != self.instruction
            ):
                raise ValueError("Prepared handoff expired or changed; prepare fresh geometry")
            if status in ("picked", "placed", "verified"):
                if (
                    feedback.source.kind != "sensor_estimate"
                    or not isinstance(feedback.evidence, dict)
                    or not feedback.evidence
                ):
                    raise ValueError(
                        "Measured execution progress requires documented sensor evidence"
                    )
                evidence_stamp = finite_time(feedback.evidence.get("observation_timestamp_s"))
                if (
                    abs(evidence_stamp - stamp)
                    > self.brain.memory.observer.config["max_input_skew_s"]
                ):
                    raise ValueError("Feedback evidence timestamp mismatch")
                evidence_key = {
                    "picked": "grasp_confirmed",
                    "placed": "release_confirmed",
                    "verified": "placement_verified",
                }[status]
                if (
                    feedback.evidence.get(evidence_key) is not True
                    or feedback.evidence.get("target_instance_id") != step["target_instance_id"]
                ):
                    raise ValueError("Feedback evidence does not confirm this target/stage")
                if (
                    status == "verified"
                    and feedback.evidence.get("destination_region_id")
                    != step["destination_region_id"]
                ):
                    raise ValueError("Verification evidence does not match this destination")
            if status == "failed" and (
                not isinstance(feedback.error_code, str) or not feedback.error_code
            ):
                raise ValueError("Failure feedback requires an error code")
            next_status = {
                "started": "executing",
                "picked": "picked",
                "placed": "awaiting_verification",
                "verified": "succeeded",
                "failed": "failed",
                "cancelled": "cancelled",
            }[status]
            step["status"] = next_status
            self.phase = next_status
            if status in ("verified", "failed"):
                self.replan_after_version = self.brain.memory.version
                if status == "verified" and self._current_step() is None:
                    self.phase = "completed"
                else:
                    self.phase = "needs_replan"
            elif status == "cancelled":
                self.phase = "cancelled"
            self.events[feedback.event_id] = encoded
            self.last_feedback_time = stamp
            return {"duplicate": False, **self.state()}

    def refresh_remaining(self, decision, *, now_s):
        now_s = finite_time(now_s)
        with self.lock:
            if self.phase != "needs_replan":
                raise ValueError(
                    "Remaining plan may only be refreshed after step completion/failure"
                )
            if self.revision >= self.max_plan_revisions:
                raise ValueError("Plan revision budget exhausted")
            value = self._accepted_decision(decision, now_s)
            if self.brain.memory.version <= self.replan_after_version:
                raise ValueError("Reobserve before replanning remaining actions")
            if self.brain.last_request["instruction"] != self.instruction:
                raise ValueError("A changed instruction requires a new handoff task")
            if self.brain.last_decision_snapshot.get("task_progress") != self.progress_context():
                raise ValueError(
                    "Remaining decision must include the current task's feedback progress"
                )
            completed = {
                step["target_instance_id"] for step in self.steps if step["status"] == "succeeded"
            }
            if any(a["target_instance_id"] in completed for a in value["actions"]):
                raise ValueError("Replan attempts to grasp an already verified completed target")
            for step in self.steps:
                if step["status"] not in ("succeeded", "failed", "cancelled"):
                    step["status"] = "superseded"
            self._append_revision(value)
            return self.state()

    def progress_context(self):
        with self.lock:
            if self.task_id is None:
                raise ValueError("No admitted task")
            return {
                "schema_version": 1,
                "task_id": self.task_id,
                "plan_revision": self.revision,
                "phase": self.phase,
                "completed_target_ids": [
                    s["target_instance_id"] for s in self.steps if s["status"] == "succeeded"
                ],
                "failed_target_ids": list(
                    dict.fromkeys(
                        s["target_instance_id"]
                        for s in self.steps
                        if s["revision"] == self.revision and s["status"] == "failed"
                    )
                ),
                "pending_target_ids": [
                    s["target_instance_id"]
                    for s in self.steps
                    if s["revision"] == self.revision
                    and s["status"] not in ("succeeded", "failed", "cancelled", "superseded")
                ],
            }

    def state(self):
        with self.lock:
            return {
                "schema_version": 1,
                "task_id": self.task_id,
                "phase": self.phase,
                "plan_revision": self.revision,
                "steps": copy.deepcopy(self.steps),
                "accepted_feedback_events": len(self.events),
                "robot_control_enabled": False,
                "completion_scope": "accepted plan actions with sensor verification; not unseen-task completeness",
            }
