"""Qwen plan -> scoped video track -> aligned RGB-D -> existing skill handoff.

One target per tracker epoch. The caller serializes bind/ingest/prepare/feedback;
buffer_frame alone may run on a capture thread while Qwen is busy. No actuation.
"""

import copy
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from piper_sim.observation import Observation
from piper_sim.scene_observation import InstanceMask, Provenance
from piper_sim.task_handoff import Preparation, TaskHandoffSession, finite_time


@dataclass
class TrackingFrame:
    frame_id: str
    observation: Observation
    calibration: object
    view_id: str
    regions: list

    def validate(self):
        obs = self.observation
        if not isinstance(self.frame_id, str) or not self.frame_id or not np.isfinite(obs.timestamp_s):
            raise ValueError("A tracking frame needs a sensor ID and finite source timestamp")
        if (obs.rgb is None or obs.rgb.dtype != np.uint8 or obs.rgb.ndim != 3
                or obs.rgb.shape[2] != 3 or obs.depth_m is None
                or obs.depth_m.shape != obs.rgb.shape[:2]):
            raise ValueError("Tracking frame requires aligned RGB and metric depth")
        self.calibration.source.require_public()


def marked_image(rgb, mask, label):
    image = Image.fromarray(rgb.copy())
    ys, xs = np.where(mask)
    draw = ImageDraw.Draw(image)
    if len(xs):
        draw.rectangle((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                       outline="#00ff88", width=3)
    draw.text((8, 8), label, fill="white", stroke_width=1, stroke_fill="black")
    image.thumbnail((640, 480))
    return image


class TrackedTaskSession:
    def __init__(self, brain, backend, *, ledger=None, max_buffer_frames=90, max_frame_gap_s=0.25,
                 clock=None, max_buffer_bytes=256 * 2**20, mask_border_margin_px=4):
        if type(max_buffer_frames) is not int or max_buffer_frames < 1:
            raise ValueError("Invalid RGB-D buffer size")
        if not np.isfinite(max_frame_gap_s) or max_frame_gap_s <= 0:
            raise ValueError("Invalid tracking continuity threshold")
        if type(max_buffer_bytes) is not int or max_buffer_bytes < 1:
            raise ValueError("Invalid RGB-D buffer byte limit")
        if clock is not None and not callable(clock):
            raise ValueError("Tracking clock must be callable")
        if type(mask_border_margin_px) is not int or mask_border_margin_px < 0:
            raise ValueError("Invalid target mask border margin")
        self.brain, self.backend = brain, backend
        self.ledger = ledger if ledger is not None else TaskHandoffSession(brain)
        if self.ledger.brain is not brain:
            raise ValueError("Tracking and handoff must share the same brain session")
        self.max_buffer_frames, self.max_frame_gap_s = max_buffer_frames, max_frame_gap_s
        self.clock, self.max_buffer_bytes = clock, max_buffer_bytes
        self.mask_border_margin_px = mask_border_margin_px
        self.buffer_bytes = 0
        self.buffer, self.buffer_lock = deque(), threading.RLock()
        self._recent_ids, self._recent_order = set(), deque()
        self._last_received_s = None
        self._published_version = self._published_frame_id = None
        self.buffer_overflow = False
        self.epoch = self.target_id = self.latest = self.source = None
        self.initial_mask = self.current_mask = self.pending = None
        self.index = -1
        self.status = "unbound"
        self.identity_required = False
        self.events = deque(maxlen=128)
        self.last_identity_inference = None
        self.last_frame_metrics = None
        self.frame_metrics = deque(maxlen=1024)

    def _now(self, now_s):
        now = finite_time(now_s)
        return now if self.clock is None else max(now, finite_time(self.clock()))

    def _finished_at(self, now_s, started):
        if self.clock is not None:
            return self._now(now_s)
        return now_s + time.perf_counter() - started

    def _validate_sensor_time(self, frame):
        if self.clock is not None and frame.observation.timestamp_s > finite_time(self.clock()) + 1e-6:
            raise ValueError("Tracking sensor frame is from the future")

    @staticmethod
    def _frame_bytes(frame):
        obs = frame.observation
        return sum(np.asarray(value).nbytes for value in (
            obs.rgb, obs.depth_m, obs.joint_position_rad, obs.joint_velocity_rad_s,
            obs.T_base_end, frame.calibration.T_end_camera))

    def _remember_id(self, frame_id):
        if frame_id in self._recent_ids:
            raise ValueError("Duplicate or reused sensor frame ID")
        self._recent_ids.add(frame_id)
        self._recent_order.append(frame_id)
        # Recent replay protection is bounded independently of image resolution/session duration.
        if len(self._recent_order) > max(4096, self.max_buffer_frames * 2):
            self._recent_ids.remove(self._recent_order.popleft())

    def _receive(self, frame):
        if self.status == "closed":
            raise ValueError("Tracking session is closed")
        self._validate_sensor_time(frame)
        stamp = frame.observation.timestamp_s
        if self._last_received_s is not None and stamp <= self._last_received_s:
            raise ValueError("Duplicate or out-of-order sensor frame")
        self._remember_id(frame.frame_id)
        self._last_received_s = stamp

    def _memory_current(self):
        memory = self.brain.memory
        return (memory.latest_frame is not None and self.latest is not None
                and memory.version == self._published_version
                and memory.latest_frame.frame_id == self._published_frame_id
                and memory.latest_frame.timestamp_s == self.latest.observation.timestamp_s)

    def buffer_frame(self, frame):
        """Retain real sensor frames without mutating the Qwen memory snapshot."""
        frame.validate()
        with self.buffer_lock:
            self._receive(frame)
            previous = self.buffer[-1] if self.buffer else self.latest
            if previous is not None and (frame.frame_id == previous.frame_id
                    or frame.observation.timestamp_s <= previous.observation.timestamp_s):
                raise ValueError("Duplicate or out-of-order sensor frame")
            size = self._frame_bytes(frame)
            while self.buffer and (len(self.buffer) >= self.max_buffer_frames
                                   or self.buffer_bytes + size > self.max_buffer_bytes):
                self.buffer_bytes -= self._frame_bytes(self.buffer.popleft())
                self.buffer_overflow = True
            if size > self.max_buffer_bytes:
                self.buffer_overflow = True
                return
            self.buffer.append(copy.deepcopy(frame))
            self.buffer_bytes += size

    def _event(self, code, **details):
        self.events.append({"code": code, "epoch": self.epoch, **details})

    def bind_decision(self, decision, *, now_s, initial_mask=None, source_frame_id=None):
        """Bind only the accepted plan, then replay buffered frames from its source.

        initial_mask may be a fresh SAM refinement on exactly the source RGB.
        A connected-component proposal is insufficient for automatic initialization.
        """
        started = time.perf_counter()
        now_s = self._now(now_s)
        if self.status == "closed":
            raise ValueError("Tracking session is closed")
        if self.epoch is not None:
            raise ValueError("Create a new tracking session for a new epoch/plan")
        memory, frame = self.brain.memory, self.brain.memory.latest_frame
        if frame is None or decision != self.brain.last_decision or decision["status"] != "ready":
            raise ValueError("Tracking needs the brain's accepted ready decision")
        target = decision["actions"][0]["target_instance_id"]
        local = [key for key, stable in frame.instances.items() if stable == target]
        if len(local) != 1:
            return Preparation("need_observation", code="target_outside_source_frame",
                               observe_region="object:" + target)
        obj = next(o for o in frame.scene.objects if o.instance_key == local[0])
        seed = initial_mask or InstanceMask(local[0], obj.observed_mask, frame.timestamp_s,
                                           obj.confidence or 0.0, obj.source, obj.label)
        seed.source.require_public()
        if (seed.mask.dtype != bool or seed.mask.shape != frame.rgb.shape[:2]
                or not seed.mask.any() or abs(seed.timestamp_s - frame.timestamp_s) > 1e-6):
            raise ValueError("Initialization mask must belong to the exact source RGB-D frame")
        if "RGB-D above-plane connected component" in seed.source.detail:
            return Preparation("need_observation", code="initial_segmentation_required",
                               message="请先为选中实例提供SAM或等效实例分割mask。")
        if not (seed.mask & obj.observed_mask).any():
            raise ValueError("Initial segmentation does not overlap the selected candidate")
        obs = Observation(frame.timestamp_s, frame.joint_position_rad, frame.joint_velocity_rad_s,
                          frame.gripper_width_m, frame.scene.T_base_end, frame.rgb,
                          frame.depth_m, frame.scene.intrinsics)
        from piper_sim.scene_observation import CalibrationEstimate
        source = TrackingFrame(source_frame_id or "memory:" + frame.frame_id, obs,
                               CalibrationEstimate(frame.scene.T_end_camera_estimated,
                                                   frame.scene.calibration_source),
                               frame.view_id, list(memory.regions.values()))
        source.validate()
        self._validate_sensor_time(source)
        with self.buffer_lock:
            if self.buffer_overflow:
                return Preparation("need_observation", code="replay_buffer_overflow")
            if self.buffer and self.buffer[0].observation.timestamp_s <= source.observation.timestamp_s:
                raise ValueError("Buffered frames must follow the initialization frame")
            self._remember_id(source.frame_id)
            self._last_received_s = max(source.observation.timestamp_s,
                                        self._last_received_s or source.observation.timestamp_s)
        if self.ledger.phase == "needs_replan":
            self.ledger.refresh_remaining(decision, now_s=now_s)
        else:
            self.ledger.accept_plan(decision, now_s=now_s)
        self.epoch, self.target_id, self.source = uuid.uuid4().hex, target, copy.deepcopy(source)
        self.initial_mask = seed.mask.copy()
        try:
            mask = self.backend.initialize(source.observation.rgb, seed.mask)
            self.index = 0
            self._consume(source, mask)
            self.replay_buffer()
        except Exception:
            self.status = "tracker_error"
            self.current_mask = None
            self.ledger.packet = None
            self.backend.close()
            raise
        self._event("bound", target_instance_id=target, source_frame_id=source.frame_id)
        # Replay tests may run slower than the recorded sensor clock: never invent wall timestamps.
        return self.prepare_current(now_s=self._finished_at(now_s, started))

    def replay_buffer(self):
        with self.buffer_lock:
            if self.status == "closed":
                return
            watermark = None if not self.buffer else self.buffer[-1].observation.timestamp_s
        while True:
            with self.buffer_lock:
                if self.buffer_overflow:
                    self.identity_required, self.status = True, "buffer_overflow"
                    self.current_mask = None
                    self.ledger.packet = None
                    return
                if not self.buffer or watermark is None or self.buffer[0].observation.timestamp_s > watermark:
                    return
                frame = self.buffer.popleft()
                self.buffer_bytes -= self._frame_bytes(frame)
            self._ingest(frame, buffered=True)

    def ingest(self, frame):
        if self.epoch is None or self.status in ("tracker_error", "closed"):
            raise ValueError("Initialize a tracker epoch before ingesting frames")
        self.replay_buffer()
        if self.buffer_overflow:
            return self.state()
        return self._ingest(frame, buffered=False)

    def _ingest(self, frame, *, buffered):
        if self.epoch is None or self.status in ("tracker_error", "closed"):
            raise ValueError("Initialize a tracker epoch before ingesting frames")
        if (self.latest is not None and (frame.frame_id == self.latest.frame_id
                or frame.observation.timestamp_s <= self.latest.observation.timestamp_s)):
            raise ValueError("Duplicate or out-of-order sensor frame")
        try:
            started = time.perf_counter()
            frame.validate()
            if not buffered:
                with self.buffer_lock:
                    self._receive(frame)
            if frame.observation.rgb.shape != self.source.observation.rgb.shape:
                raise ValueError("Image resolution changed; initialize a fresh tracker epoch")
            mask = self.backend.step(frame.observation.rgb, self.index + 1)
            self.index += 1
            self._consume(copy.deepcopy(frame), mask)
            self.last_frame_metrics = {
                "frame_id": frame.frame_id,
                "processing_s": time.perf_counter() - started,
                "source_age_s": None if self.clock is None
                else finite_time(self.clock()) - frame.observation.timestamp_s}
            self.frame_metrics.append(self.last_frame_metrics.copy())
            return self.state()
        except Exception:
            self.status, self.current_mask = "tracker_error", None
            self.ledger.packet = None
            self.backend.close()
            raise

    def _geometry_consistent(self, frame, mask):
        detection = InstanceMask("check", mask, frame.observation.timestamp_s, 0.0,
                                 Provenance("sensor_estimate", "Video tracker identity gate"))
        scene = self.brain.memory.observer.build(
            frame.observation, frame.calibration, now_s=frame.observation.timestamp_s,
            detections=[detection], regions=frame.regions, allow_geometric_proposals=False,
            estimate_table=False)
        if not scene.objects or len(scene.objects[0].points_base) < self.brain.memory.cfg["min_object_points"]:
            return "no_depth"
        previous = self.brain.memory.objects.get(self.target_id)
        if previous is None:
            return "identity_required"
        distance = np.linalg.norm(np.median(scene.objects[0].points_base, axis=0)
                                  - np.median(previous.latest.points_base, axis=0))
        # This is a continuity gate, not physical identity proof or a motion controller.
        dt = frame.observation.timestamp_s - previous.last_seen_s
        if distance > min(0.08, 0.01 + max(0.0, dt) * 0.5):
            return "identity_required"
        return "tracking"

    def _consume(self, frame, mask):
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != frame.observation.rgb.shape[:2]:
            raise ValueError("Tracker returned an unaligned mask")
        if self.latest is not None and (frame.observation.timestamp_s
                - self.latest.observation.timestamp_s > self.max_frame_gap_s):
            self.identity_required = True
            self._event("frame_gap", frame_id=frame.frame_id)
        self.pending = None
        if not mask.any():
            self.identity_required = True
            self.status = "not_visible"
        elif self._mask_clipped(mask):
            # Continue the visual track, but a clipped mask must not change task geometry
            # or be mistaken for a smaller/moving object during plan revalidation.
            self.status = "partial_view"
        elif self.identity_required:
            self.status = "identity_required"
            self.pending = (copy.deepcopy(frame), mask.copy())
        else:
            self.status = self._geometry_consistent(frame, mask)
            if self.status == "identity_required":
                self.identity_required = True
                self.pending = (copy.deepcopy(frame), mask.copy())
        self.latest = frame
        self.current_mask = mask.copy() if self.status == "tracking" else None
        self._publish(frame, self.current_mask)
        self._event(self.status, frame_id=frame.frame_id, timestamp_s=frame.observation.timestamp_s,
                    mask_pixels=int(mask.sum()))
        return self.state()

    def _mask_clipped(self, mask):
        margin = self.mask_border_margin_px
        return bool(margin and (mask[:margin].any() or mask[-margin:].any()
                               or mask[:, :margin].any() or mask[:, -margin:].any()))

    def _publish(self, frame, mask):
        key = f"video:{self.epoch}:1"
        detections = [] if mask is None else [InstanceMask(
            key, mask, frame.observation.timestamp_s, 0.0,
            Provenance("sensor_estimate", f"video-mask tracker epoch={self.epoch}; confidence uncalibrated"))]
        published = self.brain.observe(frame.observation, frame.calibration, detections=detections,
                           view_id=frame.view_id, regions=frame.regions,
                           now_s=frame.observation.timestamp_s,
                           identity_bindings={} if mask is None else {key: self.target_id},
                           allow_geometric_proposals=False)
        self._published_version = self.brain.memory.version
        self._published_frame_id = published.frame_id
        # Any new observation invalidates a previously prepared dispatch packet.
        self.ledger.packet = None

    def confirm_identity(self, qwen_session, *, now_s):
        """Low-frequency before/after identity query; ambiguous/different never publishes a mask."""
        now_s = self._now(now_s)
        if self.pending is None:
            raise ValueError("No current reacquisition candidate")
        frame, mask = self.pending
        if not self._memory_current():
            return Preparation("need_observation", code="tracking_memory_superseded")
        if now_s < frame.observation.timestamp_s or now_s - frame.observation.timestamp_s > self.brain.memory.cfg["observation_max_age_s"]:
            return Preparation("need_observation", code="reacquisition_candidate_stale")
        context = {"target_instance_id": self.target_id, "tracker_epoch": self.epoch,
                   "frame_id": frame.frame_id, "instruction": self.ledger.instruction}
        images = [marked_image(self.source.observation.rgb, self.initial_mask, "original target"),
                  marked_image(frame.observation.rgb, mask, "reacquisition candidate")]
        messages = [{"role": "system", "content": (
            "Compare marked physical objects across time. Return only JSON with exactly status, "
            "target_instance_id, tracker_epoch, frame_id. status must be same_object, different_object "
            "or ambiguous. Appearance alone does not prove identity; choose ambiguous if uncertain.")},
            {"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                          {"type": "text", "text": json.dumps(context)}]}]
        started = time.perf_counter()
        predict = getattr(qwen_session, "predict_identity_messages", qwen_session.predict_messages)
        raw, metrics = predict(images, messages)
        self.last_identity_inference = {"raw": raw, "metrics": copy.deepcopy(metrics), **context}
        result = json.loads(raw)
        if (set(result) != {"status", "target_instance_id", "tracker_epoch", "frame_id"}
                or any(result[key] != context[key] for key in ("target_instance_id", "tracker_epoch", "frame_id"))
                or result["status"] not in ("same_object", "different_object", "ambiguous")):
            raise ValueError("Identity reply does not match the candidate frame and tracker epoch")
        if self.pending is None or self.pending[0].frame_id != frame.frame_id or not self._memory_current():
            raise ValueError("Identity candidate changed during Qwen inference")
        if result["status"] != "same_object":
            return Preparation("need_observation", code="identity_" + result["status"])
        if self._geometry_consistent(frame, mask) != "tracking":
            return Preparation("needs_replan", code="reacquisition_geometry_changed")
        self.identity_required, self.pending = False, None
        self.status, self.current_mask = "tracking", mask.copy()
        self._publish(frame, mask)
        self._event("identity_revalidated", frame_id=frame.frame_id, source="qwen_estimate")
        self.replay_buffer()
        return self.prepare_current(now_s=self._finished_at(now_s, started))

    def prepare_current(self, *, now_s):
        now_s = self._now(now_s)
        if self.latest is not None and now_s < self.latest.observation.timestamp_s:
            raise ValueError("Preparation timestamp precedes the sensor frame")
        if self.status != "tracking":
            return Preparation("need_observation", code=self.status,
                               observe_region=None if self.target_id is None else "object:" + self.target_id)
        if not self._memory_current():
            self.ledger.packet = None
            return Preparation("need_observation", code="tracking_memory_superseded")
        previous_id = None if self.ledger.packet is None else self.ledger.packet.metadata["handoff_id"]
        result = self.ledger.prepare_current(now_s=now_s)
        if result.packet is not None:
            finished_at = self._now(now_s)
            if finished_at > result.packet.metadata["expires_at_s"]:
                self.ledger.packet = None
                self.ledger.phase = "waiting_observation"
                self.ledger._current_step()["status"] = "waiting_observation"
                return Preparation("need_observation", code="preparation_expired")
            if result.packet.metadata["handoff_id"] != previous_id:
                result.packet.metadata["prepared_at_s"] = finished_at
            result.packet.metadata["perception_binding"] = {
                "stable_target_id": self.target_id, "tracker_epoch": self.epoch, "track_id": 1,
                "sensor_frame_id": self.latest.frame_id, "tracker_frame_index": self.index,
                "mask_timestamp_s": self.latest.observation.timestamp_s,
                "depth_timestamp_s": self.latest.observation.timestamp_s,
                "identity_status": "estimated", "tracking_status": self.status}
            # Keep feedback validation on the same packet identity and provenance.
            self.ledger.packet = copy.deepcopy(result.packet)
        return result

    def apply_feedback(self, feedback, *, now_s):
        now_s = self._now(now_s)
        event_id = feedback.event_id if hasattr(feedback, "event_id") else feedback.get("event_id")
        if event_id in self.ledger.events:
            # The ledger checks exact event content; ACK retries must remain idempotent after loss.
            return self.ledger.apply_feedback(feedback, now_s=now_s)
        if (feedback.status if hasattr(feedback, "status") else feedback.get("status")) == "started":
            fresh = self.prepare_current(now_s=now_s)
            if fresh.packet is None:
                raise ValueError("A current tracked observation is required before skill start")
        return self.ledger.apply_feedback(feedback, now_s=now_s)

    def skill_observation(self, *, now_s, previous_action, previous_action_valid):
        """Fresh geometry/joints for OnlineSkillHistory, including during execution.

        Geometry may change during a grasp. Accepted identity and destination stay
        fixed; lack of a current measured mask always requests observation upstream.
        This method does not infer holding, success, or generate joint commands.
        """
        from piper_sim.learned_skill_execution import public_skill_observation
        now_s = self._now(now_s)
        if self.status != "tracking" or self.current_mask is None:
            raise ValueError("Current tracked geometry is unavailable")
        if not self._memory_current():
            raise ValueError("Tracking memory was superseded by a different observation")
        if self.ledger.phase not in ("prepared", "executing", "picked", "awaiting_verification"):
            raise ValueError("An admitted active skill step is required")
        memory = self.brain.memory
        frame = memory.latest_frame
        if now_s < frame.timestamp_s or now_s - frame.timestamp_s > memory.cfg["observation_max_age_s"]:
            raise ValueError("Current tracked geometry is stale or from the future")
        step = self.ledger._current_step()
        mask, goal = memory.current_selection(self.target_id, step["destination_region_id"],
                                               now_s=now_s, memory_version=memory.version)
        if frame.timestamp_s != self.latest.observation.timestamp_s:
            raise ValueError("Tracking and scene memory refer to different frames")
        scene = copy.deepcopy(frame.scene)
        for entity in scene.objects:
            entity.instance_key = frame.instances.get(entity.instance_key, entity.instance_key)
        scene.regions = [r for r in scene.regions if r.region_id != goal.region_id] + [goal]
        scene.select_target(self.target_id, mask.source)
        result = public_skill_observation(scene, self.latest.observation,
                                          destination_region_id=goal.region_id, now_s=now_s,
                                          previous_action=previous_action,
                                          previous_action_valid=previous_action_valid)
        finished_at = self._now(now_s)
        if finished_at - frame.timestamp_s > memory.cfg["observation_max_age_s"]:
            raise ValueError("Tracked geometry expired during skill feature preparation")
        result["decision_time_s"] = finished_at
        return result

    def state(self):
        return {"tracking_status": self.status, "target_instance_id": self.target_id,
                "tracker_epoch": self.epoch, "track_id": None if self.epoch is None else 1,
                "frame_id": None if self.latest is None else self.latest.frame_id,
                "timestamp_s": None if self.latest is None else self.latest.observation.timestamp_s,
                "identity_confirmation_required": self.identity_required,
                "buffer_overflow": self.buffer_overflow, "robot_control_enabled": False,
                "buffer_frames": len(self.buffer), "buffer_bytes": self.buffer_bytes,
                "last_frame_metrics": copy.deepcopy(self.last_frame_metrics),
                "execution_dispatch_enabled": False, "handoff": self.ledger.state()}

    def close(self):
        self.backend.close()
        self.status, self.current_mask, self.pending = "closed", None, None
        self.ledger.packet = None
        with self.buffer_lock:
            self.buffer.clear()
            self.buffer_bytes = 0
