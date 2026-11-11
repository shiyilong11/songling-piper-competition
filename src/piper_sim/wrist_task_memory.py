"""Multi-frame task adapter for the wrist simulation panel, using public sensors only."""

import base64
import io
import json
import os
import select
import subprocess
import time
import uuid

import numpy as np

from piper_sim.calibration import ROOT, load_config, transform_points
from piper_sim.colored_regions import estimate_colored_regions
from piper_sim.multiframe_demo import make_demo_frame
from piper_sim.multiframe_io import write_frame
from piper_sim.multiframe_task import MultiframeTaskSession
from piper_sim.observation import Observation
from piper_sim.sam_adapter import Sam2Client
from piper_sim.scene_memory import SceneMemory, load_memory_config
from piper_sim.scene_observation import CalibrationEstimate, PlacementRegion, Provenance
from piper_sim.tracked_task import TrackedTaskSession, TrackingFrame
from piper_sim.video_mask_client import DEFAULT_TRACKING_METHOD, TRACKING_METHODS, VideoMaskClient


class PreparedQwenClient:
    """Lazy model subprocess; the panel remains a CPU simulation process."""

    def __init__(self):
        self.worker = self.log = None
        self.request_file = None

    @property
    def loaded(self):
        return self.worker is not None and self.worker.poll() is None

    def predict_messages(self, images, messages):
        if self.request_file is None:
            raise ValueError("Prepare a multiframe request first")
        if not self.loaded:
            self.close()
            python = ROOT / ".local/multiframe-runtime/bin/python"
            if not python.is_file():
                raise RuntimeError("请先运行 scripts/setup_qwen_multiframe.sh")
            env = {
                **os.environ,
                "PYTHONPATH": str(ROOT / "src"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
            env.pop("CUDA_VISIBLE_DEVICES", None)
            self.log = (self.request_file.parent.parent / "multiframe_model.log").open("a")
            self.worker = subprocess.Popen(
                [str(python), str(ROOT / "scripts/qwen_prepared_worker.py")],
                cwd=ROOT,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                bufsize=1,
            )
        request_id = uuid.uuid4().hex
        try:
            self.worker.stdin.write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "request_file": str(self.request_file),
                        "followup_messages": messages[2:],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            self.worker.stdin.flush()
            ready, _, _ = select.select([self.worker.stdout], [], [], 180)
            if not ready:
                raise TimeoutError("多帧推理超时，已释放该模型进程。")
            line = self.worker.stdout.readline()
            if not line:
                raise RuntimeError("多帧模型进程退出，请查看 multiframe_model.log")
            result = json.loads(line)
            if result.get("request_id") != request_id:
                raise RuntimeError("多帧推理响应与请求不匹配")
            if result.get("error"):
                raise RuntimeError(result["error"])
            return result["raw"], result["metrics"]
        except Exception:
            self.close()
            raise

    def predict_identity_messages(self, images, messages):
        if self.request_file is None:
            raise ValueError("Prepare the task request before identity confirmation")
        previous = self.request_file
        folder = previous.parent.parent / ("identity_" + uuid.uuid4().hex)
        folder.mkdir(parents=True)
        names = []
        for i, image in enumerate(images):
            name = f"image_{i}.png"
            image.save(folder / name)
            names.append(name)
        self.request_file = folder / "request.json"
        self.request_file.write_text(json.dumps({
            "request_kind": "tracking_identity", "messages": messages,
            "annotated_image_files": names}, ensure_ascii=False, indent=2))
        try:
            return self.predict_messages(images, messages)
        finally:
            self.request_file = previous

    def close(self):
        if self.worker is not None:
            self.worker.terminate()
            try:
                self.worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait(timeout=5)
            for stream in (self.worker.stdin, self.worker.stdout):
                stream.close()
            self.worker = None
        if self.log is not None:
            self.log.close()
            self.log = None


class WristTaskMemory:
    def __init__(self, output, hand_eye_config, *, tracking_method=DEFAULT_TRACKING_METHOD,
                 sensor_only=True):
        if type(sensor_only) is not bool:
            raise ValueError("sensor_only must be boolean")
        self.sensor_only = sensor_only
        if tracking_method not in TRACKING_METHODS:
            raise ValueError("Unknown video-mask method")
        self.tracking_method = tracking_method
        self.output = output / "multiframe"
        self.output.mkdir(parents=True)
        hand_eye = load_config(ROOT / hand_eye_config)
        self.calibration = CalibrationEstimate(
            np.asarray(hand_eye["hand_eye"]["T_end_camera"], float),
            Provenance("manual_config", f"Simulation calibration placeholder: {hand_eye_config}"),
        )
        self.qwen = PreparedQwenClient()
        self.sam = Sam2Client()
        self.reset()

    def reset(self):
        if getattr(self, "tracking", None) is not None:
            self.tracking.close()
        self.tracking = None
        self.understanding_in_progress = False
        self.task = MultiframeTaskSession()
        self.regions = {}
        self.source = "wrist_rgbd"
        self.latest_observation = None
        self.last_result = None
        self.last_recording = None

    def _observe(self, obs, calibration, *, view_id, detections=()):
        detections = tuple(detections)
        regions = list(self.regions.values())
        if self.sensor_only:
            self._require_estimates([*detections, *regions])
        if self.source == "wrist_rgbd":
            scene = self.task.memory.observer.build(
                obs,
                calibration,
                now_s=obs.timestamp_s,
                detections=detections,
                regions=regions,
            )
            regions += [
                r for r in estimate_colored_regions(obs, scene) if r.region_id not in self.regions
            ]
        self.task.observe(
            obs,
            calibration,
            view_id=view_id,
            detections=detections,
            regions=regions,
            now_s=obs.timestamp_s,
        )
        record = self.output / "records" / uuid.uuid4().hex
        # Record the supplied masks. Empty masks mean SceneObserver uses RGB-D proposals.
        self.last_recording = write_frame(record, obs, calibration, detections, view_id, regions)
        self.latest_observation = obs
        self.last_result = None

    @staticmethod
    def _require_estimates(items):
        for item in items:
            item.source.require_public()
            if item.source.kind != "sensor_estimate":
                raise ValueError("Sensor-only L3 requires estimated masks/regions; manual fixtures forbidden")

    def _require_sensor_memory(self):
        if not self.sensor_only:
            return
        if self.source != "wrist_rgbd":
            raise ValueError("Sensor-only L3 forbids synthetic demonstration input")
        self._require_estimates(self.task.memory.regions.values())
        for frame in self.task.memory.frames:
            self._require_estimates([*frame.scene.objects, *frame.scene.regions])

    def capture(self, sensors, view_id="center"):
        if view_id not in self.task.memory.cfg["required_views"] or self.source != "wrist_rgbd":
            if self.source != "wrist_rgbd":
                self.reset()
            if view_id not in self.task.memory.cfg["required_views"]:
                raise ValueError("Unknown survey view")
        obs = Observation(
            time.monotonic(),
            sensors["joint_position_rad"],
            sensors["joint_velocity_rad_s"],
            sensors["gripper_width_m"],
            sensors["T_base_end"],
            sensors["rgb"],
            sensors["depth_m"],
            sensors["intrinsics"],
        )
        if self.tracking is not None:
            frame = TrackingFrame("sensor:" + uuid.uuid4().hex, obs, self.calibration,
                                  view_id, list(self.regions.values()))
            if self.understanding_in_progress or self.tracking.epoch is None:
                self.tracking.buffer_frame(frame)
            else:
                self.tracking.replay_buffer()
                self.tracking.ingest(frame)
            self.latest_observation = obs
        else:
            self._observe(obs, self.calibration, view_id=view_id)

    def buffer_tracking_sensors(self, sensors, view_id="center"):
        """Capture-owner entry during model work: never update brain or invoke the tracker."""
        if self.tracking is None:
            return
        obs = Observation(time.monotonic(), sensors["joint_position_rad"],
                          sensors["joint_velocity_rad_s"], sensors["gripper_width_m"],
                          sensors["T_base_end"], sensors["rgb"], sensors["depth_m"], sensors["intrinsics"])
        self.tracking.buffer_frame(TrackingFrame("sensor:" + uuid.uuid4().hex, obs,
                                                   self.calibration, view_id, list(self.regions.values())))

    def demo(self):
        if self.sensor_only:
            raise ValueError("Sensor-only L3 forbids synthetic demonstration input")
        self.reset()
        cfg = load_memory_config()
        cfg["required_views"] = ["left", "right"]
        self.task = MultiframeTaskSession(SceneMemory(cfg))
        self.source = "synthetic_two_views"
        stamp = time.monotonic()
        for y, view_id, delta in [(0.14, "left", -0.1), (-0.14, "right", 0)]:
            frame = make_demo_frame(y, y, stamp + delta, view_id=view_id)
            self.regions = {r.region_id: r for r in frame["regions"]}
            self._observe(
                frame["observation"],
                frame["calibration"],
                view_id=view_id,
                detections=frame["detections"],
            )
        return self.state()

    def point(self, point_xy):
        if self.source != "wrist_rgbd" or self.latest_observation is None:
            raise ValueError("请先采集真实仿真腕部画面。")
        point = np.asarray(point_xy, dtype=float)
        h, w = self.latest_observation.depth_m.shape
        if (
            point.shape != (2,)
            or not np.isfinite(point).all()
            or not (0 <= point[0] < w and 0 <= point[1] < h)
        ):
            raise ValueError("请在腕部图像内部选点。")
        return point

    def add_region(self, region_id, point_xy, half_size_xy_m):
        if self.sensor_only:
            raise ValueError("Sensor-only L3 requires visually estimated regions, not manual task annotations")
        point = self.point(point_xy)
        if not isinstance(region_id, str) or not region_id.strip() or len(region_id) > 60:
            raise ValueError("区域 ID 应为1到60字。")
        half = np.asarray(half_size_xy_m, float)
        if (
            half.shape != (2,)
            or not np.isfinite(half).all()
            or np.any(half <= 0)
            or np.any(half > 0.3)
        ):
            raise ValueError("区域半尺寸须在0到0.3米之间。")
        obs = self.latest_observation
        x, y = np.floor(point).astype(int)
        depth = obs.depth_m[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
        valid = depth[np.isfinite(depth) & (depth > 0)]
        if not len(valid):
            raise ValueError("该点没有有效深度，请重新选点。")
        z = float(np.median(valid))
        k = obs.intrinsics
        camera_point = np.array(
            [[(point[0] - k["cx"]) * z / k["fx"], (point[1] - k["cy"]) * z / k["fy"], z]]
        )
        pose = np.eye(4)
        pose[:3, 3] = transform_points(
            obs.T_base_end @ self.calibration.T_end_camera, camera_point
        )[0]
        region = PlacementRegion(
            region_id.strip(),
            pose,
            half,
            Provenance(
                "manual_config",
                "User marked wrist RGB-D centre; configured XY half-size; static simulation annotation",
            ),
        )
        region.validate()
        self.regions[region.region_id] = region
        self.task.memory.regions[region.region_id] = region
        self.task.memory.version += 1
        self.last_result = None
        return self.state()

    def segment(self, point_xy):
        if self.sensor_only:
            raise ValueError("Sensor-only L3 requires model-selected targets, not manual task annotations")
        point = self.point(point_xy)
        obs = self.latest_observation
        folder = self.output / "sam" / uuid.uuid4().hex
        mask, metadata = self.sam.segment(
            obs.rgb, point.tolist(), obs.timestamp_s, self.task.memory.version, folder
        )
        # Keep unrelated geometric candidates; replace candidates overlapping this SAM mask.
        detections = [mask]
        from piper_sim.scene_observation import InstanceMask

        for obj in self.task.memory.latest_frame.scene.objects:
            overlap = np.count_nonzero(obj.observed_mask & mask.mask)
            if overlap / max(1, np.count_nonzero(obj.observed_mask)) < 0.2:
                detections.append(
                    InstanceMask(
                        obj.instance_key,
                        obj.observed_mask,
                        obs.timestamp_s,
                        obj.confidence or 0.0,
                        obj.source,
                        obj.label,
                    )
                )
        view_id = self.task.memory.latest_frame.view_id
        self._observe(obs, self.calibration, view_id=view_id, detections=detections)
        return {**self.state(), "sam_metrics": metadata}

    def prepare(self, instruction, *, task_progress=None):
        self._require_sensor_memory()
        if (self.sensor_only and task_progress is not None
                and (self.tracking is None or task_progress != self.tracking.ledger.progress_context())):
            raise ValueError("Sensor-only L3 requires progress from its measured execution ledger")
        folder = self.output / "requests" / uuid.uuid4().hex
        request = self.task.save_request(
            folder, instruction, now_s=time.monotonic(), task_progress=task_progress
        )
        self.qwen.request_file = folder / "request.json"
        return request, folder

    def infer(self, instruction, *, task_progress=None, track=True, buffer_ready=None):
        # Freeze the scene memory during inference; capture may only buffer RGB-D frames.
        previous_ledger = None
        if self.tracking is not None:
            if self.tracking.ledger.phase in ("executing", "picked", "awaiting_verification"):
                raise ValueError("请先完成或取消当前技能，再重新理解指令。")
            if self.tracking.ledger.phase == "needs_replan":
                previous_ledger = self.tracking.ledger
                if task_progress is None:
                    task_progress = previous_ledger.progress_context()
            self.tracking.close()
            self.tracking = None
        if track and self.source == "wrist_rgbd":
            self.tracking = TrackedTaskSession(
                self.task, VideoMaskClient(self.output / "tracking", method=self.tracking_method),
                ledger=previous_ledger, clock=time.monotonic,
            )
        _, folder = self.prepare(instruction, task_progress=task_progress)
        started = time.monotonic()
        self.understanding_in_progress = True
        if buffer_ready is not None:
            buffer_ready.set()
        try:
            decision, details = self.task.decide(
                self.qwen, instruction, now_s=time.monotonic(), task_progress=task_progress
            )
            result = {
                "decision": decision,
                **details,
                "wall_time_s": time.monotonic() - started,
                "request_path": str(folder),
                "source": self.source,
            }
            self.last_result = result
            if self.tracking is not None and decision["status"] == "ready":
                prepared = self.start_tracking(decision)
                result["tracking"] = self.tracking.state()
                result["skill_preparation"] = {"status": prepared.status, "code": prepared.code}
            if self.tracking is not None and self.tracking.epoch is None:
                # Unseen/missing targets need fresh scene observations before another decision.
                self.tracking.close()
                self.tracking = None
            (folder / "decision.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            return result
        except Exception as exc:
            failure = {
                "error": str(exc),
                "inference_executed": self.task.last_inference is not None,
                "details": self.task.last_inference,
                "robot_control_enabled": False,
            }
            (folder / "error.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2))
            raise
        finally:
            self.understanding_in_progress = False

    def start_tracking(self, decision=None):
        self._require_sensor_memory()
        decision = decision or self.task.last_decision
        if self.tracking is None:
            self.tracking = TrackedTaskSession(
                self.task, VideoMaskClient(self.output / "tracking", method=self.tracking_method),
                clock=time.monotonic,
            )
        if self.tracking.epoch is not None:
            return self.tracking.prepare_current(now_s=time.monotonic())
        # Refine the Qwen-selected current candidate with SAM before starting video tracking.
        frame = self.task.memory.latest_frame
        seed = None
        if decision is not None and decision["status"] == "ready":
            target = decision["actions"][0]["target_instance_id"]
            candidates = [o for o in frame.scene.objects
                          if frame.instances.get(o.instance_key) == target]
            if candidates:
                from scipy.ndimage import distance_transform_edt
                obj = candidates[0]
                ys, xs = np.where(obj.observed_mask)
                y, x = np.unravel_index(distance_transform_edt(obj.observed_mask).argmax(),
                                       obj.observed_mask.shape)
                seed, _ = self.sam.segment(
                    frame.rgb, [int(x), int(y)], frame.timestamp_s, frame.frame_id,
                    self.output / "sam" / uuid.uuid4().hex,
                    box_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
        return self.tracking.bind_decision(decision, now_s=time.monotonic(), initial_mask=seed)

    def prepare_skill(self):
        self._require_sensor_memory()
        if self.tracking is None:
            raise ValueError("请先理解指令并启动目标跟踪。")
        result = self.tracking.prepare_current(now_s=time.monotonic())
        path = None if result.packet is None else str(result.packet.export(
            self.output / "handoffs" / result.packet.metadata["handoff_id"]))
        return {"status": result.status, "code": result.code, "packet_path": path,
                "tracking": self.tracking.state(), "robot_control_enabled": False}

    def skill_feedback(self, data):
        if self.sensor_only:
            # A browser JSON field saying sensor_estimate does not establish a
            # measured result. Strict L3 needs a trusted independent verifier.
            raise ValueError("Sensor-only L3 requires a connected sensor verifier; interactive feedback forbidden")
        if self.tracking is None:
            raise ValueError("没有已交接的技能任务。")
        from piper_sim.task_handoff import ExecutionFeedback
        feedback = ExecutionFeedback.from_dict(data)
        return self.tracking.apply_feedback(feedback, now_s=time.monotonic())

    def confirm_tracking(self):
        if self.tracking is None:
            raise ValueError("没有正在跟踪的任务。")
        self.understanding_in_progress = True
        try:
            result = self.tracking.confirm_identity(self.qwen, now_s=time.monotonic())
            return {"status": result.status, "code": result.code,
                    "tracking": self.tracking.state(), "robot_control_enabled": False}
        finally:
            self.understanding_in_progress = False

    def state(self):
        if self.task.memory.latest_frame is None:
            return {
                "source": self.source,
                "snapshot": None,
                "keyframes": [],
                "model_loaded": self.qwen.loaded,
                "sensor_only_mode": self.sensor_only,
                "tracking_method": self.tracking_method,
                "robot_control_enabled": False,
            }
        snapshot = self.task.memory.summary(time.monotonic())
        frames = []
        for frame in self.task.memory.frames:
            image = frame.annotated_image()
            image.thumbnail((280, 210))
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            frames.append(
                {
                    "frame_id": frame.frame_id,
                    "view_id": frame.view_id,
                    "instance_ids": list(frame.instances.values()),
                    "rgb": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode(),
                }
            )
        return {
            "source": self.source,
            "snapshot": snapshot,
            "keyframes": frames,
            "model_loaded": self.qwen.loaded,
            "sensor_only_mode": self.sensor_only,
            "last_result": self.last_result,
            "tracking": None if self.tracking is None else self.tracking.state(),
            "tracking_method": self.tracking_method,
            "robot_control_enabled": False,
        }

    def release(self):
        if self.tracking is not None:
            self.tracking.close()
        self.qwen.close()
        self.sam.close()
