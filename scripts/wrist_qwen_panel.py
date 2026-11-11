"""Local wrist-camera test panel; CPU simulation and an isolated on-demand GPU worker."""

import argparse
import base64
import copy
import json
import os
import select
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image

from piper_sim.calibration import ROOT
from piper_sim.competition_world import CompetitionWorld
from piper_sim.qwen_adapter import InvalidPrediction
from piper_sim.video_mask_client import DEFAULT_TRACKING_METHOD
from piper_sim.wrist_camera import wrist_scene_config
from piper_sim.wrist_task_memory import WristTaskMemory


class CameraPanel:
    def __init__(self, output):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.settings, cfg = wrist_scene_config()
        self.config_path = self.output / "scene_config.yaml"
        self.config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
        self.world = CompetitionWorld(
            self.settings["task"], self.config_path, self.output / "scene", render=True
        )
        self.worker = None
        self.worker_log = None
        self.sequence = 0
        self.requests = 0
        self.seed = self.settings["seed"]
        self.task_memory = WristTaskMemory(
            self.output, self.settings["hand_eye_config"],
            tracking_method=self.settings.get("tracking_method", DEFAULT_TRACKING_METHOD),
            sensor_only=self.settings.get("sensor_only", True),
        )
        self.view_id = "center"
        self.world.reset(self.seed)
        self.capture()

    def capture(self, view_id=None, *, remember=True):
        if not isinstance(remember, bool):
            raise TypeError("remember must be boolean")
        view_id = self.view_id if view_id is None else view_id
        if view_id not in ("left", "center", "right"):
            raise ValueError("Unknown survey view")
        self.view_id = view_id
        self.sequence += 1
        frame = self.output / "frames" / f"{self.sequence:06d}"
        frame.mkdir(parents=True)
        observation = self.world.observe()
        if self.task_memory.source != "wrist_rgbd":
            self.task_memory.reset()
        if remember:
            self.task_memory.capture(observation, view_id)
        Image.fromarray(observation.pop("rgb")).save(frame / "wrist_rgb.png")
        np.save(frame / "wrist_depth_m.npy", observation.pop("depth_m"))
        metadata = {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in observation.items()
        }
        metadata.update(
            frame_id=self.sequence,
            seed=self.seed,
            mount="wrist",
            parent_body=self.world.cfg["camera"]["parent_body"],
            memory_checkpoint=remember,
        )
        (frame / "observation.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        b = self.world.backend
        with mujoco.Renderer(b.model, height=480, width=640) as renderer:
            camera = mujoco.MjvCamera()
            view = self.world.cfg.get("overview", {})
            camera.lookat[:] = view.get("lookat_m", [0.32, -0.02, 0.15])
            camera.distance = view.get("distance_m", 1.05)
            camera.azimuth = view.get("azimuth_deg", 135)
            camera.elevation = view.get("elevation_deg", -35)
            option = mujoco.MjvOption()
            option.geomgroup[3] = False
            renderer.update_scene(b.data, camera=camera, scene_option=option)
            Image.fromarray(renderer.render().copy()).save(frame / "overview.png")
        self.frame = frame
        self.metadata = metadata
        return self.state()

    def state(self):
        b = self.world.backend

        def data_url(name):
            return (
                "data:image/png;base64,"
                + base64.b64encode((self.frame / name).read_bytes()).decode()
            )

        return {
            "frame_id": self.sequence,
            "seed": self.seed,
            "mount": "wrist",
            "parent_body": self.world.cfg["camera"]["parent_body"],
            "rgb": data_url("wrist_rgb.png"),
            "overview": data_url("overview.png"),
            "q": b.data.qpos[b.qindices].tolist(),
            "limits": b.model.jnt_range[b.joints].tolist(),
            "T_base_camera": self.metadata["T_base_camera"],
            "model_loaded": (self.worker is not None and self.worker.poll() is None)
            or self.task_memory.qwen.loaded,
            "multiframe": self.task_memory.state(),
            "view_id": self.view_id,
            "scan_poses": self.scan_poses(),
            "frame_path": str(self.frame),
            "robot_control_enabled": False,
        }

    def reset(self, seed):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")
        self.seed = seed
        self.world.reset(seed)
        self.task_memory.reset()
        return self.capture("center")

    def pose(self, q, view_id=None, *, remember=True):
        if not isinstance(remember, bool):
            raise TypeError("remember must be boolean")
        if view_id is not None and view_id not in ("left", "center", "right"):
            raise ValueError("Unknown survey view")
        b = self.world.backend
        q = np.asarray(q, dtype=float)
        limits = b.model.jnt_range[b.joints]
        if (
            q.shape != (6,)
            or not np.isfinite(q).all()
            or np.any(q < limits[:, 0])
            or np.any(q > limits[:, 1])
        ):
            raise ValueError("Six finite joint angles within limits are required")
        # Edit a paused simulation pose, not a physical trajectory or robot command.
        b.data.qpos[b.qindices] = q
        b.data.qvel[b.vindices] = 0
        b.previous_target = b.measured()
        b._set_target(b.previous_target)
        mujoco.mj_forward(b.model, b.data)
        return self.capture(view_id, remember=remember)

    def scan_poses(self):
        base = np.asarray(self.settings["observation_q_rad"], float)
        limits = self.world.backend.model.jnt_range[self.world.backend.joints]
        result = []
        # Scan labels describe configured viewpoints, not inferred object relations.
        for view_id, offset in [("left", 0.25), ("right", -0.25), ("center", 0.0)]:
            q = base.copy()
            q[0] = np.clip(q[0] + offset, limits[0, 0], limits[0, 1])
            result.append({"view_id": view_id, "q": q.tolist()})
        return result

    def scan(self):
        """Batch sampling for API callers; the UI displays interpolated preview poses."""
        for pose in self.scan_poses():
            self.pose(pose["q"], pose["view_id"])
        return self.state()

    def _release_single(self):
        if self.worker is not None:
            self.worker.terminate()
            try:
                self.worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait(timeout=5)
            self.worker = None
        if self.worker_log is not None:
            self.worker_log.close()
            self.worker_log = None

    def release(self):
        self._release_single()
        self.task_memory.release()

    def multiframe_request(self, request, *, infer=False):
        if request.get("frame_id") != self.sequence:
            raise ValueError("画面已改变，请刷新后重新提交，避免使用旧图。")
        if request.get("memory_version") != self.task_memory.task.memory.version:
            raise ValueError("场景记忆已改变，请刷新后重新提交。")
        if infer:
            self._release_single()
            ready = threading.Event()
            result = self._with_tracking_capture(
                lambda: self.task_memory.infer(
                    request.get("instruction"), task_progress=request.get("task_progress"),
                    buffer_ready=ready), ready)
            if self.task_memory.tracking is not None and self.task_memory.tracking.epoch is not None:
                self.capture(self.view_id)
                prepared = self.task_memory.tracking.prepare_current(now_s=time.monotonic())
                result["tracking"] = self.task_memory.tracking.state()
                result["skill_preparation"] = {"status": prepared.status, "code": prepared.code}
            return {
                **result,
                "multiframe": self.task_memory.state(),
                "model_loaded": self.task_memory.qwen.loaded,
            }
        payload, folder = self.task_memory.prepare(
            request.get("instruction"), task_progress=request.get("task_progress")
        )
        return {
            "request_path": str(folder),
            "image_count": len(payload["annotated_image_files"]),
            "inference_executed": False,
            "robot_control_enabled": False,
            "multiframe": self.task_memory.state(),
        }

    def _with_tracking_capture(self, operation, ready):
        """Keep GL/rendering on this owner; the other thread only runs perception models."""
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="wrist-model") as executor:
            future = executor.submit(operation)
            next_capture = time.monotonic()
            while not future.done():
                if ready.wait(0.01) and time.monotonic() >= next_capture:
                    self.task_memory.buffer_tracking_sensors(self.world.observe(), self.view_id)
                    next_capture = time.monotonic() + 0.1
                else:
                    time.sleep(0.005)
            result = future.result()
        if self.task_memory.tracking is not None and self.task_memory.tracking.epoch is not None:
            # A final frame may arrive while the model thread drains its earlier batch.
            self.task_memory.tracking.replay_buffer()
        return result

    def confirm_tracking(self):
        ready = threading.Event()
        ready.set()
        result = self._with_tracking_capture(self.task_memory.confirm_tracking, ready)
        result["tracking"] = self.task_memory.tracking.state()
        prepared = self.task_memory.tracking.prepare_current(now_s=time.monotonic())
        result.update(status=prepared.status, code=prepared.code)
        return result

    def infer(self, request):
        if request.get("frame_id") != self.sequence:
            raise ValueError("画面已改变，请刷新后重新提交，避免使用旧图。")
        instruction = request.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 2000:
            raise ValueError("请输入1到2000字的指令")
        if self.worker is None or self.worker.poll() is not None:
            self.release()
            env = dict(os.environ)
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env.update(
                PYTHONPATH=str(ROOT / "src"),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            self.worker_log = (self.output / "model_worker.log").open("a")
            self.worker = subprocess.Popen(
                [
                    os.environ.get(
                        "QWEN_PYTHON",
                        str(ROOT / ".local/qwen-runtime/bin/python")
                        if (ROOT / ".local/qwen-runtime/bin/python").exists()
                        else sys.executable,
                    ),
                    str(ROOT / "scripts/qwen_camera_worker.py"),
                    str(ROOT / self.settings["model_config"]),
                ],
                cwd=ROOT,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.worker_log,
                text=True,
                bufsize=1,
            )
        start = time.perf_counter()
        # Only this wrist RGB path and text are sent; neither overview nor scene truth.
        self.worker.stdin.write(
            json.dumps(
                {"image": str(self.frame / "wrist_rgb.png"), "instruction": instruction},
                ensure_ascii=False,
            )
            + "\n"
        )
        self.worker.stdin.flush()
        readable, _, _ = select.select([self.worker.stdout], [], [], 90)
        if not readable:
            self.release()
            raise RuntimeError("模型请求超时，已释放该模型进程。")
        line = self.worker.stdout.readline()
        if not line:
            self.release()
            raise RuntimeError("模型进程退出，请查看model_worker.log。")
        result = json.loads(line)
        result.update(
            frame_id=self.sequence,
            frame_path=str(self.frame),
            instruction=instruction,
            wall_time_s=time.perf_counter() - start,
            model_loaded=True,
        )
        if result.get("error"):
            self.release()
            result["model_loaded"] = False
        self.requests += 1
        path = self.output / f"request_{self.requests:04d}.json"
        result["result_path"] = str(path)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    def close(self):
        self.release()
        self.world.close()


class CameraPanelHost:
    """Keep all renderer operations on one owner thread; HTTP connections stay independent."""

    def __init__(self, output, factory=CameraPanel):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wrist-simulation")
        self.lock = threading.Lock()
        self.busy = False
        try:
            self.panel = self.executor.submit(factory, output).result()
            self.cached_state = self.executor.submit(self.panel.state).result()
        except Exception:
            self.executor.shutdown(wait=True)
            raise

    def call(self, operation, *args):
        with self.lock:
            if self.busy:
                raise BlockingIOError("上一项任务还在运行，请稍候再试。")
            self.busy = True

        def run():
            try:
                return operation(*args)
            finally:
                try:
                    updated = self.panel.state()
                    with self.lock:
                        self.cached_state = updated
                finally:
                    with self.lock:
                        self.busy = False

        return self.executor.submit(run).result()

    def state(self):
        try:
            return {**self.call(self.panel.state), "busy": False}
        except BlockingIOError:
            with self.lock:
                return {**copy.deepcopy(self.cached_state), "busy": True}

    def close(self):
        try:
            self.executor.submit(self.panel.close).result()
        finally:
            self.executor.shutdown(wait=True)


def handler_for(host):
    panel = host.panel

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            # Browsers may preconnect without sending a request. Bound idle sockets too.
            self.connection.settimeout(10)

        def send_json(self, value, code=200):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                data = (ROOT / "assets/ui/wrist_qwen.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/state":
                self.send_json(host.state())
            else:
                self.send_json({"error": "Not found"}, 404)

        def perform(self, request):
            if self.path == "/api/capture":
                return panel.capture(request.get("view_id"))
            if self.path == "/api/reset":
                return panel.reset(request["seed"])
            if self.path == "/api/pose":
                return panel.pose(
                    request["q"], request.get("view_id"), remember=request.get("remember", True)
                )
            if self.path == "/api/multiframe/scan":
                return panel.scan()
            if self.path == "/api/multiframe/reset":
                panel.task_memory.reset()
                return panel.capture("center")
            if self.path == "/api/multiframe/demo":
                panel.task_memory.demo()
                return panel.state()
            if self.path in ("/api/multiframe/prepare", "/api/multiframe/infer"):
                return panel.multiframe_request(request, infer=self.path.endswith("/infer"))
            if self.path == "/api/multiframe/skill/prepare":
                return panel.task_memory.prepare_skill()
            if self.path == "/api/multiframe/skill/feedback":
                return panel.task_memory.skill_feedback(request["feedback"])
            if self.path == "/api/multiframe/tracking/confirm":
                return panel.confirm_tracking()
            if self.path in ("/api/multiframe/region", "/api/multiframe/sam"):
                if request.get("frame_id") != panel.sequence:
                    raise ValueError("画面已改变，请重新选点。")
                if self.path.endswith("/region"):
                    panel.task_memory.add_region(
                        request.get("region_id"),
                        request.get("point_xy"),
                        request.get("half_size_xy_m", [0.06, 0.04]),
                    )
                else:
                    panel.task_memory.segment(request.get("point_xy"))
                return panel.state()
            if self.path == "/api/infer":
                return panel.infer(request)
            if self.path == "/api/release":
                panel.release()
                return {"model_loaded": False}
            raise ValueError("接口不存在。")

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 20000:
                    raise ValueError("Invalid request length")
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("JSON required")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise TypeError("JSON object required")
                self.send_json(host.call(self.perform, request))
            except (BrokenPipeError, ConnectionResetError):
                return  # A disconnected tab must not affect the simulation owner.
            except BlockingIOError as exc:
                self.send_json({"error": str(exc)}, 409)
            except InvalidPrediction as exc:
                self.send_json(
                    {
                        "error": "模型返回的目标或放置区域未能与当前观测对应，重新判断后仍未通过。请再观察一次。",
                        "diagnostic": str(exc),
                    },
                    422,
                )
            except Exception as exc:  # noqa: BLE001 -- local UI error boundary.
                self.send_json({"error": str(exc)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/vlm/wrist_panel" / str(time.time_ns())
    )
    args = parser.parse_args()
    host = CameraPanelHost(args.output)
    panel = host.panel
    server = ThreadingHTTPServer(
        (panel.settings["host"], panel.settings["port"]), handler_for(host)
    )
    print(f"Wrist camera panel: http://{panel.settings['host']}:{server.server_port}/", flush=True)
    print(f"Output: {panel.output}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        host.close()


if __name__ == "__main__":
    main()
