"""Lazy isolated video-mask worker. No Torch import or robot control in this process."""

import json
import os
import select
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from piper_sim.calibration import ROOT

DEFAULT_TRACKING_METHOD = "edgetam"
TRACKING_METHODS = ("edgetam", "cutie", "sam2_small")


class VideoMaskClient:
    def __init__(self, output, *, method=DEFAULT_TRACKING_METHOD, timeout_s=90.0):
        if method not in TRACKING_METHODS:
            raise ValueError("Unknown video-mask method")
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Invalid worker timeout")
        self.output = Path(output).resolve() / uuid.uuid4().hex
        self.method, self.timeout_s = method, float(timeout_s)
        self.worker = self.log = None
        self.lock = threading.RLock()
        self.next_index = 0

    @staticmethod
    def _rgb(rgb):
        image = np.asarray(rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Tracking requires HWC uint8 RGB")
        return image

    def _response(self):
        """Read a complete bounded JSON line under one deadline, including partial writes."""
        deadline = time.monotonic() + self.timeout_s
        data = bytearray()
        while b"\n" not in data:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Video-mask worker timed out")
            ready, _, _ = select.select([self.worker.stdout], [], [], remaining)
            if not ready:
                raise TimeoutError("Video-mask worker timed out")
            block = os.read(self.worker.stdout.fileno(), 4096)
            if not block:
                raise RuntimeError(f"Video-mask worker exited; see {self.output / 'worker.log'}")
            data.extend(block)
            if len(data) > 65536:
                raise ValueError("Oversized video-mask worker response")
        line, tail = bytes(data).split(b"\n", 1)
        if tail.strip():
            raise ValueError("Unexpected extra video-mask worker response")
        result = json.loads(line)
        if not isinstance(result, dict):
            raise TypeError("Video-mask worker response must be an object")
        return result

    def _start(self):
        self.close()
        python = ROOT / ".local/tracking-bench/runtime/bin/python"
        if not python.is_file():
            raise RuntimeError("Missing isolated tracking runtime")
        self.output.mkdir(parents=True, exist_ok=True)
        self.log = (self.output / "worker.log").open("a")
        self.worker = subprocess.Popen(
            [str(python), str(ROOT / "scripts/video_mask_worker.py"), self.method],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "2",
                 "OPENBLAS_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
                 "TRANSFORMERS_OFFLINE": "1"},
        )

    def _call(self, rgb, *, mask=None, index):
        rgb = self._rgb(rgb)
        if self.worker is None or self.worker.poll() is not None:
            raise RuntimeError("Tracker worker unavailable; initialize a new epoch")
        request_id = uuid.uuid4().hex
        image_path, mask_path = self.output / "rgb.npy", self.output / "initial.npy"
        result_path = self.output / "result.npy"
        # A matching response without a newly written mask must never reuse an older frame.
        result_path.unlink(missing_ok=True)
        np.save(image_path, rgb, allow_pickle=False)
        if mask is not None:
            if np.asarray(mask).dtype != bool or np.asarray(mask).shape != rgb.shape[:2]:
                raise ValueError("Initial mask must be aligned boolean HxW")
            np.save(mask_path, mask, allow_pickle=False)
        request = {"request_id": request_id, "index": index, "image": str(image_path),
                   "initial_mask": str(mask_path) if mask is not None else None,
                   "output": str(result_path)}
        try:
            self.worker.stdin.write(json.dumps(request) + "\n")
            self.worker.stdin.flush()
            response = self._response()
            if response.get("request_id") != request_id or response.get("index") != index:
                raise ValueError("Tracking response does not match the requested frame")
            if "error" in response:
                raise RuntimeError(response["error"])
            result = np.load(result_path, allow_pickle=False)
            if result.dtype != bool or result.shape != rgb.shape[:2]:
                raise ValueError("Invalid current-frame tracker mask")
            self.next_index = index + 1
            return result
        except Exception:
            self.close()
            raise

    def initialize(self, rgb, mask):
        rgb = self._rgb(rgb)
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != rgb.shape[:2] or not mask.any():
            raise ValueError("Initial mask must be nonempty aligned boolean HxW")
        with self.lock:
            # Reset means a new process and model memory, never reuse a tracker ID silently.
            self._start()
            try:
                return self._call(rgb, mask=mask, index=0)
            except Exception:
                self.close()
                raise

    def step(self, rgb, index):
        with self.lock:
            if index != self.next_index:
                raise ValueError("Tracking frames must be consecutive")
            return self._call(rgb, index=index)

    def close(self):
        with self.lock:
            if self.worker is not None:
                if self.worker.poll() is None:
                    self.worker.terminate()
                    try:
                        self.worker.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.worker.kill()
                        self.worker.wait(timeout=5)
                self.worker.stdin.close()
                self.worker.stdout.close()
                self.worker = None
            if self.log is not None:
                self.log.close()
                self.log = None
