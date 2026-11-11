"""Local, read-only dashboard for actual wrist-camera observations."""

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from piper_sim.calibration import ROOT


def jpeg(rgb):
    ok, image = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82]
    )
    if not ok:
        raise ValueError("Cannot encode dashboard image")
    return "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")


def render_segmentation(observation, candidates):
    rgb = observation.rgb.copy()
    regions = []
    palette = ((57, 211, 174), (250, 173, 67), (104, 167, 250), (244, 116, 151))
    for number, (label, zone, mask) in enumerate(candidates, 1):
        if mask.shape != rgb.shape[:2] or not mask.any():
            continue
        color = palette[(number - 1) % len(palette)]
        rgb[mask] = np.rint(0.55 * rgb[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
        outlines, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(rgb, outlines, -1, color, 2)
        y, x = np.where(mask)
        origin = (int(x.min()) + 2, max(17, int(y.min()) - 3))
        cv2.putText(rgb, str(number), origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 24, 32), 4)
        cv2.putText(rgb, str(number), origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        regions.append({"number": number, "label": label, "destination": zone})
    return {
        "stamp_s": float(observation.timestamp_s),
        "image": jpeg(rgb),
        "regions": regions,
    }


class FieldDashboard:
    def __init__(self, port, depth_range):
        self.depth_range = tuple(depth_range)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.camera_thread = None
        self.camera = self.segmentation = self.camera_updated = None
        self.camera_error = None
        self.task = {}
        page = (ROOT / "assets/ui/field_dashboard.html").read_bytes()
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/":
                    body, mime = page, "text/html; charset=utf-8"
                elif self.path == "/api/state":
                    body = json.dumps(dashboard.snapshot(), ensure_ascii=False).encode()
                    mime = "application/json; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/"

    def snapshot(self):
        with self.lock:
            return {
                "camera": self.camera,
                "camera_age_s": None
                if self.camera_updated is None
                else round(time.monotonic() - self.camera_updated, 2),
                "camera_error": self.camera_error,
                "depth_range_m": self.depth_range,
                "segmentation": self.segmentation,
                **self.task,
            }

    def attach_camera(self, capture):
        self.camera_thread = threading.Thread(
            target=self._camera_loop, args=(capture,), daemon=True
        )
        self.camera_thread.start()

    def _camera_loop(self, capture):
        previous = None
        while not self.stop.is_set():
            try:
                pair = capture()
                if pair is not None and pair[0] != previous:
                    self.publish_camera(*pair[1:])
                    previous = pair[0]
            except (ValueError, RuntimeError, OSError, cv2.error) as exc:
                with self.lock:
                    self.camera_error = str(exc)
            self.stop.wait(0.2)

    def publish_camera(self, stamp, rgb, depth):
        valid = np.isfinite(depth) & (depth > 0)
        near, far = self.depth_range
        values = np.where(valid, depth, near)
        gray = np.rint(np.clip((values - near) / (far - near), 0, 1) * 255).astype(np.uint8)
        colored = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        colored[~valid] = 0
        frame = {
            "stamp_s": float(stamp),
            "rgb": jpeg(rgb),
            "depth": jpeg(colored),
            "valid_fraction": float(valid.mean()),
        }
        with self.lock:
            self.camera = frame
            self.camera_updated = time.monotonic()
            self.camera_error = None

    def publish_segmentation(self, observation, candidates):
        segmentation = render_segmentation(observation, candidates)
        with self.lock:
            self.segmentation = segmentation

    def publish_task(self, state):
        with self.lock:
            self.task = state

    def close(self):
        self.stop.set()
        if self.camera_thread is not None:
            self.camera_thread.join(timeout=1)
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=1)
