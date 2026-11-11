"""Prompted SAM 2 masks. No robot control, simulator truth or semantic labels."""

import json
import os
import select
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import yaml

from piper_sim.calibration import ROOT
from piper_sim.scene_observation import InstanceMask, Provenance

DEFAULT_CONFIG = ROOT / "config/segmentation/sam2.yaml"


def choose_mask(masks, scores, point_xy, config):
    """Reject empty/background masks and require the selected positive point."""
    masks, scores = np.asarray(masks, dtype=bool), np.asarray(scores, dtype=float)
    if masks.ndim != 3 or scores.shape != (len(masks),):
        raise ValueError("SAM returned invalid mask dimensions")
    h, w = masks.shape[1:]
    point = np.asarray(point_xy, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("SAM point must contain two finite pixel coordinates")
    if not (0 <= point[0] < w and 0 <= point[1] < h):
        raise ValueError("SAM point is outside the image")
    x, y = np.minimum(np.rint(point).astype(int), [w - 1, h - 1])
    areas = masks.sum(axis=(1, 2))
    valid = (
        np.isfinite(scores)
        & (scores >= config["min_predicted_iou"])
        & (areas >= config["min_mask_pixels"])
        & (areas <= h * w * config["max_mask_fraction"])
        & masks[:, y, x]
    )
    if not valid.any():
        raise ValueError("SAM未返回包含选点且通过面积/质量检查的mask；请重新选点或换视角。")
    index = int(np.argmax(np.where(valid, scores, -np.inf)))
    return masks[index], {
        "predicted_iou": float(scores[index]),
        "candidate_scores": [float(s) if np.isfinite(s) else None for s in scores],
        "mask_pixels": int(areas[index]),
        "selected_index": index,
    }


class Sam2Session:
    """Loaded only in the isolated SAM process; full precision CPU by default."""

    def __init__(self, config):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.cfg = config
        self.torch = torch
        torch.set_num_threads(int(config["cpu_threads"]))
        checkpoint = ROOT / config["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing SAM checkpoint: {checkpoint}")
        self.predictor = SAM2ImagePredictor(
            build_sam2(
                config["model_config"],
                str(checkpoint),
                device=config["device"],
                apply_postprocessing=False,
            )
        )
        self._encoded_rgb = None

    def predict(self, rgb, point_xy, box_xyxy=None):
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("SAM requires HWC uint8 RGB")
        h, w = rgb.shape[:2]
        point = np.asarray(point_xy, dtype=float)
        if (
            point.shape != (2,)
            or not np.isfinite(point).all()
            or not (0 <= point[0] < w and 0 <= point[1] < h)
        ):
            raise ValueError("SAM point is outside the image or invalid")
        box = None
        if box_xyxy is not None:
            box = np.asarray(box_xyxy, dtype=float)
            if (
                box.shape != (4,)
                or not np.isfinite(box).all()
                or not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h)
            ):
                raise ValueError("Invalid SAM box_xyxy")
        start = time.perf_counter()
        cached = getattr(self, "_encoded_rgb", None)
        embedding_reused = cached is not None and np.array_equal(rgb, cached)
        with self.torch.inference_mode():
            if not embedding_reused:
                # Content equality, never a reused frame name or timestamp,
                # controls this one-image cache. Prompts still run independently.
                self._encoded_rgb = None
                self.predictor.set_image(rgb)
                self._encoded_rgb = rgb.copy()
            masks, scores, _ = self.predictor.predict(
                point_coords=point[None],
                point_labels=np.ones(1, dtype=np.int32),
                box=box,
                multimask_output=True,
            )
        mask, metadata = choose_mask(masks, scores, point, self.cfg)
        return mask, {
            **metadata,
            "inference_s": time.perf_counter() - start,
            "model": self.cfg["model"],
            "device": self.cfg["device"],
            "point_xy": point.tolist(),
            "tracking": False,
            "image_embedding_reused": embedding_reused,
        }


class Sam2Client:
    """Persistent JSON-lines worker; never changes the main training environment."""

    def __init__(self, config=DEFAULT_CONFIG):
        self.config = Path(config).resolve()
        self.cfg = yaml.safe_load(self.config.read_text())
        self.worker = self.log = None

    def _start(self, output):
        if self.worker is not None and self.worker.poll() is None:
            return
        self.close()
        python = Path(
            os.environ.get("PIPER_SAM2_PYTHON", str(ROOT / ".local/sam2-runtime/bin/python"))
        ).absolute()
        if not python.is_file():
            raise RuntimeError("SAM独立环境未安装，请执行 scripts/setup_sam.sh")
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                str(path) for path in (ROOT / "src", ROOT / "vendor") if path.is_dir()
            ),
            "OMP_NUM_THREADS": str(self.cfg["cpu_threads"]),
            "OPENBLAS_NUM_THREADS": "1",
        }
        if self.cfg["device"] == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        self.log = (output / "sam_worker.log").open("a")
        self.worker = subprocess.Popen(
            [str(python), str(ROOT / "scripts/sam_worker.py"), str(self.config)],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )

    def segment(self, rgb, point_xy, timestamp_s, frame_key, output, box_xyxy=None):
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        if not np.isfinite(timestamp_s):
            raise ValueError("SAM requires a finite sensor timestamp")
        request_id = uuid.uuid4().hex
        image_path, mask_path = output / "sam_rgb.npy", output / "sam_mask.npy"
        np.save(image_path, rgb, allow_pickle=False)
        self._start(output)
        request = {
            "request_id": request_id,
            "image": str(image_path),
            "mask": str(mask_path),
            "point_xy": list(point_xy),
            "box_xyxy": box_xyxy,
        }
        try:
            self.worker.stdin.write(json.dumps(request) + "\n")
            self.worker.stdin.flush()
            ready, _, _ = select.select([self.worker.stdout], [], [], self.cfg["timeout_s"])
            if not ready:
                raise TimeoutError("SAM分割超时；已关闭本次工作进程。")
            line = self.worker.stdout.readline()
            if not line:
                raise RuntimeError(f"SAM进程退出，请查看 {output / 'sam_worker.log'}")
            result = json.loads(line)
            if result.get("request_id") != request_id:
                raise RuntimeError("SAM response does not match the current frame")
            if "error" in result:
                raise ValueError(result["error"])
            mask = np.load(mask_path, allow_pickle=False)
            if mask.dtype != bool or mask.shape != np.asarray(rgb).shape[:2]:
                raise ValueError("SAM output mask does not match the RGB frame")
        except Exception:
            self.close()
            raise
        metadata = {**result, "frame_key": str(frame_key), "timestamp_s": float(timestamp_s)}
        (output / "sam_result.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        instance = InstanceMask(
            instance_key=f"sam2:{frame_key}",
            mask=mask,
            timestamp_s=float(timestamp_s),
            confidence=float(np.clip(result["predicted_iou"], 0, 1)),
            source=Provenance(
                "sensor_estimate", "SAM2.1 RGB point prompt; IoU heuristic; frame-local ID"
            ),
        )
        return instance, metadata

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
