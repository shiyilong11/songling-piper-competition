"""Synthetic two-view sensor fixture, explicitly not a model accuracy experiment."""

import json

import numpy as np
import yaml
from PIL import Image, ImageDraw

from piper_sim.multiframe_io import write_frame
from piper_sim.observation import Observation
from piper_sim.scene_memory import load_memory_config
from piper_sim.scene_observation import (
    CalibrationEstimate,
    InstanceMask,
    PlacementRegion,
    Provenance,
)

FIXTURE_SOURCE = Provenance(
    "manual_config", "synthetic unit fixture; supplied test masks, not SAM output"
)


def make_demo_frame(target_y, camera_y, timestamp_s, *, view_id="left"):
    height, width, focal = 240, 320, 500.0
    z = 0.8 - 0.17
    center = (160, round(120 + (camera_y - target_y) * focal / z))
    ys, xs = np.indices((height, width))
    mask = (xs - center[0]) ** 2 + (ys - center[1]) ** 2 <= 24**2
    rgb = Image.new("RGB", (width, height), "#a4a096")
    draw = ImageDraw.Draw(rgb)
    x, y = center
    draw.ellipse((x - 24, y - 24, x + 24, y + 24), fill="#e88828", outline="#995117", width=2)
    draw.polygon([(x, y - 27), (x - 13, y - 12), (x, y - 18), (x + 13, y - 12)], fill="#42703a")
    depth = np.full((height, width), 0.68, dtype=np.float32)
    depth[mask] = z
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[:3, 3] = [0.4, camera_y, 0.8]
    obs = Observation(
        timestamp_s,
        np.zeros(6),
        np.zeros(6),
        0.06,
        pose,
        np.asarray(rgb),
        depth,
        {"fx": focal, "fy": focal, "cx": 160.0, "cy": 120.0},
    )
    calibration = CalibrationEstimate(np.eye(4), FIXTURE_SOURCE)
    detection = InstanceMask("frame_local", mask, timestamp_s, 0.8, FIXTURE_SOURCE)
    region_pose = np.eye(4)
    region_pose[:3, 3] = [0.55, -0.3, 0.12]
    region = PlacementRegion("blue_tray", region_pose, np.array([0.06, 0.04]), FIXTURE_SOURCE)
    return {
        "observation": obs,
        "calibration": calibration,
        "detections": [detection],
        "view_id": view_id,
        "regions": [region],
    }


def create_demo_bundle(output):
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    for index, values in enumerate([(0.14, 0.14, 0.0, "left"), (-0.14, -0.14, 1.0, "right")]):
        frame = make_demo_frame(*values[:3], view_id=values[3])
        path = write_frame(output / f"view_{index + 1}", **frame)
        paths.append(str(path.relative_to(output)))
    cfg = load_memory_config()
    cfg["required_views"] = ["left", "right"]
    (output / "memory.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    manifest = {
        "schema_version": 1,
        "fixture_only": True,
        "description": "Synthetic RGB-D: two separate views, one orange fruit per frame. No SAM or Qwen inference has run.",
        "instruction": "将最左边的柿子放入蓝色托盘。左右按桌面固定坐标，+Y为左。",
        "memory_config": "memory.yaml",
        "frame_files": paths,
    }
    (output / "bundle.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return output / "bundle.json"
