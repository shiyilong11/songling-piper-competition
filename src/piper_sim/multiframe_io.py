"""Portable, no-pickle recording of public multi-view sensor inputs."""

import json
from pathlib import Path

import numpy as np

from piper_sim.observation import Observation
from piper_sim.scene_observation import (
    CalibrationEstimate,
    InstanceMask,
    PlacementRegion,
    Provenance,
)


def write_frame(folder, observation, calibration, detections, view_id, regions):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    calibration.source.require_public()
    detections, regions = list(detections), list(regions)
    for detection in detections:
        detection.source.require_public()
    if observation.rgb is None or observation.depth_m is None:
        raise ValueError("Recorded frame requires synchronized RGB and depth")
    np.savez_compressed(
        folder / "sensors.npz",
        rgb=observation.rgb,
        depth_m=observation.depth_m,
        joint_position_rad=observation.joint_position_rad,
        joint_velocity_rad_s=observation.joint_velocity_rad_s,
        T_base_end=observation.T_base_end,
        masks=np.stack([d.mask for d in detections])
        if detections
        else np.empty((0, *observation.depth_m.shape), dtype=bool),
    )
    metadata = {
        "schema_version": 1,
        "sensor_file": "sensors.npz",
        "view_id": view_id,
        "timestamp_s": observation.timestamp_s,
        "gripper_width_m": observation.gripper_width_m,
        "intrinsics": observation.intrinsics,
        "calibration": {
            "T_end_camera": calibration.T_end_camera.tolist(),
            "source": calibration.source.as_dict(),
        },
        "detections": [
            {
                "instance_key": d.instance_key,
                "timestamp_s": d.timestamp_s,
                "confidence": d.confidence,
                "source": d.source.as_dict(),
                "label": d.label,
            }
            for d in detections
        ],
        "regions": [r.as_dict() for r in regions],
    }
    path = folder / "frame.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    return path


def read_frame(path):
    path = Path(path)
    data = json.loads(path.read_text())
    if data["schema_version"] != 1:
        raise ValueError("Unsupported recorded frame schema")
    sensor = (path.parent / data["sensor_file"]).resolve()
    if not sensor.is_relative_to(path.parent.resolve()):
        raise ValueError("Sensor file must be inside the recorded frame folder")
    with np.load(sensor, allow_pickle=False) as arrays:
        obs = Observation(
            data["timestamp_s"],
            arrays["joint_position_rad"].copy(),
            arrays["joint_velocity_rad_s"].copy(),
            data["gripper_width_m"],
            arrays["T_base_end"].copy(),
            arrays["rgb"].copy(),
            arrays["depth_m"].copy(),
            data["intrinsics"],
        )
        masks = arrays["masks"].copy()
    if len(masks) != len(data["detections"]):
        raise ValueError("Detection/mask count mismatch")
    calibration = CalibrationEstimate(
        np.asarray(data["calibration"]["T_end_camera"], float),
        Provenance(**data["calibration"]["source"]),
    )
    detections = [
        InstanceMask(
            d["instance_key"],
            mask,
            d["timestamp_s"],
            d["confidence"],
            Provenance(**d["source"]),
            d["label"],
        )
        for d, mask in zip(data["detections"], masks)
    ]
    regions = [
        PlacementRegion(
            r["region_id"],
            np.asarray(r["T_base_region"], float),
            np.asarray(r["half_size_xy_m"], float),
            Provenance(**r["source"]),
            r["kind"],
            r["timestamp_s"],
        )
        for r in data["regions"]
    ]
    return {
        "observation": obs,
        "calibration": calibration,
        "detections": detections,
        "view_id": data["view_id"],
        "regions": regions,
    }
