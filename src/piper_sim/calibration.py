"""Explicit optical-frame transforms shared by simulation and future real perception."""

from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config/eye_in_hand.yaml"
# OpenCV optical (+X right,+Y down,+Z forward) to MuJoCo camera (+X right,+Y up,-Z forward).
R_OPTICAL_MUJOCO = np.diag([1.0, -1.0, -1.0])


def validate_transform(value):
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("T_end_camera must be a finite 4x4 matrix")
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("Transform last row must be [0,0,0,1]")
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(r), 1.0, atol=1e-5
    ):
        raise ValueError("Rotation must be orthonormal with determinant +1")
    return t


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open() as f:
        cfg = yaml.safe_load(f)
    if cfg["schema_version"] != 1:
        raise ValueError("Unsupported camera config schema")
    cam = cfg["camera"]
    for key in ["width", "height"]:
        if not isinstance(cam[key], int) or not 1 <= cam[key] <= 4096:
            raise ValueError(f"{key} must be an integer in [1,4096]")
    k = cam["intrinsics"]
    if not np.isfinite([k[x] for x in ["fx", "fy", "cx", "cy"]]).all():
        raise ValueError("Intrinsics must be finite")
    if k["fx"] <= 0 or k["fy"] <= 0:
        raise ValueError("Focal lengths must be positive")
    if cam["depth"]["meters_per_unit"] <= 0:
        raise ValueError("Depth scale must be positive")
    validate_transform(cfg["hand_eye"]["T_end_camera"])
    if not np.isfinite(cam["distortion"]["coefficients"]).all():
        raise ValueError("Distortion coefficients must be finite")
    q = np.asarray(cfg["simulation"]["initial_qpos"])
    if q.shape != (8,) or not np.isfinite(q).all():
        raise ValueError("initial_qpos must contain 8 finite values")
    return cfg


def transform_points(transform, points):
    """Apply T_destination_source to (...,3) source points; units must match."""
    t = validate_transform(transform)
    p = np.asarray(points, dtype=float)
    if p.shape[-1] != 3 or not np.isfinite(p).all():
        raise ValueError("Points must be finite and have shape (...,3)")
    return p @ t[:3, :3].T + t[:3, 3]


def pixel_to_camera(u, v, depth_m, intrinsics):
    """Backproject rectified pixels with optical-axis Z depth in metres (not ray range)."""
    if not np.isfinite([u, v, depth_m]).all() or depth_m <= 0:
        raise ValueError("Pixel and depth must be finite, with depth > 0 metres")
    k = intrinsics
    return np.array([(u - k["cx"]) * depth_m / k["fx"], (v - k["cy"]) * depth_m / k["fy"], depth_m])


def camera_to_base(points_camera, T_base_end, cfg):
    """Use the robot end pose captured at the same time as the image."""
    t = validate_transform(T_base_end) @ validate_transform(cfg["hand_eye"]["T_end_camera"])
    return transform_points(t, points_camera)


def body_pose(model, data, name):
    """MuJoCo world-from-body pose; world coincides with robot base in this scene."""
    body = model.body(name).id
    t = np.eye(4)
    t[:3, :3] = data.xmat[body].reshape(3, 3)
    t[:3, 3] = data.xpos[body]
    return t


def simulated_depth_outputs(depth_m, camera):
    """Ideal registered Z depth, range mask and integer units; invalid is NaN/0."""
    depth = np.asarray(depth_m, dtype=np.float32)
    limits = camera["depth"]
    valid = np.isfinite(depth) & (depth >= limits["min_valid_m"]) & (depth <= limits["max_valid_m"])
    units = np.rint(np.where(valid, depth, 0) / limits["meters_per_unit"])
    valid &= units <= np.iinfo(np.uint16).max
    raw = np.where(valid, units, 0).astype(np.uint16)
    return np.where(valid, depth, np.nan).astype(np.float32), raw, valid
