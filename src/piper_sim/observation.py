"""Build the same observation from decoded ROS camera/state messages, without importing ROS."""

from dataclasses import dataclass

import numpy as np

from piper_sim.calibration import validate_transform
from piper_sim.control import canonical_state


@dataclass
class Observation:
    timestamp_s: float
    joint_position_rad: np.ndarray
    joint_velocity_rad_s: np.ndarray
    gripper_width_m: float
    T_base_end: np.ndarray
    rgb: np.ndarray | None = None
    depth_m: np.ndarray | None = None
    intrinsics: dict | None = None
    # Optional calibrated pose for a camera fixed to the bench.
    T_base_camera: np.ndarray | None = None

    @property
    def depth_valid_mask(self):
        """The policy may use validity, never infer that missing depth equals zero metres."""
        return None if self.depth_m is None else np.isfinite(self.depth_m) & (self.depth_m > 0)


def observation_from_sensors(
    *,
    rgb,
    depth,
    depth_encoding,
    intrinsics,
    names,
    positions,
    velocities,
    T_base_end,
    timestamps_s,
    camera_cfg,
    interface_cfg,
):
    """Caller aligns depth to colour, converts RGB to rgb8 and aligns end coordinate frames.

    timestamps_s must include rgb, depth, joints, end_pose in one common clock.
    No transform guess is made for the firmware end frame.
    """
    if set(timestamps_s) != {"rgb", "depth", "joints", "end_pose"}:
        raise ValueError("Four sensor timestamps required")
    stamps = np.asarray(list(timestamps_s.values()), dtype=float)
    if (
        not np.isfinite(stamps).all()
        or np.ptp(stamps) > interface_cfg["observation"]["max_sensor_skew_s"]
    ):
        raise ValueError("Sensor timestamps are not synchronized")
    cam = camera_cfg["camera"]
    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    if rgb.shape != (cam["height"], cam["width"], 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected configured HWC uint8 RGB (not BGR)")
    if depth.shape != rgb.shape[:2]:
        raise ValueError("Depth must be aligned to RGB at the same resolution")
    if depth_encoding == "16UC1":
        if depth.dtype != np.uint16:
            raise ValueError("16UC1 depth requires uint16 data")
        metric = depth.astype(np.float32) * cam["depth"]["meters_per_unit"]
    elif depth_encoding == "32FC1":
        metric = depth.astype(np.float32)
    else:
        raise ValueError("Expected 16UC1 raw units or 32FC1 metres")
    metric[~np.isfinite(metric) | (metric <= 0)] = np.nan
    k = np.asarray([intrinsics[n] for n in ["fx", "fy", "cx", "cy"]])
    if not np.isfinite(k).all() or (k[:2] <= 0).any():
        raise ValueError("Invalid rectified camera intrinsics")
    q, dq = canonical_state(names, positions, velocities, interface_cfg)
    return Observation(
        float(timestamps_s["rgb"]),
        q[:6],
        dq,
        float(q[6]),
        validate_transform(T_base_end),
        rgb.copy(),
        metric,
        intrinsics.copy(),
    )
