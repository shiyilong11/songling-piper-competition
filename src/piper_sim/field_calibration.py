"""Calibrated ROS RGB-D input for the two competition controllers."""

from pathlib import Path

import cv2
import numpy as np
import yaml

from piper_sim.calibration import validate_transform
from piper_sim.observation import Observation


class FieldCalibration:
    def __init__(self, path):
        cfg = yaml.safe_load(Path(path).read_text())
        if not isinstance(cfg, dict) or cfg.get("schema_version") != 1:
            raise ValueError("Expected field_calibration.yaml schema_version 1")
        camera, hand_eye = cfg["camera"], cfg["hand_eye"]
        self.width, self.height = camera["width"], camera["height"]
        if any(type(n) is not int or not 16 <= n <= 4096 for n in (self.width, self.height)):
            raise ValueError("Fill the calibrated camera width and height")
        self.optical_frame = camera["optical_frame"]
        if not isinstance(self.optical_frame, str) or not self.optical_frame.strip():
            raise ValueError("Fill camera.optical_frame from the registered stream")
        k = np.asarray([camera["intrinsics"][name] for name in ("fx", "fy", "cx", "cy")], float)
        if not np.isfinite(k).all() or min(k[:2]) <= 0:
            raise ValueError("Fill calibrated fx, fy, cx, cy for the configured resolution")
        self.intrinsics = dict(zip(("fx", "fy", "cx", "cy"), k.tolist(), strict=True))
        depth = camera["depth"]
        try:
            self.depth_scale = float(depth["meters_per_unit"])
            self.depth_min = float(depth["min_valid_m"])
            self.depth_max = float(depth["max_valid_m"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Fill camera.depth meters_per_unit, min_valid_m and max_valid_m for the bench stream"
            ) from exc
        if not 0 < self.depth_scale <= 0.1 or not 0 < self.depth_min < self.depth_max <= 3:
            raise ValueError("Invalid registered depth scale or working range")
        if hand_eye["parent_frame"] not in ("link6", "base_link"):
            raise ValueError("hand_eye.parent_frame must be link6 for wrist or base_link for fixed")
        self.fixed = hand_eye["parent_frame"] == "base_link"
        self.transform = validate_transform(hand_eye["T_parent_camera_optical"])
        if np.linalg.norm(self.transform[:3, 3]) > 2.0:
            raise ValueError("Hand-eye translation must be in metres")
        distortion = np.asarray(camera.get("distortion_coefficients", []), float)
        if (
            distortion.ndim != 1
            or distortion.size not in (0, 4, 5, 8, 12, 14)
            or not np.isfinite(distortion).all()
        ):
            raise ValueError("Expected OpenCV Brown-Conrady coefficients or [] for rectified input")
        self.maps = None
        if np.any(distortion):
            fx, fy, cx, cy = k
            matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            self.maps = cv2.initUndistortRectifyMap(
                matrix, distortion, None, matrix, (self.width, self.height), cv2.CV_32FC1
            )

    def controller_config(self):
        return {
            "hand_eye": {"T_end_camera": np.eye(4) if self.fixed else self.transform},
            "camera": {"depth": {"min_valid_m": self.depth_min}},
        }

    def _image_array(self, msg, dtype, channels):
        if (msg.width, msg.height) != (self.width, self.height):
            raise ValueError("ROS image resolution differs from the calibration")
        if msg.header.frame_id != self.optical_frame:
            raise ValueError(
                "RGB and registered depth must name the calibrated colour optical frame"
            )
        dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
        pixel = dtype.itemsize * channels
        buffer = memoryview(msg.data)
        if msg.step < self.width * pixel or buffer.nbytes < self.height * msg.step:
            raise ValueError("ROS image buffer/row stride is invalid")
        if channels == 1:
            return np.ndarray((self.height, self.width), dtype, buffer, strides=(msg.step, pixel))
        return np.ndarray(
            (self.height, self.width, channels),
            dtype,
            buffer,
            strides=(msg.step, pixel, dtype.itemsize),
        )

    def aligned_images(self, rgb_msg, depth_msg):
        if rgb_msg.encoding not in ("rgb8", "bgr8"):
            raise ValueError("Expected ROS rgb8 or bgr8 image")
        rgb = self._image_array(rgb_msg, np.uint8, 3)
        rgb = (rgb if rgb_msg.encoding == "rgb8" else rgb[..., ::-1]).copy()
        if depth_msg.encoding in ("16UC1", "mono16"):
            depth = self._image_array(depth_msg, np.uint16, 1).astype(np.float32)
            depth *= self.depth_scale
        elif depth_msg.encoding == "32FC1":
            depth = self._image_array(depth_msg, np.float32, 1).astype(np.float32)
        else:
            raise ValueError("Expected aligned ROS 16UC1, mono16 or metric 32FC1 depth")
        depth[(depth < self.depth_min) | (depth > self.depth_max) | ~np.isfinite(depth)] = np.nan
        if self.maps is not None:
            rgb = cv2.remap(rgb, *self.maps, cv2.INTER_LINEAR)
            depth = cv2.remap(depth, *self.maps, cv2.INTER_NEAREST, borderValue=float("nan"))
        return rgb, depth

    def observation(self, rgb_msg, depth_msg, state, velocity, stamp, end_pose):
        rgb, depth = self.aligned_images(rgb_msg, depth_msg)
        return Observation(
            stamp,
            state[:6].copy(),
            velocity.copy(),
            float(state[6]),
            end_pose,
            rgb,
            depth,
            self.intrinsics.copy(),
            self.transform if self.fixed else None,
        )
