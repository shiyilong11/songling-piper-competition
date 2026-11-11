"""Colour/contour RGB-D detections; no access to simulator object states."""

from dataclasses import dataclass

import cv2
import numpy as np

from piper_sim.calibration import camera_to_base


@dataclass
class Detection:
    color: str
    position: np.ndarray
    yaw: float
    pixels: int


def detect_blocks(obs, camera_cfg, table_top):
    hsv = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2HSV)
    masks = {
        "red": cv2.inRange(hsv, (0, 100, 55), (10, 255, 255))
        | cv2.inRange(hsv, (170, 100, 55), (180, 255, 255)),
        "green": cv2.inRange(hsv, (35, 90, 45), (85, 255, 255)),
        "blue": cv2.inRange(hsv, (95, 90, 45), (135, 255, 255)),
    }
    results = []
    k = obs.intrinsics
    for color, mask in masks.items():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for index in range(1, n):
            if stats[index, cv2.CC_STAT_AREA] < 60:
                continue
            inside = cv2.erode((labels == index).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            v, u = np.where(inside & np.isfinite(obs.depth_m))
            if len(u) < 20:
                continue
            z = obs.depth_m[v, u]
            points = np.stack([(u - k["cx"]) * z / k["fx"], (v - k["cy"]) * z / k["fy"], z], axis=1)
            base = camera_to_base(points, obs.T_base_end, camera_cfg)
            # Recover the horizontal top face, excluding side faces of a tilted view.
            top = np.percentile(base[:, 2], 90)
            base = base[base[:, 2] > top - 0.003]
            if len(base) < 15 or top < table_top + 0.01:
                continue
            xy = np.median(base[:, :2], axis=0)
            rect = cv2.minAreaRect(base[:, :2].astype(np.float32))
            corners = cv2.boxPoints(rect)
            edge = corners[1] - corners[0]
            yaw = float(np.arctan2(edge[1], edge[0]))
            # Squares have equivalent grasps every 90 degrees; select nearest upright yaw.
            yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
            results.append(Detection(color, np.r_[xy, top], yaw, len(base)))
    return results
