"""Competition perception and verification using camera measurements only.

No world, object catalog, instance IDs, contacts or evaluator is accepted here.
All coordinates are estimates in the robot base frame. Calibration remains required.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from piper_sim.calibration import transform_points, validate_transform
from piper_sim.demo_skill import TargetEstimate
from piper_sim.l1_rgb import L1RGBDetector, color_masks
from piper_sim.scene_observation import _estimate_plane


@dataclass
class ZoneEstimate:
    color: str
    corners: np.ndarray
    floor_m: float
    timestamp_s: float

    @property
    def center(self):
        return self.corners.mean(axis=0)

    def contains(self, xy, margin=0.008):
        return all(
            cv2.pointPolygonTest(self.corners.astype(np.float32), tuple(map(float, p)), True)
            >= margin
            for p in np.atleast_2d(xy)
        )

    def as_dict(self):
        return {
            "color": self.color,
            "corners_base_xy_m": self.corners.tolist(),
            "floor_m": self.floor_m,
            "timestamp_s": self.timestamp_s,
            "source": "RGB planar mask + registered depth; full square size gate",
        }


def backproject(observation, T_end_camera):
    rgb, z, k = observation.rgb, observation.depth_m, observation.intrinsics
    if rgb is None or z is None or z.shape != rgb.shape[:2] or z.dtype.kind != "f":
        raise ValueError("Aligned RGB8 and floating-point metric depth required")
    if rgb.dtype != np.uint8 or rgb.shape[-1] != 3:
        raise ValueError("RGB8 required")
    if (
        not np.isfinite([k[n] for n in ("fx", "fy", "cx", "cy")]).all()
        or min(k["fx"], k["fy"]) <= 0
    ):
        raise ValueError("Invalid intrinsics")
    valid = np.isfinite(z) & (z > 0.05) & (z < 3)
    v, u = np.indices(z.shape)
    points = np.stack(((u - k["cx"]) * z / k["fx"], (v - k["cy"]) * z / k["fy"], z), -1)
    base = np.full_like(points, np.nan)
    fixed = getattr(observation, "T_base_camera", None)
    T = (
        validate_transform(fixed)
        if fixed is not None
        else validate_transform(observation.T_base_end) @ validate_transform(T_end_camera)
    )
    base[valid] = transform_points(T, points[valid])
    return base, valid


def components(mask, minimum=40):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return [labels == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= minimum]


def separate_zone_from_table(patch, saturation):
    """Split a merged planar component only when its colours have clear contrast."""
    samples = saturation[patch]
    threshold, _ = cv2.threshold(samples, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    high = samples > threshold
    if not high.any() or high.all():
        return []
    # Ignore splits caused by small variations in the wood grain or lighting.
    if np.median(samples[high]) - np.median(samples[~high]) < 25:
        return []
    return components(patch & (saturation > threshold), 100)


class SensorScene:
    def __init__(self, calibration, plane_config, target_edge_m=0.30):
        self.calibration = validate_transform(calibration)
        self.plane_config = plane_config
        self.detector = L1RGBDetector()
        # Public physical-size prior, not simulator target coordinates.
        if not np.isfinite(target_edge_m) or target_edge_m <= 0:
            raise ValueError("Target edge must be a positive metre value")
        self.target_edge_m = float(target_edge_m)
        self.table = None
        self.zones = {}
        self.partial_zones = {}

    def observe(self, obs):
        cloud, valid = backproject(obs, self.calibration)
        # This is a configured robot workspace, not a hidden table or target extent.
        valid &= (cloud[..., 0] > 0.08) & (cloud[..., 0] < 0.80)
        valid &= (np.abs(cloud[..., 1]) < 0.55) & (cloud[..., 2] > -0.03) & (cloud[..., 2] < 0.45)
        if self.table is None:
            plane = _estimate_plane(cloud[valid][::4], self.plane_config)
            if plane is None or plane.normal_base[2] < np.cos(np.deg2rad(3)):
                raise ValueError("Cannot establish a level table from current RGB-D")
            self.table = float(-plane.offset_m / plane.normal_base[2])
        hsv = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2HSV)
        cfg = self.detector.config
        height_above_table = cloud[..., 2] - self.table
        planar = valid & (np.abs(height_above_table) < 0.006)
        elevated = valid & (height_above_table > 0.012) & (height_above_table < 0.080)
        found_zones, blocks = {}, []
        self.partial_zones = {}
        for color, spec in cfg["colors"].items():
            # Build colour connectivity within each depth band. Table pixels
            # cannot seed or grow a colour component on an elevated surface.
            ranges = spec["hsv_ranges"]
            support = cfg["color_support"]
            plane_mask, _ = color_masks(hsv, planar, ranges, support)
            zone_patches = []
            for patch in components(plane_mask, 100):
                size = cv2.minAreaRect(cloud[patch, :2].astype(np.float32))[1]
                if max(size) > 1.065 * self.target_edge_m:
                    zone_patches.extend(separate_zone_from_table(patch, hsv[..., 1]))
                else:
                    zone_patches.append(patch)
            for patch in zone_patches:
                ys, xs = np.where(patch)
                pts = cloud[patch]
                if len(pts) > 300 and np.max(np.ptp(pts[:, :2], axis=0)) > 0.09:
                    self.partial_zones[color] = pts
                if (
                    min(xs.min(), ys.min()) < 2
                    or xs.max() >= valid.shape[1] - 2
                    or ys.max() >= valid.shape[0] - 2
                ):
                    continue  # clipped cards are not complete goal bounds
                rect = cv2.minAreaRect(pts[:, :2].astype(np.float32))
                if not (
                    0.925 * self.target_edge_m <= min(rect[1])
                    and max(rect[1]) <= 1.065 * self.target_edge_m
                ):
                    continue
                corners = cv2.boxPoints(rect)
                zone = ZoneEstimate(color, corners, float(np.median(pts[:, 2])), obs.timestamp_s)
                found_zones[color] = zone
                self.zones[color] = zone
            foreground, _ = color_masks(hsv, elevated, ranges, support)
            for patch in components(foreground):
                pts = cloud[patch]
                top = float(np.percentile(pts[:, 2], 98))
                height = top - self.table
                face = pts[pts[:, 2] >= top - 0.004]
                if not 0.040 <= height <= 0.065 or len(face) < 20:
                    continue
                rect = cv2.minAreaRect(face[:, :2].astype(np.float32))
                if min(rect[1]) < 0.030 or max(rect[1]) > 0.070:
                    continue
                ys, xs = np.where(patch)
                if (
                    min(xs.min(), ys.min()) < 3
                    or xs.max() >= valid.shape[1] - 3
                    or ys.max() >= valid.shape[0] - 3
                ):
                    continue
                corners = cv2.boxPoints(rect)
                edge = corners[1] - corners[0]
                yaw = (np.arctan2(edge[1], edge[0]) + np.pi / 4) % (np.pi / 2) - np.pi / 4
                target = TargetEstimate(
                    np.r_[rect[0], self.table + height / 2],
                    height,
                    float(yaw),
                    patch,
                    pts,
                    [float(xs.mean()), float(ys.mean())],
                    0,
                    np.median(obs.rgb[patch], axis=0),
                )
                blocks.append((color, target))
        return blocks, found_zones


class PlacementCheck:
    """Consecutive visible, released and stable observations; missing data resets.

    Cube footprint uses the public 52 mm upper size bound plus uncertainty. It is
    an estimated acceptance signal, not an official score or hidden-body test.
    """

    def __init__(self, zone, cube=True):
        self.zone, self.cube = zone, cube
        self.start = self.last = self.anchor = None
        self.yaw = None
        self.shape = None

    def update(self, target, timestamp_s, width_m):
        if target is None or not np.isfinite([timestamp_s, width_m]).all() or width_m < 0.064:
            self.start = self.last = self.anchor = None
            return False
        center = target.center
        # Conservative circle bounds every yaw of a 52mm cube, plus 4mm uncertainty.
        radius = 0.052 / np.sqrt(2) + 0.004 if self.cube else None
        contained = (
            self.zone.contains(center[:2], radius)
            if self.cube
            else self.zone.contains(target.points[:, :2], 0.012)
        )
        height = center[2] - target.height / 2
        if self.cube:
            height = float(np.percentile(target.points[:, 2], 98)) - 0.052
        valid = contained and abs(height - self.zone.floor_m) < 0.008
        consecutive = self.last is not None and 0 < timestamp_s - self.last <= 0.65
        stable = self.anchor is not None and np.linalg.norm(center - self.anchor) < 0.003
        if not self.cube and self.shape is not None:
            # A fruit can rotate/roll with almost unchanged centroid. Require
            # consistent visible geometry in this fixed verification view too.
            points = target.points[:: max(1, len(target.points) // 500)]
            forward = cKDTree(self.shape).query(points)[0]
            backward = cKDTree(points).query(self.shape)[0]
            stable &= max(np.quantile(forward, 0.95), np.quantile(backward, 0.95)) < 0.004
        if self.cube and self.yaw is not None:
            stable &= abs(
                (target.yaw - self.yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
            ) < np.deg2rad(3)
        if not valid:
            self.start = self.last = self.anchor = None
            return False
        if not consecutive or not stable:
            self.start, self.anchor = timestamp_s, center.copy()
            self.yaw = target.yaw
            self.shape = target.points[:: max(1, len(target.points) // 500)].copy()
        self.last = timestamp_s
        return bool(timestamp_s - self.start >= 2 - 1e-8)
