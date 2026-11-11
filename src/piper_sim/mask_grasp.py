"""Observed mask axes and local parallel-jaw grasp proposals, without object truth.

Image PCA is diagnostic only. Metric directions and widths use registered depth
and calibration in the base XY plane (level-table assumption). Invisible surfaces,
force closure and collision freedom are not inferred from a single mask.
"""

import cv2
import numpy as np
import yaml

from piper_sim.calibration import ROOT
from piper_sim.competition_sensor import backproject


def load_grasp_config():
    return yaml.safe_load((ROOT / "config/competition/mask_grasp.yaml").read_text())


def principal_axis(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("PCA requires finite Nx2 observed points")
    center = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((points - center).T))
    if values[-1] < 1e-12:
        raise ValueError("Degenerate observed geometry")
    axis = vectors[:, -1]
    # This is an undirected axis. Canonical sign prevents arbitrary +/- flips;
    # it does not identify the fruit's stem/head.
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    ratio = float(values[-1] / max(values[0], 1e-12))
    return center, axis, ratio


def mask_grasp_geometry(obs, calibration, table_top_m, mask, config=None):
    cfg = config or load_grasp_config()
    mask = np.asarray(mask)
    if (
        obs.depth_m is None
        or mask.dtype != bool
        or mask.ndim != 2
        or mask.shape != obs.depth_m.shape
    ):
        raise ValueError("A same-frame boolean mask aligned to metric depth is required")
    if not np.isfinite([table_top_m, obs.timestamp_s]).all():
        raise ValueError("Finite observed table height and timestamp required")
    if cfg["centerline_bins"] < 3 or cfg["voxel_xy_m"] <= 0:
        raise ValueError("Invalid centerline resolution")
    if mask.sum() < cfg["min_mask_pixels"]:
        raise ValueError("Insufficient mask pixels")
    v, u = np.where(mask)
    border = cfg["image_border_px"]
    if (
        min(u.min(), v.min()) < border
        or u.max() >= mask.shape[1] - border
        or v.max() >= mask.shape[0] - border
    ):
        raise ValueError("Clipped mask: reobserve before estimating a grasp")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n > 2:
        sizes = np.sort(stats[1:, cv2.CC_STAT_AREA])
        if sizes[-2] / sizes.sum() > cfg["max_secondary_component_fraction"]:
            raise ValueError("Disconnected/ambiguous mask: separate instances first")
        mask = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        v, u = np.where(mask)
    pixel_center, pixel_axis, pixel_ratio = principal_axis(np.column_stack((u, v)))
    cloud, valid = backproject(obs, calibration)
    if (mask & valid).sum() / mask.sum() < cfg["min_depth_fraction"]:
        raise ValueError("Insufficient depth for a metric grasp vector")
    valid &= mask & (cloud[..., 2] > table_top_m + 0.008) & (cloud[..., 2] < table_top_m + 0.20)
    vv, uu = np.where(valid)
    points = cloud[valid]
    if len(points) < cfg["min_mask_pixels"]:
        raise ValueError("No supported above-table target geometry")
    # Equal XY voxels reduce image-perspective sampling density bias.
    _, indices = np.unique(
        np.floor(points[:, :2] / cfg["voxel_xy_m"]).astype(np.int64), axis=0, return_index=True
    )
    xy = points[indices, :2]
    center, axis, ratio = principal_axis(xy)
    elongated = ratio >= cfg["elongation_eigenvalue_ratio"]
    projection = (xy - center) @ axis
    lo, hi = np.quantile(projection, [0.02, 0.98])
    grid = np.linspace(lo, hi, cfg["centerline_bins"])
    halfbin = (hi - lo) / max(len(grid) - 1, 1) * 0.75
    line = []
    for t in grid:
        section = xy[np.abs(projection - t) <= halfbin]
        line.append(np.median(section, axis=0) if len(section) >= 3 else None)
    anchors = []
    if elongated:
        for fraction in cfg["candidate_fractions"]:
            i = round(fraction * (len(line) - 1))
            if not 0 < i < len(line) - 1:
                raise ValueError("Candidate fractions must select interior centerline sections")
            if line[i] is None or line[i - 1] is None or line[i + 1] is None:
                continue
            tangent = line[i + 1] - line[i - 1]
            if np.linalg.norm(tangent) < 0.004:
                continue
            tangent /= np.linalg.norm(tangent)
            if tangent @ axis < 0:
                tangent = -tangent
            anchors.append((line[i], tangent, fraction))
    else:
        anchors = [
            (center, np.array([np.cos(yaw), np.sin(yaw)]), 0.5)
            for yaw in cfg["round_object_yaws_rad"]
        ]
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    candidates, rejected = [], []
    expanded = []
    for anchor, axis_tangent, fraction in anchors:
        for degrees in cfg.get("oblique_yaw_offsets_deg", [0]) if elongated else [0]:
            angle = np.deg2rad(degrees)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            expanded.append((anchor, rotation @ axis_tangent, axis_tangent, fraction, angle))
    for anchor, tangent, axis_tangent, fraction, angle in expanded:
        closing = np.array([-tangent[1], tangent[0]])
        band = np.abs((points[:, :2] - anchor) @ tangent) <= cfg["finger_pad_length_m"] / 2
        section = points[band]
        if len(section) < 25:
            rejected.append("insufficient_local_section")
            continue
        lateral = (section[:, :2] - anchor) @ closing
        width = float(np.ptp(lateral))
        opening = width + cfg["opening_margin_m"]
        if width < cfg["minimum_observed_width_m"] or opening > cfg["opening_limit_m"]:
            rejected.append("local_width_outside_gripper_limits")
            continue
        grasp_xy = anchor + closing * (lateral.min() + lateral.max()) / 2
        distances = np.linalg.norm(points[:, :2] - grasp_xy, axis=1)
        j = int(np.argmin(distances))
        if distances[j] > cfg["max_center_surface_gap_m"] or distance[vv[j], uu[j]] < 2:
            rejected.append("center_in_gap_or_at_mask_edge")
            continue
        local_top = float(np.percentile(section[:, 2], 95))
        position = np.r_[grasp_xy, table_top_m + (local_top - table_top_m) / 2]
        candidates.append(
            {
                "position_base_m": position.tolist(),
                "surface_marker_base_m": np.r_[grasp_xy, local_top].tolist(),
                "pixel_uv": [int(uu[j]), int(vv[j])],
                "long_axis_base": np.r_[axis_tangent, 0.0].tolist(),
                "finger_long_axis_base": np.r_[tangent, 0.0].tolist(),
                "oblique_angle_rad": float(angle),
                "closing_axis_base": np.r_[closing, 0.0].tolist(),
                "nominal_approach_axis_base": [0.0, 0.0, -1.0],
                "yaw_rad": float(np.arctan2(tangent[1], tangent[0])),
                "observed_section_width_m": width,
                "required_opening_m": opening,
                "centerline_fraction": float(fraction),
                "selection_cost": float(
                    abs(fraction - 0.5) + 0.15 * opening / cfg["opening_limit_m"] + 0.1 * abs(angle)
                ),
                "collision_checked": False,
                "force_closure_verified": False,
            }
        )
    candidates.sort(key=lambda c: c["selection_cost"])
    return {
        "source": "same-frame instance mask + aligned depth + calibrated FK",
        "timestamp_s": float(obs.timestamp_s),
        "frame": "base_link",
        "units": "metres",
        "image_center_uv": pixel_center.tolist(),
        "image_major_axis_uv": pixel_axis.tolist(),
        "image_eigenvalue_ratio": pixel_ratio,
        "metric_major_axis_base": np.r_[axis, 0.0].tolist(),
        "metric_eigenvalue_ratio": ratio,
        "major_axis_reliable": bool(elongated),
        "axis_is_undirected": True,
        "candidates": candidates,
        "rejected": rejected,
        "limitations": [
            "visible surface only",
            "level-table assumption",
            "local width is an estimate",
            "nominal approach must be replaced by actual reachable robot orientation",
        ],
    }
