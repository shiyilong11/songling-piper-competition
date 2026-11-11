"""L1 RGB colour-component proposals, independent of depth and simulator truth.

Colour/contour evidence cannot distinguish same-colour cubes from planar zones.
Every proposal remains role=unknown and grasp_ready=False until RGB-D validation.
IDs identify one component in one frame, not tracked or referee object identities.
"""

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from piper_sim.calibration import ROOT

COLORS = ("red", "green", "yellow")
DRAW_RGB = {"red": (255, 75, 75), "green": (70, 240, 120), "yellow": (255, 220, 40)}


def validate_config(config):
    cfg = copy.deepcopy(config)
    if cfg["schema_version"] != 1 or cfg["input"] != {"encoding": "RGB8", "resize": False}:
        raise ValueError("Expected schema 1 with unresized RGB8 input")
    if set(cfg["colors"]) != set(COLORS):
        raise ValueError("L1 requires exactly red, green, yellow")
    occupied = []
    for color in COLORS:
        ranges = cfg["colors"][color]["hsv_ranges"]
        if not ranges:
            raise ValueError("Each colour needs an HSV range")
        own = []
        for pair in ranges:
            bounds = np.asarray(pair)
            if (
                bounds.shape != (2, 3)
                or bounds.dtype.kind not in "iu"
                or (bounds[0] < 0).any()
                or (bounds[1] > [179, 255, 255]).any()
                or (bounds[0] > bounds[1]).any()
            ):
                raise ValueError("Invalid OpenCV HSV bounds")
            low, high = bounds
            for other_low, other_high in occupied:
                if np.all(np.maximum(low, other_low) <= np.minimum(high, other_high)):
                    raise ValueError("HSV ranges of different colours must not overlap")
            own.append((low, high))
        occupied.extend(own)
    support = cfg.setdefault(
        "color_support",
        {
            "enabled": False,
            "saturation_margin": 20,
            "value_margin": 15,
            "saturation_floor": 35,
            "value_floor": 12,
            "min_seed_pixels": 40,
            "min_seed_fraction": 0.20,
        },
    )
    if not isinstance(support["enabled"], bool):
        raise TypeError("color_support.enabled must be boolean")
    for key in ("saturation_margin", "value_margin", "saturation_floor", "value_floor"):
        if not isinstance(support[key], int) or not 0 <= support[key] <= 255:
            raise ValueError(f"Invalid colour support {key}")
    if (
        not isinstance(support["min_seed_pixels"], int)
        or support["min_seed_pixels"] < 1
        or not np.isfinite(support["min_seed_fraction"])
        or not 0 < support["min_seed_fraction"] <= 1
    ):
        raise ValueError("Invalid colour support seed requirement")
    s = cfg["segmentation"]
    for key in ("open_kernel_px", "close_kernel_px"):
        if not isinstance(s[key], int) or s[key] < 1 or s[key] % 2 != 1:
            raise ValueError("Morphology kernels must be positive odd integers")
    for key in ("min_component_area_px", "boundary_margin_px"):
        if not isinstance(s[key], int) or s[key] < 1:
            raise ValueError(f"Invalid {key}")
    for key in ("max_component_area_fraction", "min_solidity", "min_rectangle_fill"):
        if not np.isfinite(s[key]) or not 0 < s[key] <= 1:
            raise ValueError(f"Invalid {key}")
    if not np.isfinite(s["max_rectangle_aspect"]) or s["max_rectangle_aspect"] < 1:
        raise ValueError("Invalid rectangle aspect ratio")
    include = cfg["roi"]["include_polygon"]
    for poly in ([include] if include is not None else []) + cfg["roi"]["exclude_polygons"]:
        points = np.asarray(poly, float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 3
            or not np.isfinite(points).all()
            or np.any(points < 0)
            or np.any(points > 1)
        ):
            raise ValueError("ROI polygons need >=3 finite normalized uv pairs")
        if cv2.contourArea(points.astype(np.float32)) <= 0:
            raise ValueError("Degenerate ROI polygon")
    return cfg


def load_l1_rgb_config(path=ROOT / "config/competition/l1_rgb.yaml"):
    return validate_config(yaml.safe_load(Path(path).read_text()))


@dataclass
class ColorProposal:
    instance_id: str
    color: str
    mask: np.ndarray
    contour_xy: np.ndarray
    bbox_xywh: tuple
    centroid_uv: tuple
    rectangle_xy: np.ndarray
    area_px: int
    solidity: float
    rectangle_fill: float
    rectangle_aspect: float
    median_hsv: tuple
    rejection_reasons: list[str]
    color_evidence: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "instance_id": self.instance_id,
            "color": self.color,
            "identity_scope": "frame_local_component",
            "role": "unknown",
            "rgb_candidate": not self.rejection_reasons,
            "grasp_ready": False,
            "requires_depth_validation": True,
            "rejection_reasons": self.rejection_reasons,
            "bbox_xywh": list(self.bbox_xywh),
            "centroid_uv": list(self.centroid_uv),
            "contour_xy": self.contour_xy.tolist(),
            "rectangle_xy": self.rectangle_xy.tolist(),
            "area_px": self.area_px,
            "median_hsv": list(self.median_hsv),
            "shape_metrics": {
                "solidity": self.solidity,
                "rectangle_fill": self.rectangle_fill,
                "rectangle_aspect": self.rectangle_aspect,
            },
            "confidence": None,
            "color_evidence": self.color_evidence,
            "warnings": [
                "colour and silhouette do not prove cube identity",
                "touching same-colour objects may form one component",
                "pixel centroid/rectangle are not a 3D grasp pose",
            ],
        }


@dataclass
class RGBDetectionFrame:
    frame_id: str
    timestamp_s: float
    height: int
    width: int
    config_sha256: str
    proposals: list[ColorProposal]
    discarded_small_components: int
    image_quality: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "schema_version": 1,
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "image_size_wh": [self.width, self.height],
            "image_encoding": "RGB8",
            "pixel_coordinates": "original unresized image; u right, v down",
            "config_sha256": self.config_sha256,
            "identity_scope": "frame_local_component",
            "proposals": [p.as_dict() for p in self.proposals],
            "rgb_candidate_counts": {
                c: sum(p.color == c and not p.rejection_reasons for p in self.proposals)
                for c in COLORS
            },
            "discarded_small_components": self.discarded_small_components,
            "depth_used": False,
            "block_count_confirmed": False,
            "grasp_ready": False,
            "image_quality": self.image_quality,
        }


def color_masks(hsv, domain, ranges, support):
    """Strict seeds + bounded connected same-hue support; never fill white glare."""
    strong = np.zeros(hsv.shape[:2], np.uint8)
    weak = np.zeros_like(strong)
    for low, high in ranges:
        strong |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
        low_weak = [
            low[0],
            min(low[1], max(support["saturation_floor"], low[1] - support["saturation_margin"])),
            min(low[2], max(support["value_floor"], low[2] - support["value_margin"])),
        ]
        weak |= cv2.inRange(hsv, np.array(low_weak, np.uint8), np.array(high, np.uint8))
    strong[domain == 0] = 0
    weak[domain == 0] = 0
    if not support["enabled"]:
        return strong, strong
    _, labels, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    seeds = np.bincount(labels[strong != 0], minlength=len(stats))
    keep = (seeds >= support["min_seed_pixels"]) & (
        seeds / np.maximum(stats[:, cv2.CC_STAT_AREA], 1) >= support["min_seed_fraction"]
    )
    keep[0] = False
    # If growth is unsupported, retain only strict evidence rather than expanding
    # into a large connected dark/background region.
    grown = np.where(keep[labels], weak, strong).astype(np.uint8)
    return grown, strong


def image_quality(rgb, hsv, domain):
    active = domain != 0
    if not active.any():
        return {"roi_pixels": 0, "warnings": ["empty_roi"]}
    pixels = hsv[active]
    dark = float(np.mean(pixels[:, 2] < 20))
    white = float(np.mean(np.min(rgb[active], axis=1) >= 250))
    warnings = []
    if dark > 0.25:
        warnings.append("large_dark_region_check_exposure")
    if white > 0.10:
        warnings.append("white_clipping_or_white_surface_check_exposure")
    return {
        "roi_pixels": int(active.sum()),
        "near_black_fraction": dark,
        "white_clipped_fraction": white,
        "low_saturation_fraction": float(np.mean(pixels[:, 1] < 25)),
        "warnings": warnings,
        "white_balance_estimated": False,
        "note": "Image statistics, not camera calibration or probabilities.",
    }


class L1RGBDetector:
    def __init__(self, config=None):
        self.config = validate_config(config) if config is not None else load_l1_rgb_config()
        self.config_sha256 = hashlib.sha256(
            json.dumps(self.config, sort_keys=True).encode()
        ).hexdigest()

    def _domain(self, h, w):
        roi = self.config["roi"]
        domain = np.ones((h, w), np.uint8)

        def polygon(points):
            return np.rint(np.asarray(points) * [w - 1, h - 1]).astype(np.int32)

        if roi["include_polygon"] is not None:
            domain[:] = 0
            cv2.fillPoly(domain, [polygon(roi["include_polygon"])], 1)
        for points in roi["exclude_polygons"]:
            cv2.fillPoly(domain, [polygon(points)], 0)
        return domain

    def detect(self, rgb, *, frame_id, timestamp_s):
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
            or min(rgb.shape[:2]) < 3
        ):
            raise ValueError("Expected HxWx3 RGB uint8 image")
        if not isinstance(frame_id, str) or not frame_id.strip() or not np.isfinite(timestamp_s):
            raise ValueError("Explicit frame ID and finite sensor timestamp are required")
        h, w = rgb.shape[:2]
        s = self.config["segmentation"]
        domain = self._domain(h, w)
        kernel = np.ones((2 * s["boundary_margin_px"] + 1,) * 2, np.uint8)
        interior = cv2.erode(domain, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        boundary = (domain != 0) & (interior == 0)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        proposals, small = [], 0
        for color in COLORS:
            raw, strong = color_masks(
                hsv,
                domain,
                self.config["colors"][color]["hsv_ranges"],
                self.config["color_support"],
            )
            clean = raw.copy()
            for operation, key in (
                (cv2.MORPH_OPEN, "open_kernel_px"),
                (cv2.MORPH_CLOSE, "close_kernel_px"),
            ):
                clean = cv2.morphologyEx(clean, operation, np.ones((s[key], s[key]), np.uint8))
                clean[domain == 0] = 0
            n, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
            components = []
            for label in range(1, n):
                if stats[label, cv2.CC_STAT_AREA] < s["min_component_area_px"]:
                    small += 1
                    continue
                components.append(label)
            # Stable ordering within the frame only; no claim of tracking identity.
            components.sort(key=lambda i: (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]))
            for index, label in enumerate(components):
                component = labels == label
                contours, _ = cv2.findContours(
                    component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contour = max(contours, key=cv2.contourArea)
                area = float(cv2.contourArea(contour))
                hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
                rectangle = cv2.minAreaRect(contour)
                sides = rectangle[1]
                solidity = min(1.0, area / max(hull_area, 1e-8))
                fill = min(1.0, area / max(sides[0] * sides[1], 1e-8))
                aspect = max(sides) / max(min(sides), 1e-8)
                reasons = []
                if component.sum() > s["max_component_area_fraction"] * h * w:
                    reasons.append("too_large_in_image")
                if np.any(component & boundary):
                    reasons.append("touches_image_or_roi_boundary")
                if solidity < s["min_solidity"]:
                    reasons.append("low_solidity")
                if fill < s["min_rectangle_fill"]:
                    reasons.append("low_rectangle_fill")
                if aspect > s["max_rectangle_aspect"]:
                    reasons.append("elongated_component")
                # Morphology joins/cleans proposals; measurement mask retains only
                # pixels with actual colour evidence, not filled background holes.
                mask = component & (raw != 0)
                area_px = int(mask.sum())
                if area_px < s["min_component_area_px"]:
                    small += 1
                    continue
                v, u = np.where(mask)
                median = np.median(hsv[mask], axis=0)
                proposals.append(
                    ColorProposal(
                        f"{frame_id}:{color}:{index + 1:03d}",
                        color,
                        mask,
                        contour.reshape(-1, 2),
                        tuple(int(x) for x in stats[label, :4]),
                        (float(np.mean(u)), float(np.mean(v))),
                        cv2.boxPoints(rectangle),
                        area_px,
                        solidity,
                        fill,
                        float(aspect),
                        tuple(float(x) for x in median),
                        reasons,
                        {
                            "strict_pixels": int(np.count_nonzero(mask & (strong != 0))),
                            "supported_pixels": int(np.count_nonzero(mask & (strong == 0))),
                            "strict_fraction": float(np.mean(strong[mask] != 0)),
                            "method": "seeded_hsv"
                            if self.config["color_support"]["enabled"]
                            else "fixed_hsv",
                            "calibrated_probability": False,
                        },
                    )
                )
        return RGBDetectionFrame(
            frame_id,
            float(timestamp_s),
            h,
            w,
            self.config_sha256,
            proposals,
            small,
            image_quality(rgb, hsv, domain),
        )


def annotate_rgb(rgb, result):
    """Diagnostic overlay, preserving the original image and its pixel geometry."""
    if rgb.shape != (result.height, result.width, 3) or rgb.dtype != np.uint8:
        raise ValueError("Result/image shape mismatch")
    annotated = rgb.copy()
    for p in result.proposals:
        color = DRAW_RGB[p.color] if not p.rejection_reasons else (150, 150, 150)
        annotated[p.mask] = np.rint(annotated[p.mask] * 0.8 + np.array(color) * 0.2).astype(
            np.uint8
        )
        cv2.polylines(annotated, [p.contour_xy.reshape(-1, 1, 2)], True, color, 2)
        x, y, _, _ = p.bbox_xywh
        number = p.instance_id.rsplit(":", 1)[-1]
        suffix = "RGB candidate" if not p.rejection_reasons else "REVIEW"
        label = f"{p.color} #{number}"
        if result.width >= 480 or p.rejection_reasons:
            label += " " + suffix
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        lx = max(0, min(x, result.width - tw - 4))
        ly = max(th + 5, min(y - 5, result.height - baseline - 3))
        cv2.rectangle(
            annotated, (lx, ly - th - 3), (lx + tw + 3, ly + baseline + 2), (18, 20, 24), -1
        )
        cv2.putText(annotated, label, (lx + 1, ly), font, scale, color, thickness, cv2.LINE_AA)
    return annotated
