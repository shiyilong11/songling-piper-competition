"""Sensor-only scene estimates shared by simulation and deployment.

No simulator/model/judge imports. Visible geometry is not a complete object model,
and unobserved space is never reported as collision free.
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from piper_sim.calibration import ROOT, transform_points, validate_transform
from piper_sim.observation import Observation


@dataclass(frozen=True)
class Provenance:
    kind: str  # sensor_estimate / manual_config / oracle (rejected by the observer)
    detail: str

    def require_public(self):
        if (
            self.kind not in {"sensor_estimate", "manual_config"}
            or not isinstance(self.detail, str)
            or not self.detail.strip()
        ):
            raise ValueError(
                "Scene inputs require documented estimates/configuration; oracle forbidden"
            )

    def as_dict(self):
        return {"kind": self.kind, "detail": self.detail}


@dataclass
class CalibrationEstimate:
    T_end_camera: np.ndarray
    source: Provenance


@dataclass
class InstanceMask:
    instance_key: str  # detector/tracker key, never a simulator body ID
    mask: np.ndarray
    timestamp_s: float
    confidence: float
    source: Provenance
    label: str | None = None


@dataclass
class PlacementRegion:
    region_id: str
    T_base_region: np.ndarray
    half_size_xy_m: np.ndarray
    source: Provenance
    kind: str = "planar_region"
    timestamp_s: float | None = None  # Static manual measurements may omit the timestamp.

    def validate(self):
        self.source.require_public()
        validate_transform(self.T_base_region)
        half = np.asarray(self.half_size_xy_m)
        if half.shape != (2,) or not np.isfinite(half).all() or np.any(half <= 0):
            raise ValueError("Region half size must contain two positive metre values")
        if self.kind not in {"planar_region", "container"} or not self.region_id:
            raise ValueError("Invalid placement region")
        return self

    def as_dict(self):
        return {
            "region_id": self.region_id,
            "kind": self.kind,
            "T_base_region": np.asarray(self.T_base_region).tolist(),
            "half_size_xy_m": np.asarray(self.half_size_xy_m).tolist(),
            "source": self.source.as_dict(),
            "timestamp_s": self.timestamp_s,
        }


@dataclass
class PlaneEstimate:
    normal_base: np.ndarray
    offset_m: float  # n @ p + d = 0, normal points upward
    confidence: float  # geometric fit quality heuristic, not a calibrated probability
    inlier_ratio: float
    rmse_m: float
    observed_points_base: np.ndarray
    source: Provenance


@dataclass
class ObjectEstimate:
    instance_key: str
    points_base: np.ndarray
    observed_mask: np.ndarray
    source: Provenance
    confidence: float | None
    label: str | None = None
    identity_scope: str = "frame_local"  # Detector IDs are only persistent if its tracker says so.

    def as_dict(self):
        return {
            "instance_key": self.instance_key,
            "label": self.label,
            "identity_scope": self.identity_scope,
            "source": self.source.as_dict(),
            "confidence": self.confidence,
            "observed_point_count": len(self.points_base),
            "observed_centroid_base_m": np.mean(self.points_base, axis=0).tolist(),
            "observed_aabb_base_m": [
                self.points_base.min(axis=0).tolist(),
                self.points_base.max(axis=0).tolist(),
            ],
            "geometry_complete": False,
        }


@dataclass
class SceneEstimate:
    timestamp_s: float
    frame: str
    T_base_end: np.ndarray
    calibration_source: Provenance
    T_end_camera_estimated: np.ndarray
    intrinsics: dict | None
    points_base: np.ndarray
    pixel_uv: np.ndarray
    table: PlaneEstimate | None
    objects: list[ObjectEstimate]
    regions: list[PlacementRegion]
    exclusion_source: Provenance | None = None
    warnings: list[str] = field(default_factory=list)
    target_key: str | None = None
    target_source: Provenance | None = None

    @property
    def target(self):
        return next((o for o in self.objects if o.instance_key == self.target_key), None)

    def select_target(self, key, source):
        source.require_public()
        if not any(o.instance_key == key for o in self.objects):
            raise ValueError("Selected target is not observed in this frame")
        self.target_key, self.target_source = key, source
        return self

    def as_dict(self):
        table = None
        if self.table is not None:
            t = self.table
            table = {
                "normal_base": t.normal_base.tolist(),
                "offset_m": t.offset_m,
                "confidence": t.confidence,
                "confidence_is_probability": False,
                "inlier_ratio": t.inlier_ratio,
                "rmse_m": t.rmse_m,
                "observed_point_count": len(t.observed_points_base),
                "boundary_complete": False,
                "source": t.source.as_dict(),
            }
        return {
            "schema_version": 1,
            "timestamp_s": self.timestamp_s,
            "frame": self.frame,
            "units": "metres",
            "calibration_source": self.calibration_source.as_dict(),
            "T_base_end": self.T_base_end.tolist(),
            "T_end_camera_estimated": self.T_end_camera_estimated.tolist(),
            "intrinsics_estimated": copy.deepcopy(self.intrinsics),
            "exclusion_source": None
            if self.exclusion_source is None
            else self.exclusion_source.as_dict(),
            "observed_point_count": len(self.points_base),
            "table": table,
            "objects": [o.as_dict() for o in self.objects],
            "regions": [r.as_dict() for r in self.regions],
            "target_key": self.target_key,
            "target_source": None if self.target_source is None else self.target_source.as_dict(),
            "warnings": self.warnings,
            "unobserved_space": "unknown",
            "collision_free_certified": False,
        }


def load_scene_observation_config(path=None):
    cfg = yaml.safe_load(Path(path or ROOT / "config/scene_observation.yaml").read_text())
    return validate_scene_observation_config(cfg)


def validate_scene_observation_config(cfg):
    if cfg["schema_version"] != 1:
        raise ValueError("Unsupported scene observation schema")
    low, high = np.asarray(cfg["workspace"]["min_m"]), np.asarray(cfg["workspace"]["max_m"])
    if (
        low.shape != (3,)
        or high.shape != (3,)
        or not np.isfinite([low, high]).all()
        or np.any(low >= high)
    ):
        raise ValueError("Invalid scene observation workspace")
    if not isinstance(cfg["pixel_stride"], int) or cfg["pixel_stride"] < 1:
        raise ValueError("pixel_stride must be a positive integer")
    positive = [
        cfg["max_observation_age_s"],
        cfg["max_input_skew_s"],
        cfg["plane"]["distance_threshold_m"],
        cfg["plane"]["min_observed_span_m"],
        cfg["proposals"]["voxel_size_m"],
        cfg["proposals"]["connection_radius_m"],
    ]
    if not np.isfinite(positive).all() or np.any(np.asarray(positive) <= 0):
        raise ValueError("Scene observation thresholds must be finite and positive")
    for value in [
        cfg["plane"]["iterations"],
        cfg["plane"]["max_fit_points"],
        cfg["plane"]["min_inliers"],
        cfg["proposals"]["min_points"],
        cfg["policy_points"]["target_count"],
        cfg["policy_points"]["scene_count"],
    ]:
        if not isinstance(value, int) or value < 1:
            raise ValueError("Scene observation counts must be positive integers")
    if cfg["plane"]["max_fit_points"] < 3 or cfg["plane"]["min_inliers"] < 3:
        raise ValueError("Plane fitting requires at least three points")
    if not 0 < cfg["plane"]["min_inlier_ratio"] <= 1 or not 0 < cfg["plane"]["max_tilt_deg"] < 90:
        raise ValueError("Invalid plane ratio/tilt")
    if not 0 <= cfg["policy_points"]["foreground_fraction"] <= 1:
        raise ValueError("Invalid foreground sampling fraction")
    if not isinstance(cfg["base_frame"], str) or not cfg["base_frame"]:
        raise ValueError("Base frame must be specified")
    for seed in [cfg["plane"]["ransac_seed"], cfg["policy_points"]["seed"]]:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("Random seeds must be nonnegative integers")
    for limits in [
        cfg["depth_range_m"],
        [cfg["proposals"]["min_height_m"], cfg["proposals"]["max_height_m"]],
    ]:
        if len(limits) != 2 or not np.isfinite(limits).all() or not 0 < limits[0] < limits[1]:
            raise ValueError("Invalid depth/height range")
    configured_regions(cfg)  # Validate measured regions too.
    return cfg


def configured_regions(cfg):
    return [
        PlacementRegion(
            region_id=r["region_id"],
            T_base_region=np.asarray(r["T_base_region"], dtype=float),
            half_size_xy_m=np.asarray(r["half_size_xy_m"], dtype=float),
            source=Provenance(**r["source"]),
            kind=r.get("kind", "planar_region"),
            timestamp_s=r.get("timestamp_s"),
        ).validate()
        for r in cfg["regions"]
    ]


def _check_stamp(stamp, reference, tolerance):
    if (
        stamp is None
        or not np.isfinite([stamp, reference]).all()
        or abs(stamp - reference) > tolerance
    ):
        raise ValueError("Scene input timestamp does not match the RGB-D observation")


def _estimate_plane(points, cfg):
    # Accumulate/refine in float64; large constant float32 planes otherwise drift in their mean.
    points = np.asarray(points, dtype=np.float64)
    if len(points) < cfg["min_inliers"]:
        return None
    rng = np.random.default_rng(cfg["ransac_seed"])
    fit = points[rng.choice(len(points), min(len(points), cfg["max_fit_points"]), replace=False)]
    best = None
    cosine = np.cos(np.deg2rad(cfg["max_tilt_deg"]))
    for _ in range(cfg["iterations"]):
        a, b, c = fit[rng.choice(len(fit), 3, replace=False)]
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length < 1e-9:
            continue
        n /= length
        if n[2] < 0:
            n = -n
        if n[2] < cosine:
            continue
        d = -float(n @ a)
        mask = np.abs(fit @ n + d) <= cfg["distance_threshold_m"]
        count = int(mask.sum())
        if best is None or count > best[0]:
            best = (count, mask)
    if best is None or best[0] < 3:
        return None
    inliers = fit[best[1]]
    for _ in range(2):
        center = inliers.mean(axis=0)
        _, _, vt = np.linalg.svd(inliers - center, full_matrices=False)
        n = vt[-1]
        if n[2] < 0:
            n = -n
        d = -float(n @ center)
        residual = points @ n + d
        inliers = points[np.abs(residual) <= cfg["distance_threshold_m"]]
        if len(inliers) < cfg["min_inliers"]:
            return None
    ratio = len(inliers) / len(points)
    span = np.ptp(inliers[:, :2], axis=0)
    if (
        n[2] < cosine
        or ratio < cfg["min_inlier_ratio"]
        or np.min(span) < cfg["min_observed_span_m"]
    ):
        return None
    rmse = float(np.sqrt(np.mean((inliers @ n + d) ** 2)))
    confidence = float(ratio * np.exp(-rmse / cfg["distance_threshold_m"]))
    return PlaneEstimate(
        n,
        d,
        confidence,
        ratio,
        rmse,
        inliers,
        Provenance("sensor_estimate", "upward RGB-D plane RANSAC + SVD"),
    )


def _geometric_proposals(points, uv, image_shape, plane, cfg):
    heights = points @ plane.normal_base + plane.offset_m
    ids = np.flatnonzero((heights >= cfg["min_height_m"]) & (heights <= cfg["max_height_m"]))
    if not len(ids):
        return []
    pts = points[ids]
    cells, inverse = np.unique(
        np.floor(pts / cfg["voxel_size_m"]).astype(np.int64), axis=0, return_inverse=True
    )
    # Voxel centres only construct connectivity; estimates retain the original measured points.
    centres = (cells + 0.5) * cfg["voxel_size_m"]
    pairs = cKDTree(centres).query_pairs(cfg["connection_radius_m"], output_type="ndarray")
    if len(pairs):
        rows, cols = np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]]
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(cells), len(cells)))
        _, labels = connected_components(graph, directed=False)
    else:
        labels = np.arange(len(cells))
    point_labels = labels[inverse]
    result = []
    for label in np.unique(point_labels):
        group = ids[point_labels == label]
        if len(group) < cfg["min_points"]:
            continue
        mask = np.zeros(image_shape, dtype=bool)
        mask[uv[group, 1], uv[group, 0]] = True
        result.append(
            ObjectEstimate(
                f"proposal_{len(result):03d}",
                points[group].copy(),
                mask,
                Provenance(
                    "sensor_estimate", "RGB-D above-plane connected component; class unknown"
                ),
                None,
            )
        )
    return result


class SceneObserver:
    def __init__(self, config=None):
        self.config = validate_scene_observation_config(
            copy.deepcopy(config or load_scene_observation_config())
        )

    def build(
        self,
        observation: Observation,
        calibration: CalibrationEstimate,
        *,
        now_s,
        detections=(),
        regions=None,
        exclusion_mask=None,
        exclusion_source=None,
        exclusion_timestamp_s=None,
        allow_geometric_proposals=True,
        estimate_table=True,
    ):
        cfg, obs = self.config, observation
        _check_stamp(obs.timestamp_s, now_s, cfg["max_observation_age_s"])
        if obs.timestamp_s > now_s + 1e-6:
            raise ValueError("Observation is from the future; use one common clock")
        calibration.source.require_public()
        detections = list(detections)
        t_base_end = validate_transform(obs.T_base_end).copy()
        t_base_camera = t_base_end @ validate_transform(calibration.T_end_camera)
        regions = copy.deepcopy(configured_regions(cfg) if regions is None else list(regions))
        for r in regions:
            r.validate()
            if r.source.kind == "sensor_estimate" or r.timestamp_s is not None:
                _check_stamp(r.timestamp_s, obs.timestamp_s, cfg["max_input_skew_s"])
        if len({r.region_id for r in regions}) != len(regions):
            raise ValueError("Duplicate placement region ID")
        scene = SceneEstimate(
            timestamp_s=float(obs.timestamp_s),
            frame=cfg["base_frame"],
            T_base_end=t_base_end,
            calibration_source=calibration.source,
            T_end_camera_estimated=np.asarray(calibration.T_end_camera).copy(),
            intrinsics=copy.deepcopy(obs.intrinsics),
            points_base=np.empty((0, 3), np.float32),
            pixel_uv=np.empty((0, 2), int),
            table=None,
            objects=[],
            regions=regions,
        )
        if obs.depth_m is None:
            scene.warnings.append("missing_depth")
            return scene
        depth = np.asarray(obs.depth_m)
        if depth.ndim != 2 or (obs.rgb is not None and obs.rgb.shape[:2] != depth.shape):
            raise ValueError("Expected aligned HxW optical-axis Z depth")
        if obs.intrinsics is None:
            raise ValueError("Rectified intrinsics required")
        k = obs.intrinsics
        if (
            not np.isfinite([k[x] for x in ["fx", "fy", "cx", "cy"]]).all()
            or min(k["fx"], k["fy"]) <= 0
        ):
            raise ValueError("Invalid rectified intrinsics")
        for detection in detections:
            detection.source.require_public()
            _check_stamp(detection.timestamp_s, obs.timestamp_s, cfg["max_input_skew_s"])
            if (
                np.asarray(detection.mask).shape != depth.shape
                or np.asarray(detection.mask).dtype != bool
            ):
                raise ValueError("Detection masks must be aligned HxW boolean arrays")
            if (
                not detection.instance_key
                or not np.isfinite(detection.confidence)
                or not 0 <= detection.confidence <= 1
            ):
                raise ValueError("Invalid instance key/confidence")
        if len({d.instance_key for d in detections}) != len(detections):
            raise ValueError("Duplicate detection instance key")
        v, u = np.mgrid[
            0 : depth.shape[0] : cfg["pixel_stride"], 0 : depth.shape[1] : cfg["pixel_stride"]
        ]
        z = depth[v, u]
        valid = np.isfinite(z) & (z >= cfg["depth_range_m"][0]) & (z <= cfg["depth_range_m"][1])
        if exclusion_mask is not None:
            if exclusion_source is None:
                raise ValueError("Robot/exclusion masks need explicit provenance")
            exclusion_source.require_public()
            _check_stamp(exclusion_timestamp_s, obs.timestamp_s, cfg["max_input_skew_s"])
            scene.exclusion_source = exclusion_source
            if (
                np.asarray(exclusion_mask).shape != depth.shape
                or np.asarray(exclusion_mask).dtype != bool
            ):
                raise ValueError("Exclusion mask must be an aligned boolean array")
            valid &= ~exclusion_mask[v, u]
        else:
            scene.warnings.append("self_robot_not_masked")
        u, v, z = u[valid], v[valid], z[valid]
        camera_points = np.column_stack(
            [(u - k["cx"]) * z / k["fx"], (v - k["cy"]) * z / k["fy"], z]
        )
        points = transform_points(t_base_camera, camera_points)
        inside = np.all(
            (points >= cfg["workspace"]["min_m"]) & (points <= cfg["workspace"]["max_m"]), axis=1
        )
        scene.points_base = points[inside].astype(np.float32)
        scene.pixel_uv = np.column_stack([u[inside], v[inside]])
        if not len(scene.points_base):
            scene.warnings.append("no_valid_workspace_depth")
            return scene
        if estimate_table:
            scene.table = _estimate_plane(scene.points_base, cfg["plane"])
        if estimate_table and scene.table is None:
            scene.warnings.append("table_plane_not_found")
        uv = scene.pixel_uv
        if detections:
            for detection in detections:
                keep = detection.mask[uv[:, 1], uv[:, 0]]
                if keep.any():
                    scene.objects.append(
                        ObjectEstimate(
                            detection.instance_key,
                            scene.points_base[keep].copy(),
                            detection.mask.copy(),
                            detection.source,
                            detection.confidence,
                            detection.label,
                            "detector_key_tracking_not_verified",
                        )
                    )
                else:
                    scene.warnings.append(f"instance_without_valid_depth:{detection.instance_key}")
        elif scene.table is not None and allow_geometric_proposals:
            scene.objects = _geometric_proposals(
                scene.points_base, uv, depth.shape, scene.table, cfg["proposals"]
            )
            scene.warnings.append("geometric_proposals_have_no_semantic_labels_or_tracking")
        return scene


def _fixed_points(points, count, rng):
    result = np.zeros((count, 3), np.float32)
    valid = np.zeros(count, bool)
    n = min(count, len(points))
    if n:
        result[:n] = points[rng.choice(len(points), n, replace=False)]
        valid[:n] = True
    return result, valid


def _context_points(scene, end_points, cfg, rng):
    count = cfg["policy_points"]["scene_count"]
    if scene.table is None:
        return _fixed_points(end_points, count, rng)
    height = scene.points_base @ scene.table.normal_base + scene.table.offset_m
    foreground = np.flatnonzero(height >= cfg["proposals"]["min_height_m"])
    foreground_count = min(
        len(foreground), round(count * cfg["policy_points"]["foreground_fraction"])
    )
    chosen = rng.choice(foreground, foreground_count, replace=False)
    background = np.flatnonzero(height < cfg["proposals"]["min_height_m"])
    chosen = np.r_[
        chosen, rng.choice(background, min(len(background), count - len(chosen)), replace=False)
    ]
    if len(chosen) < min(count, len(end_points)):
        available = np.ones(len(end_points), bool)
        available[chosen] = False
        rest = np.flatnonzero(available)
        chosen = np.r_[chosen, rng.choice(rest, min(len(rest), count - len(chosen)), replace=False)]
    return _fixed_points(end_points[chosen], count, rng)


def policy_scene_features(scene, config=None, *, goal_region_id=None):
    """Fixed shapes with explicit validity; never promote unobserved data to known geometry."""
    cfg = config or load_scene_observation_config()
    scene.calibration_source.require_public()
    for item in [*scene.objects, *scene.regions]:
        item.source.require_public()
    if scene.table is not None:
        scene.table.source.require_public()
    if scene.target_source is not None:
        scene.target_source.require_public()
    if scene.exclusion_source is not None:
        scene.exclusion_source.require_public()
    rng = np.random.default_rng(cfg["policy_points"]["seed"])
    end_from_base = np.linalg.inv(validate_transform(scene.T_base_end))
    scene_end = transform_points(end_from_base, scene.points_base)
    target_end = (
        np.empty((0, 3))
        if scene.target is None
        else transform_points(end_from_base, scene.target.points_base)
    )
    target, target_valid = _fixed_points(target_end, cfg["policy_points"]["target_count"], rng)
    context, context_valid = _context_points(scene, scene_end, cfg, rng)
    region = next((r for r in scene.regions if r.region_id == goal_region_id), None)
    if goal_region_id is not None and region is None:
        raise ValueError("Requested placement region is unavailable")
    plane = np.zeros(4, np.float32)
    if scene.table is not None:
        plane[:] = np.r_[scene.table.normal_base, scene.table.offset_m]
    return {
        "timestamp_s": scene.timestamp_s,
        "point_frame": "estimated_end_frame",
        "target_points": target,
        "target_valid_mask": target_valid,
        "scene_points": context,
        "scene_valid_mask": context_valid,
        "target_visible": bool(target_valid.any()),
        "table_plane_base": plane,
        "table_valid": scene.table is not None,
        "table_confidence": 0.0 if scene.table is None else scene.table.confidence,
        "T_base_goal": np.eye(4, dtype=np.float32)
        if region is None
        else np.asarray(region.T_base_region, dtype=np.float32).copy(),
        "T_end_goal": np.eye(4, dtype=np.float32)
        if region is None
        else (end_from_base @ region.T_base_region).astype(np.float32),
        "goal_half_size_xy_m": np.zeros(2, np.float32)
        if region is None
        else np.asarray(region.half_size_xy_m, dtype=np.float32),
        "goal_valid": region is not None,
    }
