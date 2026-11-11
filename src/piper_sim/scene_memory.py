"""Bounded RGB-D object memory with conservative identity association.

Consumes only measured observations, estimated calibration and public masks.
Does not select task targets or infer task semantics; those belong to Qwen.
"""

import copy
from dataclasses import dataclass, field

import numpy as np
import yaml
from PIL import Image, ImageDraw

from piper_sim.calibration import ROOT
from piper_sim.scene_observation import InstanceMask, ObjectEstimate, SceneObserver
from piper_sim.temporal_evidence import DEFAULT_SELECTION, frame_quality, select_keyframes


def load_memory_config(path=ROOT / "config/vlm/multiframe.yaml"):
    cfg = yaml.safe_load(path.read_text())
    # Older recorded bundles inherit the user's fixed base-distance convention.
    cfg.setdefault("upper_lower_reference", "robot_base_horizontal_distance")
    cfg.setdefault("robot_reference_point_base_m", [0.0, 0.0, 0.0])
    cfg.setdefault("relation_min_distance_gap_m", 0.02)
    cfg.setdefault("keyframe_selection", copy.deepcopy(DEFAULT_SELECTION))
    cfg.setdefault("event_history_size", 8)
    if cfg["schema_version"] != 1:
        raise ValueError("Unsupported scene memory schema")
    for key in (
        "max_keyframes",
        "max_objects",
        "min_object_points",
        "max_fused_points",
        "max_total_image_pixels",
        "min_image_pixels",
        "max_new_tokens",
        "event_history_size",
    ):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"Invalid memory count: {key}")
    for key in (
        "association_radius_m",
        "association_margin_m",
        "max_chroma_distance",
        "association_max_age_s",
        "observation_max_age_s",
        "memory_max_age_s",
        "geometry_reset_distance_m",
        "max_plan_position_change_m",
        "max_plan_extent_change_m",
        "relation_min_distance_gap_m",
    ):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"Invalid memory threshold: {key}")
    axis = np.asarray(cfg["left_axis_base"], float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1):
        raise ValueError("Left axis must be a measured/configured unit vector")
    origin = np.asarray(cfg["robot_reference_point_base_m"], float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Robot reference point must be a finite base-frame position")
    if cfg["upper_lower_reference"] != "robot_base_horizontal_distance":
        raise ValueError("Unsupported upper/lower spatial convention")
    if (
        not isinstance(cfg["required_views"], list)
        or not cfg["required_views"]
        or len(set(cfg["required_views"])) != len(cfg["required_views"])
    ):
        raise ValueError("Required survey views must be nonempty and unique")
    if cfg["max_total_image_pixels"] < cfg["max_keyframes"] * cfg["min_image_pixels"]:
        raise ValueError("Image budget is smaller than minimum keyframe budget")
    settings = cfg["keyframe_selection"]
    if not isinstance(settings, dict) or set(settings) != set(DEFAULT_SELECTION):
        raise ValueError("Invalid keyframe selection settings")
    if any(not np.isfinite(value) or value <= 0 for value in settings.values()):
        raise ValueError("Keyframe thresholds must be positive and finite")
    if settings["min_rotation_deg"] > 180 or settings["max_rgb_difference"] > 1:
        raise ValueError("Invalid keyframe angular/image threshold")
    return cfg


def chroma(rgb, mask):
    color = np.median(rgb[mask], axis=0).astype(float)
    return color / max(float(color.sum()), 1)


@dataclass
class RememberedObject:
    instance_id: str
    latest: ObjectEstimate
    color: np.ndarray
    last_seen_s: float
    first_seen_s: float
    last_frame_id: str
    points_base: np.ndarray
    evidence_frames: list[str] = field(default_factory=list)
    unresolved_candidates: list[str] = field(default_factory=list)


@dataclass
class MemoryFrame:
    frame_id: str
    view_id: str
    timestamp_s: float
    rgb: np.ndarray
    scene: object
    instances: dict[str, str]
    importance: int
    thumbnail: np.ndarray
    quality: dict
    joint_position_rad: np.ndarray
    joint_velocity_rad_s: np.ndarray
    gripper_width_m: float
    depth_m: np.ndarray | None

    def annotated_image(self):
        image = Image.fromarray(self.rgb.copy())
        draw = ImageDraw.Draw(image)
        for obj in self.scene.objects:
            key = self.instances.get(obj.instance_key)
            if key is None:
                continue
            ys, xs = np.where(obj.observed_mask)
            if not len(xs):
                continue
            box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            draw.rectangle(box, outline="#12d7ac", width=2)
            draw.text(
                (box[0], max(0, box[1] - 13)),
                key,
                fill="white",
                stroke_width=1,
                stroke_fill="#143b32",
            )
        # Show the measured candidate centre, never an invented complete region boundary.
        k = self.scene.intrinsics
        if k is not None:
            camera_from_base = np.linalg.inv(
                self.scene.T_base_end @ self.scene.T_end_camera_estimated
            )
            for region in self.scene.regions:
                point = camera_from_base @ np.r_[region.T_base_region[:3, 3], 1.0]
                if point[2] <= 0:
                    continue
                x = k["fx"] * point[0] / point[2] + k["cx"]
                y = k["fy"] * point[1] / point[2] + k["cy"]
                if 0 <= x < image.width and 0 <= y < image.height:
                    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#80caff", outline="white")
                    draw.text(
                        (min(x + 7, max(0, image.width - 100)), max(0, y - 14)),
                        region.region_id,
                        fill="white",
                        stroke_width=1,
                        stroke_fill="#17334c",
                    )
        return image


class SceneMemory:
    def __init__(self, config=None, observer=None):
        self.cfg = copy.deepcopy(config or load_memory_config())
        self.observer = observer or SceneObserver()
        self.objects = {}
        self.frames = []
        self.views = {}
        self.regions = {}
        self.version = 0
        self.latest_frame = None
        self.timestamp_s = None
        self.warnings = []
        self._next_id = 1
        self.events = []
        self.latest_event = None
        self.selection_report = {}

    def update(self, observation, calibration, *, detections, view_id, now_s=None, regions=None,
               identity_bindings=None, allow_geometric_proposals=True):
        """Atomic ingest. Masks may come from SAM or another documented adapter."""
        if not isinstance(view_id, str) or not view_id:
            raise ValueError("View id must be a nonempty string")
        stamp = float(observation.timestamp_s)
        if not np.isfinite(stamp) or (self.timestamp_s is not None and stamp < self.timestamp_s):
            raise ValueError("Memory requires monotonically ordered sensor timestamps")
        if observation.rgb is None or observation.rgb.dtype != np.uint8:
            raise ValueError("Memory requires an RGB uint8 observation")
        detections = list(detections)
        bindings = dict(identity_bindings or {})
        if (not set(bindings).issubset({d.instance_key for d in detections})
                or len(set(bindings.values())) != len(bindings)
                or any(key not in self.objects or self.objects[key].unresolved_candidates
                       or stamp - self.objects[key].last_seen_s > self.cfg["memory_max_age_s"]
                       for key in bindings.values())):
            raise ValueError("Explicit tracking bindings require unique, live, resolved memory IDs")
        scene = self.observer.build(
            observation,
            calibration,
            now_s=stamp if now_s is None else now_s,
            detections=detections,
            regions=regions,
            allow_geometric_proposals=allow_geometric_proposals,
        )
        thumbnail, quality = frame_quality(observation)
        frame_id = f"frame_{self.version + 1:06d}"
        observed = [o for o in scene.objects if len(o.points_base) >= self.cfg["min_object_points"]]
        colors = [chroma(observation.rgb, o.observed_mask) for o in observed]
        active_objects = {
            key: old
            for key, old in self.objects.items()
            if stamp - old.last_seen_s <= self.cfg["memory_max_age_s"]
        }
        options = []
        for obj, color in zip(observed, colors):
            if obj.instance_key in bindings:
                # The owner validates epoch, continuity and identity before supplying this map.
                options.append([(0.0, bindings[obj.instance_key])])
                continue
            center = np.median(obj.points_base, axis=0)
            choices = []
            for key, old in active_objects.items():
                if key in bindings.values():
                    continue
                age = stamp - old.last_seen_s
                distance = float(np.linalg.norm(center - np.median(old.latest.points_base, axis=0)))
                color_distance = float(np.linalg.norm(color - old.color))
                if (
                    age <= self.cfg["association_max_age_s"]
                    and distance <= self.cfg["association_radius_m"]
                    and color_distance <= self.cfg["max_chroma_distance"]
                ):
                    choices.append((distance + 0.02 * color_distance, key))
            options.append(sorted(choices))
        proposed = []
        for choices in options:
            unique = bool(choices) and (
                len(choices) == 1
                or choices[1][0] - choices[0][0] >= self.cfg["association_margin_m"]
            )
            proposed.append(choices[0][1] if unique else None)
        # Two observations competing for one memory ID are not silently merged.
        conflicts = {key for key in proposed if key is not None and proposed.count(key) > 1}
        new_count = sum(key is None or key in conflicts for key in proposed)
        if len(active_objects) + new_count > self.cfg["max_objects"]:
            raise ValueError("Object memory full; explicitly reset or resolve associations")
        updated = copy.deepcopy(active_objects)
        aliases, warnings = {}, list(scene.warnings)
        new_ids, geometry_changes, appearance_changes = [], {}, []
        for obj, color, choices, key in zip(observed, colors, options, proposed):
            ambiguous = (
                [candidate for _, candidate in choices] if key is None or key in conflicts else []
            )
            if key in conflicts:
                key = None
            if key is None:
                key = f"obj_{self._next_id:03d}"
                self._next_id += 1
                updated[key] = RememberedObject(
                    key,
                    obj,
                    color,
                    stamp,
                    stamp,
                    frame_id,
                    obj.points_base.copy(),
                    unresolved_candidates=ambiguous,
                )
                new_ids.append(key)
            else:
                old = updated[key]
                displacement = np.linalg.norm(
                    np.median(obj.points_base, axis=0) - np.median(old.latest.points_base, axis=0)
                )
                if displacement > self.cfg["geometry_reset_distance_m"]:
                    geometry_changes[key] = float(displacement)
                if np.linalg.norm(color - old.color) > self.cfg["max_chroma_distance"] / 2:
                    appearance_changes.append(key)
                points = (
                    obj.points_base
                    if displacement > self.cfg["geometry_reset_distance_m"]
                    else np.concatenate([old.points_base, obj.points_base])
                )
                # Deterministic bounded visible geometry, never a complete model.
                cells = np.floor(points / 0.003).astype(np.int64)
                _, ids = np.unique(cells, axis=0, return_index=True)
                points = points[np.sort(ids)]
                if len(points) > self.cfg["max_fused_points"]:
                    points = points[
                        np.linspace(0, len(points) - 1, self.cfg["max_fused_points"], dtype=int)
                    ]
                old.latest, old.color, old.points_base = obj, color, points.copy()
                old.last_seen_s, old.last_frame_id = stamp, frame_id
            track = updated[key]
            track.evidence_frames = [*track.evidence_frames[-7:], frame_id]
            if track.unresolved_candidates:
                warnings.append(f"ambiguous_association:{key}")
            aliases[obj.instance_key] = key
        new_view = view_id not in self.views
        frame = MemoryFrame(
            frame_id,
            view_id,
            stamp,
            observation.rgb.copy(),
            scene,
            aliases,
            new_count * 5 + int(new_view) * 3,
            thumbnail,
            quality,
            np.asarray(observation.joint_position_rad).copy(),
            np.asarray(observation.joint_velocity_rad_s).copy(),
            float(observation.gripper_width_m),
            None if observation.depth_m is None else np.asarray(observation.depth_m).copy(),
        )
        pool = [
            *[f for f in self.frames if stamp - f.timestamp_s <= self.cfg["memory_max_age_s"]],
            frame,
        ]
        selected, selection_report = select_keyframes(pool, self.cfg)
        visible = set(aliases.values())
        previous_visible = (
            set() if self.latest_frame is None else set(self.latest_frame.instances.values())
        )
        expired = sorted(set(self.objects) - set(active_objects))
        new_regions = sorted(r.region_id for r in scene.regions if r.region_id not in self.regions)
        region_changes = {}
        for region in scene.regions:
            old = self.regions.get(region.region_id)
            if old is not None:
                change = np.linalg.norm(region.T_base_region[:3, 3] - old.T_base_region[:3, 3])
                if change > self.cfg["geometry_reset_distance_m"]:
                    region_changes[region.region_id] = float(change)
        ambiguous = sorted(key for key in visible if updated[key].unresolved_candidates)
        new_survey = view_id in self.cfg["required_views"] and (
            view_id not in self.views or stamp - self.views[view_id] > self.cfg["memory_max_age_s"]
        )
        event = {
            "memory_version": self.version + 1,
            "timestamp_s": stamp,
            "view_id": view_id,
            "new_instance_ids": new_ids,
            "expired_instance_ids": expired,
            "observed_center_changes_m": geometry_changes,
            "appearance_changed_ids": appearance_changes,
            "ambiguous_instance_ids": ambiguous,
            "reobserved_instance_ids": sorted((visible & set(active_objects)) - previous_visible),
            "not_observed_in_current_frame": sorted(previous_visible - visible),
            "new_region_ids": new_regions,
            "observed_region_center_changes_m": region_changes,
            "new_survey_evidence": bool(new_survey),
            "semantic_refresh_recommended": bool(
                new_ids
                or expired
                or geometry_changes
                or appearance_changes
                or ambiguous
                or new_regions
                or region_changes
                or new_survey
            ),
            "scope": "observation/config estimates; missing from a frame is not disappearance; centre changes may reflect partial views",
        }
        # Do not accumulate a repeated event for every unchanged video frame.
        meaningful = (
            event["semantic_refresh_recommended"]
            or event["reobserved_instance_ids"]
            or event["not_observed_in_current_frame"]
        )
        if meaningful:
            self.events = [*self.events, event][-self.cfg["event_history_size"] :]
        self.latest_event, self.selection_report = event, selection_report
        self.objects = updated
        self.frames = selected
        self.latest_frame, self.timestamp_s = frame, stamp
        self.version += 1
        self.views[view_id] = stamp
        self.regions = {
            key: region
            for key, region in self.regions.items()
            if region.timestamp_s is None
            or stamp - region.timestamp_s <= self.cfg["memory_max_age_s"]
        }
        for region in scene.regions:
            self.regions[region.region_id] = region
        self.warnings = warnings
        return frame

    def summary(self, now_s=None):
        if self.latest_frame is None:
            raise ValueError("No observations in scene memory")
        now = self.timestamp_s if now_s is None else float(now_s)
        if not np.isfinite(now) or now < self.timestamp_s:
            raise ValueError("Invalid memory query timestamp")
        evidence = {f.frame_id for f in self.frames}
        missing = [
            view
            for view in self.cfg["required_views"]
            if view not in self.views or now - self.views[view] > self.cfg["memory_max_age_s"]
        ]
        rows = []
        for key, track in sorted(self.objects.items()):
            points = track.points_base
            center = np.median(track.latest.points_base, axis=0)
            distance = np.linalg.norm(
                center[:2] - np.asarray(self.cfg["robot_reference_point_base_m"])[:2]
            )
            rows.append(
                {
                    "instance_id": key,
                    "label_hint": track.latest.label,
                    "label_hint_source": track.latest.source.as_dict(),
                    "last_seen_s": track.last_seen_s,
                    "age_s": now - track.last_seen_s,
                    "currently_visible": track.last_frame_id == self.latest_frame.frame_id,
                    "last_observed_center_base_m": center.tolist(),
                    "robot_base_horizontal_distance_m": float(distance),
                    "observed_extent_base_m": np.ptp(points, axis=0).tolist(),
                    "geometry_complete": False,
                    "uncertainty_status": "uncalibrated; grows while unseen",
                    "identity_status": "ambiguous" if track.unresolved_candidates else "estimated",
                    "candidate_previous_ids": track.unresolved_candidates,
                    "evidence_frame_ids": [f for f in track.evidence_frames if f in evidence],
                }
            )
        region_rows = []
        for region in self.regions.values():
            row = region.as_dict()
            row["robot_base_horizontal_distance_m"] = float(
                np.linalg.norm(
                    np.asarray(region.T_base_region)[:2, 3]
                    - np.asarray(self.cfg["robot_reference_point_base_m"])[:2]
                )
            )
            row["evidence_frame_ids"] = [
                f.frame_id
                for f in self.frames
                if any(r.region_id == region.region_id for r in f.scene.regions)
            ]
            row["currently_observed"] = any(
                r.region_id == region.region_id for r in self.latest_frame.scene.regions
            )
            row["geometry_complete"] = False
            region_rows.append(row)
        return {
            "memory_version": self.version,
            "timestamp_s": now,
            "latest_frame_id": self.latest_frame.frame_id,
            "base_frame": self.latest_frame.scene.frame,
            "units": "metres",
            "left_axis_base": self.cfg["left_axis_base"],
            "coordinate_source": self.cfg["coordinate_source"],
            "spatial_reference": {
                "upper_lower": self.cfg["upper_lower_reference"],
                "robot_reference_point_base_m": self.cfg["robot_reference_point_base_m"],
                "upper": "larger horizontal XY distance to the fixed robot base",
                "lower": "smaller horizontal XY distance to the fixed robot base",
                "ignored_for_upper_lower": ["image_pixel_y", "base_height_z"],
                "distance_measurement": "visible-point median XY centre estimate; not full-object truth",
                "min_resolvable_distance_gap_m": self.cfg["relation_min_distance_gap_m"],
                "source": "User convention: upper means farther from the robot base; lower means nearer.",
            },
            "objects": rows,
            "temporal_context": {
                "selection": copy.deepcopy(self.selection_report),
                "semantic_refresh_hint": {
                    "memory_version": self.version,
                    "recommended": (
                        self.latest_event["semantic_refresh_recommended"]
                        or self.latest_event["memory_version"] != self.version
                    ),
                    "external_update_since_observation": self.latest_event["memory_version"]
                    != self.version,
                },
                "latest_observation_event": copy.deepcopy(self.latest_event),
                "recent_events": [
                    copy.deepcopy(event)
                    for event in self.events
                    if now - event["timestamp_s"] <= self.cfg["memory_max_age_s"]
                ],
                "scope": "hints only; no automatic model call, target switch, or robot action",
            },
            "regions": region_rows,
            "coverage": {
                "required_views": self.cfg["required_views"],
                "missing_views": missing,
                "survey_complete": not missing,
                "scope": "configured views sampled; occluded/unseen space remains unknown",
            },
            "keyframes": [
                {
                    "frame_id": f.frame_id,
                    "view_id": f.view_id,
                    "timestamp_s": f.timestamp_s,
                    "observation_quality": copy.deepcopy(f.quality),
                    "instance_ids": list(f.instances.values()),
                    "T_base_camera_estimated": (
                        f.scene.T_base_end @ f.scene.T_end_camera_estimated
                    ).tolist(),
                }
                for f in self.frames
            ],
            "warnings": self.warnings,
        }

    def current_selection(self, instance_id, region_id, *, now_s, memory_version):
        """Fresh public binding for SceneObserver/policy input, not a robot command."""
        if memory_version != self.version:
            raise ValueError("Scene memory changed; ask Qwen to validate the task again")
        summary = self.summary(now_s)
        track = self.objects.get(instance_id)
        if track is None or region_id not in self.regions:
            raise ValueError("Unknown object or destination ID")
        if track.unresolved_candidates or track.last_frame_id != self.latest_frame.frame_id:
            raise ValueError("Selected target must be reobserved and unambiguous before execution")
        if summary["timestamp_s"] - track.last_seen_s > self.cfg["observation_max_age_s"]:
            raise ValueError("Selected target observation is stale")
        detection = InstanceMask(
            instance_id,
            track.latest.observed_mask.copy(),
            track.last_seen_s,
            track.latest.confidence or 0.0,
            track.latest.source,
            track.latest.label,
        )
        region = self.regions[region_id]
        if (
            region.timestamp_s is not None
            and now_s - region.timestamp_s > self.cfg["observation_max_age_s"]
        ):
            raise ValueError("Destination estimate is stale")
        return detection, copy.deepcopy(region)
