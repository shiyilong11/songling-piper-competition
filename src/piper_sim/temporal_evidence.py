"""Select bounded, complementary sensor evidence without interpreting task language."""

import numpy as np
from PIL import Image

DEFAULT_SELECTION = {
    "min_translation_m": 0.015,
    "min_rotation_deg": 10.0,
    "max_rgb_difference": 0.02,
    "min_quality_gain": 0.25,
}


def frame_quality(observation):
    thumbnail = np.asarray(Image.fromarray(observation.rgb).resize((32, 24))).copy()
    gray = np.mean(observation.rgb.astype(np.float32) / 255.0, axis=2)
    edge = (
        sum(
            float(np.mean(np.diff(gray, axis=axis) ** 2)) for axis in (0, 1) if gray.shape[axis] > 1
        )
        / 2
    )
    depth = observation.depth_m
    valid = 0.0 if depth is None else float(np.mean(np.isfinite(depth) & (depth > 0)))
    return thumbnail, {
        "image_edge_energy": edge,
        "depth_valid_fraction": valid,
        "scope": "uncalibrated evidence utility; not detection probability",
    }


def evidence_quality(frame):
    result = {
        frame.instances[obj.instance_key]: float(len(obj.points_base))
        for obj in frame.scene.objects
        if obj.instance_key in frame.instances
    }
    for region in frame.scene.regions:
        if region.source.kind == "sensor_estimate":
            result["region:" + region.region_id] = 1.0
    return result


def near_same_pose(a, b, settings):
    ta = a.scene.T_base_end @ a.scene.T_end_camera_estimated
    tb = b.scene.T_base_end @ b.scene.T_end_camera_estimated
    distance = np.linalg.norm(ta[:3, 3] - tb[:3, 3])
    cosine = np.clip((np.trace(ta[:3, :3].T @ tb[:3, :3]) - 1) / 2, -1, 1)
    angle = np.degrees(np.arccos(cosine))
    return distance <= settings["min_translation_m"] and angle <= settings["min_rotation_deg"]


def image_difference(a, b):
    return float(np.mean(np.abs(a.thumbnail.astype(float) - b.thumbnail.astype(float))) / 255)


def select_keyframes(pool, cfg):
    """Keep current and survey views; spend remaining budget on new/better evidence."""
    settings = cfg["keyframe_selection"]
    limit = cfg["max_keyframes"]
    selected = [pool[-1]]
    for view in cfg["required_views"]:
        candidate = next((f for f in reversed(pool) if f.view_id == view), None)
        if (
            candidate is not None
            and all(candidate.frame_id != f.frame_id for f in selected)
            and len(selected) < limit
        ):
            selected.append(candidate)
    redundant = set()
    while len(selected) < limit:
        best = {}
        for frame in selected:
            for key, quality in evidence_quality(frame).items():
                best[key] = max(best.get(key, 0.0), quality)
        scores = []
        for candidate in pool:
            if any(candidate.frame_id == f.frame_id for f in selected):
                continue
            evidence = evidence_quality(candidate)
            new_count = len(set(evidence) - set(best))
            gains = [
                quality / max(best[key], 1.0) - 1
                for key, quality in evidence.items()
                if key in best
            ]
            gain = sum(min(value, 10.0) for value in gains if value >= settings["min_quality_gain"])
            detail = candidate.quality["image_edge_energy"]
            old_detail = max(f.quality["image_edge_energy"] for f in selected)
            detail_gain = detail / max(old_detail, 1e-8) - 1
            if detail_gain >= settings["min_quality_gain"]:
                gain += min(detail_gain, 1.0)
            equivalent = any(
                near_same_pose(candidate, f, settings)
                and image_difference(candidate, f) <= settings["max_rgb_difference"]
                and set(evidence) == set(evidence_quality(f))
                for f in selected
            )
            if equivalent and not new_count and not gain:
                redundant.add(candidate.frame_id)
                continue
            pose_novelty = not any(near_same_pose(candidate, f, settings) for f in selected)
            score = new_count * 10 + gain + float(pose_novelty)
            if score > 0:
                scores.append((score, candidate.timestamp_s, candidate.frame_id, candidate))
        if not scores:
            break
        selected.append(max(scores, key=lambda item: item[:3])[3])
    selected.sort(key=lambda f: f.timestamp_s)
    return selected, {
        "candidate_frames": len(pool),
        "selected_frames": len(selected),
        "redundant_candidate_frames": len(redundant),
        "selection": "current + survey views + complementary/better visible evidence",
    }
