"""Competition adapters around existing user-configured segmentation clients.

Input prompts are public RGB-D proposals or text. No simulator labels are used.
The research adapters are reused, not edited. Models are loaded lazily, one route
at a time; errors are reported, never replaced by a hidden ground-truth mask.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from piper_sim.competition_sensor import backproject, components
from piper_sim.demo_skill import TargetEstimate
from piper_sim.grasp_feedback import GraspFeedbackError, match_fresh_target
from piper_sim.l2_qwen_categories import L2_QWEN_CATEGORIES, masked_category_crop
from piper_sim.mask_grasp import mask_grasp_geometry
from piper_sim.sam_adapter import Sam2Client


@dataclass
class SegmentedTarget:
    key: str
    target: TargetEstimate
    color: str
    metadata: dict


def split_depth_occlusions(foreground, depth_m):
    """Separate touching projections at an observed depth step."""
    height, width = foreground.shape
    padded_foreground = np.pad(foreground, 1)
    padded_depth = np.pad(depth_m, 1, constant_values=np.nan)
    far_edge = np.zeros_like(foreground)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            window = np.s_[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            far_edge |= padded_foreground[window] & (depth_m - padded_depth[window] > 0.012)
    return foreground & ~far_edge


def _depth_prompt_regions(obs, calibration, table, zones, *, split_depth, focus=()):
    cloud, valid = backproject(obs, calibration)
    foreground = valid & (cloud[..., 2] > table + 0.01) & (cloud[..., 2] < table + 0.20)
    foreground &= (cloud[..., 0] > 0.12) & (cloud[..., 0] < 0.75) & (np.abs(cloud[..., 1]) < 0.5)
    for zone in zones.values():
        center = zone.center
        # Exclude already sorted objects only for new pick proposals.
        foreground &= ~(
            (np.abs(cloud[..., 0] - center[0]) < 0.102)
            & (np.abs(cloud[..., 1] - center[1]) < 0.102)
        )
    if split_depth:
        foreground = split_depth_occlusions(foreground, obs.depth_m)
    prompts = []
    relevant = None
    if focus:
        relevant = np.zeros_like(valid)
        for center, radius in focus:
            relevant |= np.sum((cloud[..., :2] - np.asarray(center)[:2]) ** 2, axis=-1) < radius**2
    for patch in components(foreground, 60):
        if relevant is not None and not np.any(patch & relevant):
            continue
        ys, xs = np.where(patch)
        distance = cv2.distanceTransform(patch.astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(np.argmax(distance), distance.shape)
        if (
            min(xs.min(), ys.min()) < 4
            or xs.max() >= patch.shape[1] - 4
            or ys.max() >= patch.shape[0] - 4
        ):
            continue
        prompts.append(
            (
                [int(x), int(y)],
                [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                patch,
            )
        )
    return prompts


def public_geometry_prompts(obs, calibration, table, zones, *, split_depth=False):
    return [
        (point, box)
        for point, box, _ in _depth_prompt_regions(
            obs, calibration, table, zones, split_depth=split_depth
        )
    ]


def observed_color(rgb, mask):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[mask]
    saturated = hsv[(hsv[:, 1] >= 50) & (hsv[:, 2] >= 25)]
    if len(saturated) < 40:
        return "unknown"
    h = saturated[:, 0]
    bins = {
        "red": (h <= 9) | (h >= 170),
        "orange_brown": (h > 9) & (h < 20),
        "yellow": (h >= 20) & (h < 35),
        "green": (h >= 35) & (h <= 89),
    }
    scores = {k: float(v.mean()) for k, v in bins.items()}
    best = max(scores, key=scores.get)
    if scores["yellow"] + scores["green"] > 0.9 and max(scores["yellow"], scores["green"]) < 0.6:
        return "yellow_green"  # explicit boundary category, task card must decide its route
    return best if scores[best] >= 0.6 else "unknown"


def observed_fruit_category(rgb, target):
    """Classify the six supplied fruit types using mask colour and metric shape."""
    color = observed_color(rgb, target.mask)
    extent = float(np.max(np.ptp(target.points[:, :2], axis=0)))
    ratio = target.grasp_geometry["metric_eigenvalue_ratio"]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[target.mask]
    hue, saturation = np.median(hsv[:, :2], axis=0)
    green_fraction = float(np.mean((hsv[:, 0] >= 35) & (hsv[:, 0] <= 89)))
    # Use the orange skin and green calyx together. Hue 8 lies in the
    # generic red bin, so that bin alone confuses persimmon with red pepper.
    if 6 <= hue <= 18 and saturation >= 185 and green_fraction >= 0.08 and ratio < 3:
        return "persimmon"
    if color in {"green", "yellow_green"}:
        if extent >= 0.085 and ratio >= (4.0 if hue >= 45 else 5.0):
            return "cucumber"
        return "pear"
    if color == "red":
        return "bell_pepper"
    if color == "orange_brown":
        if saturation >= 185:
            return "persimmon"
        if saturation <= 170:
            return "potato"
    if color == "yellow":
        if extent >= 0.085 and ratio >= 3.0:
            return "banana"
        if extent <= 0.090 and ratio <= 3.0 and hue <= 27 and saturation <= 175:
            return "potato"
        if extent <= 0.090 and ratio <= 3.0 and saturation >= 185 and green_fraction >= 0.015:
            return "bell_pepper"
    return "unknown"


def target_from_mask(obs, calibration, table, mask, *, projected=None, compute_grasps=True):
    cloud, valid = backproject(obs, calibration) if projected is None else projected
    if np.asarray(mask).dtype != bool or mask.shape != valid.shape:
        raise ValueError("Invalid aligned instance mask")
    if mask.sum() < 60 or (mask & valid).sum() / mask.sum() < 0.65:
        raise ValueError("Insufficient registered depth inside model mask")
    elevated = mask & valid & (cloud[..., 2] > table + 0.008) & (cloud[..., 2] < table + 0.20)
    if elevated.sum() < 0.6 * (mask & valid).sum():
        raise ValueError("Mask is mostly supporting surface rather than an object")
    mask = elevated
    ys, xs = np.where(mask)
    if (
        len(xs) < 60
        or min(xs.min(), ys.min()) < 4
        or xs.max() >= mask.shape[1] - 4
        or ys.max() >= mask.shape[0] - 4
    ):
        raise ValueError("Insufficient or clipped fruit geometry")
    pts = cloud[mask]
    rect = cv2.minAreaRect(pts[:, :2].astype(np.float32))
    if max(rect[1]) > 0.22 or min(rect[1]) < 0.012:
        raise ValueError("Invalid fruit dimensions in observed point cloud")
    corners = cv2.boxPoints(rect)
    edges = np.roll(corners, -1, axis=0) - corners
    edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    yaw = float(np.arctan2(edge[1], edge[0]))
    top = float(np.percentile(pts[:, 2], 98))
    height = top - table
    if not 0.015 < height < 0.16:
        raise ValueError("Fruit height outside configured gripper envelope")
    target = TargetEstimate(
        np.r_[rect[0], table + height / 2],
        height,
        yaw,
        mask,
        pts,
        [float(xs.mean()), float(ys.mean())],
        0,
        np.median(obs.rgb[mask], axis=0),
    )
    if compute_grasps:
        target.grasp_geometry = mask_grasp_geometry(obs, calibration, table, mask)
        if target.grasp_geometry["major_axis_reliable"]:
            axis = target.grasp_geometry["metric_major_axis_base"]
            target.yaw = float(np.arctan2(axis[1], axis[0]))
    return target


def geometry_targets(obs, calibration, table, *, split_depth=False, compute_grasps=True):
    cloud, valid = backproject(obs, calibration)
    fg = valid & (cloud[..., 2] > table + 0.012) & (cloud[..., 2] < table + 0.20)
    fg &= (cloud[..., 0] > 0.08) & (cloud[..., 0] < 0.80) & (np.abs(cloud[..., 1]) < 0.55)
    if split_depth:
        fg = split_depth_occlusions(fg, obs.depth_m)
    targets = []
    for mask in components(fg, 60):
        try:
            targets.append(
                target_from_mask(
                    obs,
                    calibration,
                    table,
                    mask,
                    projected=(cloud, valid),
                    compute_grasps=compute_grasps,
                )
            )
        except ValueError:
            continue
    return targets


class CompetitionSegmenter:
    def __init__(self, method="sam2", *, classify_fruits=False, category_backend="rules"):
        if method not in {"sam2", "yoloe_sam2", "sam31", "qwen_sam2"}:
            raise ValueError("Choose sam2, yoloe_sam2, sam31 or qwen_sam2")
        self.method = method
        self.classify_fruits = classify_fruits
        if category_backend not in {"rules", "qwen"} or (
            category_backend == "qwen" and not classify_fruits
        ):
            raise ValueError("Qwen classification requires the fruit-category task card")
        self.category_client = None
        if category_backend == "qwen":
            from piper_sim.segmentation_lab import GroundingClient

            self.category_client = GroundingClient()
        self.sam = Sam2Client() if method in {"sam2", "yoloe_sam2"} else None
        self.yoloe = None
        self.yoloe_error = None
        self.pipeline = None
        if self.sam is None:
            from piper_sim.segmentation_lab import SegmentationPipeline

            self.pipeline = SegmentationPipeline()

    def classify(self, rgb, target):
        if self.classify_fruits:
            return observed_fruit_category(rgb, target)
        return observed_color(rgb, target.mask)

    def quick_masks(self, obs, calibration, table, prompts, *, split_depth):
        from piper_sim.yoloe_mask import YoloeMaskModel

        if self.yoloe_error or obs.rgb.shape != (480, 640, 3) or not prompts:
            return {}
        try:
            if self.yoloe is None:
                self.yoloe = YoloeMaskModel()
            instances = self.yoloe.predict(obs.rgb)
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            self.yoloe_error = str(exc)
            return {}
        # Only replace a depth prompt with a unique, graspable mask. Missing
        # or competing masks leave that prompt on the SAM2 path.
        matches = [
            [index for index, ((x, y), _, _) in enumerate(prompts) if mask[y, x]]
            for mask, _ in instances
        ]
        quick = {}
        for (mask, score), hits in zip(instances, matches, strict=True):
            if len(hits) != 1 or sum(hits[0] in other for other in matches) != 1:
                continue
            support = prompts[hits[0]][2]
            if split_depth:
                mask = mask & support
                if mask.sum() < 0.60 * support.sum():
                    continue
            try:
                target = target_from_mask(obs, calibration, table, mask)
                depth_target = target_from_mask(
                    obs, calibration, table, support, compute_grasps=False
                )
                match_fresh_target(target, [depth_target], {})
            except (ValueError, GraspFeedbackError):
                continue
            if not target.grasp_geometry["candidates"] or (
                self.category_client is None and self.classify(obs.rgb, target) == "unknown"
            ):
                continue
            quick[hits[0]] = (
                mask,
                {
                    "model": "yoloe-26n-visual",
                    "visual_prompt_score": score,
                    "prompt_source": "cached_visual_examples_and_registered_depth",
                    "semantic_label": None,
                },
                target,
            )
        return quick

    def detect(
        self,
        obs,
        calibration,
        table,
        zones,
        output,
        *,
        include_zones=False,
        split_depth=False,
        focus=(),
    ):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        masks, rejected = [], []
        quick = {}
        if self.sam is not None:
            prompts = _depth_prompt_regions(
                obs,
                calibration,
                table,
                {} if include_zones else zones,
                split_depth=split_depth,
                focus=focus,
            )
            if self.method == "yoloe_sam2":
                quick = self.quick_masks(obs, calibration, table, prompts, split_depth=split_depth)
            for i, (point, box, support) in enumerate(prompts):
                if i in quick:
                    folder = output / f"proposal_{i}"
                    folder.mkdir(exist_ok=True)
                    np.save(folder / "yoloe_mask.npy", quick[i][0], allow_pickle=False)
                    masks.append(quick[i])
                    continue
                try:
                    instance, meta = self.sam.segment(
                        obs.rgb, point, obs.timestamp_s, output.name, output / f"proposal_{i}", box
                    )
                except ValueError as exc:
                    rejected.append({"proposal": i, "reason": str(exc)})
                    continue
                mask = instance.mask & support if split_depth else instance.mask
                if split_depth and mask.sum() < 0.60 * support.sum():
                    rejected.append({"proposal": i, "reason": "low mask overlap with depth region"})
                    continue
                masks.append(
                    (
                        mask,
                        {
                            **meta,
                            "prompt_source": "registered_depth_foreground",
                            "semantic_label": None,
                        },
                        None,
                    )
                )
        else:
            path = output / "rgb.png"
            Image.fromarray(obs.rgb).save(path)
            engine = "sam31" if self.method == "sam31" else "qwen"
            prompts = (
                ["pear", "banana", "persimmon", "bell pepper", "cucumber", "potato"]
                if engine == "sam31"
                else ["找出图中所有果蔬模型，逐个给出名称、框和物体内的点，不包括目标区和机械臂。"]
            )
            for p, prompt in enumerate(prompts):
                folder = output / f"prompt_{p}"
                folder.mkdir(exist_ok=True)
                result = self.pipeline.run_engine(
                    engine, path, prompt, None, folder, lambda *_: None
                )
                for i, item in enumerate(result.get("targets", [])):
                    mask = np.array(Image.open(folder / f"target_{i + 1}" / "mask.png")) > 0
                    masks.append(
                        (
                            mask,
                            {
                                **item,
                                "prompt_source": "public_text_and_RGB",
                                "actual_prompt": prompt,
                            },
                            None,
                        )
                    )
        targets, view_requests = [], []
        for i, (mask, meta, cached_target) in enumerate(masks):
            try:
                target = (
                    target_from_mask(obs, calibration, table, mask)
                    if cached_target is None
                    else cached_target
                )
                if not include_zones and any(
                    z.contains(target.center[:2], 0) for z in zones.values()
                ):
                    continue
                if any(np.linalg.norm(target.center - t.target.center) < 0.025 for t in targets):
                    rejected.append(
                        {"index": i, "reason": "duplicate_or_ambiguous_nearby_instance"}
                    )
                    continue
                label = (
                    self.classify(obs.rgb, target) if self.category_client is None else "unknown"
                )
                targets.append(SegmentedTarget(f"{output.name}:{i}", target, label, meta))
            except ValueError as exc:
                rejected.append({"index": i, "reason": str(exc)})
                # Partial real depth may guide a view change, never a grasp.
                cloud, valid = backproject(obs, calibration)
                partial = (
                    valid & mask & (cloud[..., 2] > table + 0.008) & (cloud[..., 2] < table + 0.20)
                )
                points = cloud[partial]
                if "depth" in str(exc).lower() and len(points) >= 30:
                    view_requests.append(
                        {
                            "reason": str(exc),
                            "points_base_m": points[:: max(1, len(points) // 40)].tolist(),
                        }
                    )
        if self.category_client is not None and targets:
            folder = output / "qwen_categories"
            folder.mkdir(exist_ok=True)
            crops = {}
            for i, candidate in enumerate(targets, 1):
                identity = str(i)
                path = folder / f"{identity}.png"
                masked_category_crop(obs.rgb, candidate.target.mask, identity).save(path)
                crops[identity] = path.resolve()
            categories = self.category_client.predict_l2_categories(crops, folder)["categories"]
            if categories.keys() != crops.keys() or any(
                label not in L2_QWEN_CATEGORIES for label in categories.values()
            ):
                raise ValueError("Invalid Qwen category response")
            for i, candidate in enumerate(targets, 1):
                candidate.color = categories[str(i)]
                candidate.metadata["semantic_label"] = candidate.color
                candidate.metadata["category_candidate_id"] = str(i)
        return targets, {
            "method": self.method,
            "inference_wall_s": time.perf_counter() - start,
            "raw_masks": len(masks),
            "accepted": len(targets),
            "yoloe_masks": len(quick),
            "yoloe_error": self.yoloe_error,
            "split_depth_occlusions": split_depth,
            "focused_verification": bool(focus) and self.sam is not None,
            "accepted_targets": [
                {
                    "id": t.key,
                    "color": t.color,
                    "model": t.metadata.get("model", self.method),
                    "center_m": t.target.center.tolist(),
                    "span_m": np.ptp(t.target.points, axis=0).tolist(),
                    "grasp_geometry": t.target.grasp_geometry,
                }
                for t in targets
            ],
            "rejected": rejected,
            "view_requests": view_requests,
            "mask_source": "learned_RGB_segmentation",
            "routing_labels": "fruit_category" if self.classify_fruits else "colour",
            "semantic_classification": self.category_client is not None,
            "category_backend": "qwen" if self.category_client is not None else "rules",
        }

    def close(self):
        self.yoloe = None
        if self.category_client is not None:
            self.category_client.close()
        if self.sam:
            self.sam.close()
        if self.pipeline:
            self.pipeline.close()
