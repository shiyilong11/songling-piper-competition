"""Visible coloured near-table patches, estimated only from RGB-D and calibration.

These are incomplete placement candidates, not certified tray boundaries.
"""

import cv2
import numpy as np

from piper_sim.scene_observation import PlacementRegion, Provenance


def estimate_colored_regions(observation, scene):
    if scene.table is None or not len(scene.points_base):
        return []
    rgb, uv, points = observation.rgb, scene.pixel_uv, scene.points_base
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    pixels = hsv[uv[:, 1], uv[:, 0]]
    excluded = np.zeros(rgb.shape[:2], np.uint8)
    for obj in scene.objects:
        excluded[obj.observed_mask] = 1
    excluded = cv2.dilate(excluded, np.ones((7, 7), np.uint8))
    heights = points @ scene.table.normal_base + scene.table.offset_m
    near_table = (np.abs(heights) < 0.009) & (excluded[uv[:, 1], uv[:, 0]] == 0)
    saturated = (pixels[:, 1] >= 60) & (pixels[:, 2] >= 35)
    hues = pixels[:, 0]
    color_masks = {
        "red": (hues <= 12) | (hues >= 170),
        "yellow": (hues >= 15) & (hues < 35),
        "green": (hues >= 35) & (hues < 90),
        "blue": (hues >= 90) & (hues < 135),
    }
    regions = []
    for color, color_mask in color_masks.items():
        ids = np.flatnonzero(near_table & saturated & color_mask)
        if len(ids) < 120:
            continue
        mask = np.zeros(rgb.shape[:2], np.uint8)
        mask[uv[ids, 1], uv[ids, 0]] = 1
        # Fill the regular sampling gaps, without grouping distant colour patches.
        dense = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        _, labels = cv2.connectedComponents(dense)
        memberships = labels[uv[ids, 1], uv[ids, 0]]
        candidates = []
        for label in np.unique(memberships):
            selected = ids[memberships == label]
            cloud = points[selected]
            if len(cloud) < 120:
                continue
            extent = np.ptp(cloud, axis=0)
            if np.any(extent[:2] < 0.02) or np.ptp(heights[selected]) > 0.008:
                continue
            candidates.append(cloud)
        if len(candidates) != 1:
            # Multiple same-colour patches require disambiguation, not arbitrary selection.
            continue
        cloud = candidates[0]
        median = np.median(cloud, axis=0)
        center = cloud[np.argmin(np.linalg.norm(cloud - median, axis=1))]
        pose = np.eye(4)
        pose[:3, 3] = center
        regions.append(
            PlacementRegion(
                color + "_region",
                pose,
                np.minimum(np.ptp(cloud, axis=0)[:2] / 4, 0.04),
                Provenance(
                    "sensor_estimate",
                    f"RGB-D HSV {color} near-table connected patch; "
                    "above-plane objects excluded; incomplete visible footprint; "
                    "not a certified placement boundary",
                ),
                timestamp_s=observation.timestamp_s,
            )
        )
    return regions
