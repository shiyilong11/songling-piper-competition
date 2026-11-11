"""Causal, bounded memory of a selected target's measured geometry.

This is a static-object hypothesis, not a tracker or contact estimator. Identity
and mask quality must be checked upstream. It never fills in unobserved surfaces.
"""

import numpy as np

from piper_sim.calibration import transform_points, validate_transform


class GraspTargetMemory:
    def __init__(
        self, *, min_points=12, max_age_s=2.0, max_sensor_age_s=0.1, max_center_jump_m=0.04
    ):
        if not isinstance(min_points, int) or min_points < 1:
            raise ValueError("min_points must be a positive integer")
        limits = [max_age_s, max_sensor_age_s, max_center_jump_m]
        if not np.isfinite(limits).all() or min(limits) <= 0:
            raise ValueError("Memory limits must be finite and positive")
        self.min_points = min_points
        self.max_age_s = max_age_s
        self.max_sensor_age_s = max_sensor_age_s
        self.max_center_jump_m = max_center_jump_m
        self.reset()

    def reset(self):
        """Call after explicit reacquisition, new episode, or calibration change."""
        self.target_id = None
        self.points_base = self.point_mask = self.seen_s = None
        self.last_decision_s = self.last_sensor_s = None
        self.blocked = False

    def update(
        self,
        points_end,
        valid_mask,
        T_base_end,
        *,
        target_id,
        timestamp_s,
        now_s,
        allow_static_memory=True,
        invalidate=False,
    ):
        """T_base_end belongs to the image timestamp; current pose via express().

        invalidate latches until reset/new target; use after suspected object
        motion, failed grasp or ambiguous identity. allow_static_memory=False
        permits fresh measurements only (e.g. during contact/lifting).
        """
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("An externally associated target ID is required")
        points = np.asarray(points_end, dtype=float)
        mask = np.asarray(valid_mask)
        if points.ndim != 2 or points.shape[1] != 3 or mask.shape != points.shape[:1]:
            raise ValueError("Expected Nx3 points and an N validity mask")
        if mask.dtype != np.bool_:
            raise ValueError("Point mask must be boolean")
        pose = validate_transform(T_base_end)
        if not np.isfinite([timestamp_s, now_s]).all() or timestamp_s > now_s + 1e-8:
            raise ValueError("Invalid or future sensor timestamp")
        if self.target_id is not None and target_id != self.target_id:
            self.reset()
        if self.last_decision_s is not None and now_s < self.last_decision_s:
            raise ValueError("Decision times must be monotonic")
        if self.last_sensor_s is not None and timestamp_s < self.last_sensor_s:
            raise ValueError("Sensor times must be monotonic")
        repeated = self.last_sensor_s is not None and timestamp_s == self.last_sensor_s
        self.target_id = target_id
        self.last_decision_s, self.last_sensor_s = now_s, timestamp_s
        mask = mask & np.isfinite(points).all(axis=1)
        if invalidate:
            self.blocked = True
        status = "missing"
        visible = False
        if self.blocked:
            status = "invalidated"
        elif (
            not repeated
            and now_s - timestamp_s <= self.max_sensor_age_s
            and mask.sum() >= self.min_points
        ):
            base = np.zeros_like(points)
            base[mask] = transform_points(pose, points[mask])
            # Compare observed centroids; partial views can trigger conservative rejection.
            if (
                self.seen_s is not None
                and now_s - self.seen_s <= self.max_age_s
                and np.linalg.norm(
                    np.median(base[mask], axis=0)
                    - np.median(self.points_base[self.point_mask], axis=0)
                )
                > self.max_center_jump_m
            ):
                self.blocked = True
                status = "association_conflict"
            else:
                self.points_base, self.point_mask = base, mask.copy()
                self.seen_s = timestamp_s
                status, visible = "fresh", True
        if not visible and not self.blocked and self.seen_s is not None:
            if not allow_static_memory:
                status = "memory_disabled"
            elif now_s - self.seen_s <= self.max_age_s:
                status = "remembered"
            else:
                status = "expired"
        usable = status in {"fresh", "remembered"}
        age_known = self.seen_s is not None
        # Zero padding always has a false mask. No stale geometry is exposed as valid.
        result = {
            "points_base": self.points_base.copy() if usable else np.zeros_like(points),
            "valid_mask": self.point_mask.copy() if usable else np.zeros(len(points), bool),
            "visible": visible,
            "usable": usable,
            "remembered": status == "remembered",
            "age_s": float(now_s - self.seen_s) if age_known else 0.0,
            "age_known": age_known,
            "status": status,
        }
        return result

    @staticmethod
    def express(result, T_base_end_current):
        """Express a remembered base-frame estimate at the current control pose."""
        pose = validate_transform(T_base_end_current)
        points = np.zeros_like(result["points_base"], dtype=np.float32)
        mask = result["valid_mask"]
        points[mask] = transform_points(np.linalg.inv(pose), result["points_base"][mask])
        return {**result, "points_end": points}
