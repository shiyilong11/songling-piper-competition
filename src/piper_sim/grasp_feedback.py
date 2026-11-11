"""Sensor-only grasp checks. Width evidence is an estimate, not proof of contact."""

import numpy as np


class GraspFeedbackError(RuntimeError):
    def __init__(self, message, code, evidence=None, retryable=False):
        super().__init__(message)
        self.code, self.evidence, self.retryable = code, evidence or {}, retryable


class GraspEmptyError(GraspFeedbackError):
    def __init__(self, message="夹爪接近完全闭合，疑似抓空。", evidence=None):
        super().__init__(message, "empty_grasp", evidence, retryable=True)


def match_fresh_target(previous, candidates, config):
    """Associate nearby visible geometry; never silently switch to another object.

    Bounded position/shape/color association is not semantic object tracking.
    Ambiguity, large motion or clipped geometry forces a new selection cycle.
    """
    if not candidates:
        raise GraspFeedbackError(
            "抓前复核未找到有效目标，重新观察。", "target_changed", retryable=True
        )
    distances = np.array([np.linalg.norm(c.center - previous.center) for c in candidates])
    order = np.argsort(distances)
    index = int(order[0])
    correction = float(distances[index])
    if correction > config.get("max_target_correction_m", 0.015):
        raise GraspFeedbackError(
            "目标位置变化过大，拒绝沿用旧抓取位置。",
            "target_changed",
            {"correction_m": correction},
            retryable=True,
        )
    if len(order) > 1 and distances[order[1]] - correction < config.get(
        "association_margin_m", 0.015
    ):
        raise GraspFeedbackError(
            "抓前目标关联不唯一，需要重新选物。", "target_changed", retryable=True
        )
    fresh = candidates[index]
    ratio = fresh.height / max(previous.height, 1e-6)
    if not 0.7 <= ratio <= 1.4:
        raise GraspFeedbackError(
            "目标几何发生明显变化，重新识别。", "target_changed", retryable=True
        )
    if previous.appearance_rgb is not None and fresh.appearance_rgb is not None:
        a, b = np.array(previous.appearance_rgb), np.array(fresh.appearance_rgb)
        if np.linalg.norm(a / max(a.sum(), 1) - b / max(b.sum(), 1)) > 0.18:
            raise GraspFeedbackError(
                "目标外观不一致，拒绝切换物体。", "target_changed", retryable=True
            )
    ys, xs = np.where(fresh.mask)
    h, w = fresh.mask.shape
    margin = config.get("image_border_margin_px", 4)
    if (
        not len(xs)
        or xs.min() < margin
        or ys.min() < margin
        or xs.max() >= w - margin
        or ys.max() >= h - margin
    ):
        raise GraspFeedbackError(
            "抓前图像中的目标不完整，需要重新观察。", "target_changed", retryable=True
        )
    return fresh, {"correction_m": correction, "source": "fresh_rgbd_geometry_and_color"}


class WidthGuard:
    """Debounced retention check using measured width and observation timestamps."""

    def __init__(self, reference_width_m, config):
        self.reference = float(reference_width_m)
        self.config = config
        self.bad_since = None
        self.bad_code = None

    def update(self, width_m, timestamp_s, stage):
        evidence = {
            "width_m": float(width_m),
            "reference_width_m": self.reference,
            "timestamp_s": float(timestamp_s),
            "stage": stage,
        }
        if not np.isfinite([width_m, timestamp_s]).all():
            # Keep all persisted diagnostics valid JSON.
            evidence = {
                k: (v if not isinstance(v, float) or np.isfinite(v) else None)
                for k, v in evidence.items()
            }
            raise GraspFeedbackError(
                "夹爪反馈无效，保持当前姿态。", "grip_feedback_invalid", evidence
            )
        if width_m < self.config.get("empty_width_m", 0.006):
            code = "grip_lost"
        elif abs(width_m - self.reference) > self.config.get("width_change_m", 0.008):
            code = "grip_unstable"
        else:
            self.bad_since = self.bad_code = None
            return
        if code != self.bad_code:
            self.bad_code, self.bad_since = code, timestamp_s
        duration = timestamp_s - self.bad_since
        if duration + 1e-9 >= self.config.get("loss_debounce_s", 0.10):
            evidence["abnormal_duration_s"] = float(duration)
            raise GraspFeedbackError(
                "夹爪闭合到空夹范围，疑似滑落；停止搬运并重新观察。"
                if code == "grip_lost"
                else "夹持宽度持续变化，夹持状态不确定；保持姿态等待处理。",
                code,
                evidence,
                retryable=(code == "grip_lost"),
            )
