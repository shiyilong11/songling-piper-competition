"""Sensor-grounded single-object pick/place demo. No simulator object poses in planning."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from piper_sim.calibration import camera_to_base
from piper_sim.control import target_to_action
from piper_sim.grasp_feedback import GraspEmptyError, GraspFeedbackError, WidthGuard
from piper_sim.traditional import IK


@dataclass
class TargetEstimate:
    center: np.ndarray
    height: float
    yaw: float
    mask: np.ndarray
    points: np.ndarray
    pixel: list
    snap_distance_px: float
    appearance_rgb: np.ndarray | None = None
    grasp_geometry: dict | None = None


def candidates_from_rgbd(
    obs, calibration, table_top, workspace=None, excluded_regions=None, instance_mask=None
):
    """Visible connected geometry above a configured table; no hidden-instance labels."""
    height, width = obs.rgb.shape[:2]
    v, u = np.indices((height, width))
    z = obs.depth_m
    k = obs.intrinsics
    valid = np.isfinite(z) & (z > 0)
    points = np.stack([(u - k["cx"]) * z / k["fx"], (v - k["cy"]) * z / k["fy"], z], -1)
    base = np.zeros_like(points)
    base[valid] = camera_to_base(points[valid], obs.T_base_end, calibration)
    foreground = valid & (base[..., 2] > table_top + 0.007) & (base[..., 2] < table_top + 0.14)
    workspace = workspace or {"x_m": [0.20, 0.63], "y_m": [-0.14, 0.25]}
    for axis, name in enumerate(("x_m", "y_m")):
        foreground &= (base[..., axis] > workspace[name][0]) & (
            base[..., axis] < workspace[name][1]
        )
    for region in (excluded_regions or {}).values():
        center, half = np.array(region["center_xy_m"]), np.array(region["half_size_xy_m"])
        foreground &= ~np.all(np.abs(base[..., :2] - center) <= half, axis=-1)
    if instance_mask is not None:
        mask = np.asarray(instance_mask)
        if mask.dtype != bool or mask.shape != foreground.shape:
            raise ValueError("Instance mask must be boolean and aligned to the depth image")
        foreground &= mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8))
    result = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < 60:
            continue
        mask = labels == label
        cloud = base[mask]
        hi = np.percentile(cloud, 98, axis=0)
        top = float(hi[2])
        top_points = cloud[cloud[:, 2] >= top - 0.004]
        xy = np.median(top_points[:, :2], axis=0)
        if len(top_points) < 15:
            continue
        rect = cv2.minAreaRect(top_points[:, :2].astype(np.float32))
        corners = cv2.boxPoints(rect)
        edge = corners[1] - corners[0]
        yaw = (np.arctan2(edge[1], edge[0]) + np.pi / 4) % (np.pi / 2) - np.pi / 4
        ys, xs = np.where(mask)
        result.append(
            TargetEstimate(
                np.r_[xy, (top + table_top) / 2],
                top - table_top,
                float(yaw),
                mask,
                cloud,
                [float(np.mean(xs)), float(np.mean(ys))],
                0,
                np.median(obs.rgb[mask], axis=0).astype(float),
            )
        )
    return result


def select_pixel(candidates, pixel, max_distance_px=18):
    if not candidates:
        raise ValueError("当前腕部深度图中没有可抓取候选；请换布局或观察姿态。")
    scores = []
    for candidate in candidates:
        y, x = np.where(candidate.mask)
        scores.append(float(np.hypot(x - pixel[0], y - pixel[1]).min()))
    index = int(np.argmin(scores))
    if scores[index] > max_distance_px:
        raise ValueError("模型选点离任何有效深度候选过远，停止执行。")
    target = candidates[index]
    target.snap_distance_px = scores[index]
    return target


def select_rule(candidates, rgb, instruction):
    """Explicit optional CPU baseline, not a substitute presented as Qwen."""
    import re

    colors = {"红": ("red", 0), "绿": ("green", 1), "蓝": ("blue", 2), "黄": ("yellow", None)}
    match = re.fullmatch(
        r"\s*把([红绿蓝黄])色?方块放(?:到|进|入)?([红绿蓝黄])色?(?:托盘|篮子|目标区|区域)(?:里|里面)?[。.!！]?\s*",
        instruction,
    )
    if not match:
        raise ValueError(
            "CPU规则模式仅支持“把红色方块放到蓝色托盘”这一类明确颜色指令；自然语言请选千问。"
        )
    source, destination = match.groups()
    channel = colors[source][1]
    options = []
    for candidate in candidates:
        color = np.median(rgb[candidate.mask], axis=0).astype(float)
        if source == "黄":
            matches = (
                min(color[0], color[1]) > max(40, color[2] * 1.5)
                and max(color[:2]) < min(color[:2]) * 1.8
            )
        else:
            other = np.delete(color, channel).max()
            matches = color[channel] > other * 1.25 and color[channel] > 40
            # Yellow's red channel can dominate under lighting; exclude it from red.
            if source == "红":
                matches &= color[1] < color[0] * 0.55
        if matches:
            options.append(candidate)
    if len(options) != 1:
        raise ValueError("当前画面中没有唯一的指定颜色方块；请调整观察姿态。")
    return options[0], colors[destination][0]


class PickPlaceSkill:
    ik_type = IK

    def __init__(self, backend, table_top, callback, config, stepper=None):
        self.b, self.table, self.callback, self.cfg = backend, table_top, callback, config
        self.stepper = stepper or backend.step
        self.stage = "planning"
        self.started = backend.data.time
        self.guard = None
        self.feedback_events = []
        self.feedback_cfg = self.cfg.get("grasp_feedback", {})
        self.last_plan = None

    def feedback(self, event, **values):
        self.feedback_events.append({"event": event, "time_s": float(self.b.data.time), **values})

    def hold_current(self):
        """Cancel remaining joint motion while keeping the gripper commanded closed."""
        measured = self.b.measured()
        q = measured[:6] if np.isfinite(measured[:6]).all() else self.b.previous_target[:6]
        width = self.b.previous_target[6] if self.cfg.get("preserve_gripper_on_hold") else 0.0
        goal = np.r_[q, width]
        self.b.previous_target = goal.copy()
        self.b._set_target(goal)

    def recover_empty(self):
        """After suspected empty/lost grasp, retract before opening or changing view."""
        self.guard = None
        if self.last_plan is None:
            return
        self.stage = "recover_lift"
        point = self.b.data.site_xpos[self.b.model.site("tcp").id].copy()
        point[2] = max(point[2], self.cfg.get("transport_height_m", 0.22))
        self.cartesian(point, self.last_plan["yaw"], 0.0)
        self.move(self.b.measured()[:6], 0.07)
        self.feedback("empty_recovery_retracted")

    def recenter(self, observation, points):
        from piper_sim.active_view import recenter_camera

        return recenter_camera(
            self.b, observation, points, self.table, self.cfg.get("active_view", {})
        )

    def tick(self, goal):
        if self.b.data.time - self.started > self.cfg.get("max_task_s", 120):
            raise TimeoutError("抓放技能超过时间预算")
        action = target_to_action(goal, self.b.previous_target, self.b.interface)
        self.stepper(action)
        if self.guard is not None:
            try:
                self.guard.update(float(self.b.measured()[6]), float(self.b.data.time), self.stage)
            except GraspFeedbackError as exc:
                self.hold_current()
                self.guard = None
                self.feedback("retention_failed", code=exc.code, **exc.evidence)
                raise
        self.callback(self.stage)

    def move(self, q, width):
        start = self.b.data.time
        while True:
            self.tick(np.r_[q, width])
            grip_ok = (
                self.b.previous_target[6] < 0.004
                if width == 0
                else abs(self.b.measured()[6] - width) < 0.004
            )
            if np.max(np.abs(self.b.measured()[:6] - q)) < 0.010 and grip_ok:
                return
            if self.b.data.time - start > 18:
                raise TimeoutError(f"{self.stage}: 关节未在限时内到达目标")

    def hold(self, seconds):
        for _ in range(int(np.ceil(seconds * self.b.interface["control"]["policy_hz"]))):
            self.tick(self.b.previous_target.copy())

    def cartesian(self, destination, yaw, width):
        current = self.b.data.site_xpos[self.b.model.site("tcp").id].copy()
        steps = max(1, int(np.ceil(np.linalg.norm(destination - current) / 0.02)))
        for position in np.linspace(current, destination, steps + 1)[1:]:
            self.move(self.ik.solve(position, yaw, self.b.measured()[:6]), width)

    def plan(self, target, destination, tray_floor):
        errors = []
        for tilt in self.cfg.get("grasp_tilt_candidates_rad", [0.6, 0.9, 0.35, 0.15]):
            for yaw in self.cfg.get(
                "grasp_yaw_candidates_rad",
                [
                    target.yaw + np.pi,
                    np.pi,
                    target.yaw + np.pi / 2,
                    target.yaw,
                    target.yaw - np.pi / 2,
                ],
            ):
                self.ik = self.ik_type(self.b, tilt=tilt)
                rot = (
                    Rotation.from_euler("z", yaw).as_matrix()
                    @ Rotation.from_euler("y", tilt).as_matrix()
                    @ np.diag([1.0, -1.0, -1.0])
                )
                lowest = (rot @ [0, 0, 0.1088 - 0.122])[2] - (np.abs(rot) @ [0.022, 0.003, 0.023])[
                    2
                ]
                clearance = self.cfg.get("extra_table_clearance_m", 0.0)
                center = target.center.copy()
                center[2] = max(center[2], self.table - lowest + 0.001 + clearance)
                hover = center.copy()
                hover[2] = self.cfg.get("transport_height_m", 0.22)
                place = np.r_[
                    destination,
                    max(tray_floor + target.height / 2 + 0.003, tray_floor - lowest + 0.002)
                    + clearance
                    + self.cfg.get("release_clearance_m", 0.0),
                ]
                above = place.copy()
                above[2] = hover[2]
                try:
                    seed = self.b.measured()[:6]
                    for point in [hover, center, above, place]:
                        seed = self.ik.solve(point, yaw, seed)
                except RuntimeError as exc:
                    errors.append(str(exc))
                    continue
                break
            else:
                continue
            break
        else:
            raise RuntimeError("没有找到可达抓放姿态：" + errors[-1])
        return {
            "center": center,
            "hover": hover,
            "place": place,
            "above": above,
            "yaw": yaw,
            "tilt": tilt,
            "height": target.height,
        }

    def pick(self, plan):
        self.guard = None
        self.last_plan = plan
        center, hover, yaw = plan["center"], plan["hover"], plan["yaw"]
        self.stage = "approach"
        self.move(self.ik.solve(hover, yaw, self.b.measured()[:6]), 0.07)
        self.stage = "descend"
        self.cartesian(center, yaw, 0.07)
        self.stage = "close"
        self.move(self.b.measured()[:6], 0)
        self.hold(0.3)
        self.stage = "check_grip"
        widths = []
        for _ in range(
            max(
                2,
                int(
                    np.ceil(
                        self.feedback_cfg.get("close_check_s", 0.20)
                        * self.b.interface["control"]["policy_hz"]
                    )
                ),
            )
        ):
            self.tick(self.b.previous_target.copy())
            widths.append(float(self.b.measured()[6]))
        if not np.isfinite(widths).all():
            self.hold_current()
            raise GraspFeedbackError("夹爪反馈无效，保持当前姿态。", "grip_feedback_invalid")
        width = float(np.median(widths))
        evidence = {
            "width_m": width,
            "width_span_m": float(np.ptp(widths)),
            "source": "measured_gripper_width",
        }
        self.feedback("closure_checked", **evidence)
        if min(widths) < self.feedback_cfg.get("empty_width_m", 0.006):
            self.hold_current()
            raise GraspEmptyError(evidence=evidence)
        if np.ptp(widths) > self.feedback_cfg.get("close_stability_m", 0.004):
            self.hold_current()
            raise GraspFeedbackError("闭合后夹持宽度不稳定，暂停抬升。", "grip_unstable", evidence)
        self.guard = WidthGuard(width, self.feedback_cfg)
        self.stage = "test_lift"
        test_lift = center.copy()
        test_lift[2] = min(hover[2], center[2] + self.feedback_cfg.get("test_lift_m", 0.04))
        self.cartesian(test_lift, yaw, 0)
        self.hold(self.feedback_cfg.get("test_lift_hold_s", 1.0))
        self.feedback(
            "test_lift_retained",
            width_m=float(self.b.measured()[6]),
            evidence_status="width_based_estimate_not_contact_proof",
        )
        self.stage = "lift"
        self.cartesian(hover, yaw, 0)
        self.hold(0.2)
        return {
            "gripper_width_m": float(self.b.measured()[6]),
            "holding_estimate": True,
            "evidence_status": "width_based_estimate_not_contact_proof",
            "feedback_events": list(self.feedback_events),
        }

    def descend_for_release(self, place, yaw):
        self.cartesian(place, yaw, 0)

    def retreat_after_release(self, above, yaw, approach_q, guarded_release):
        self.cartesian(above, yaw, 0.07)

    def place(self, plan):
        place, above, yaw = plan["place"], plan["above"], plan["yaw"]
        self.stage = "transport"
        self.cartesian(above, yaw, 0)
        approach_q = self.b.measured()[:6].copy()
        self.stage = "place"
        guarded_release = self.descend_for_release(place, yaw)
        # Intentional release must not be interpreted as slip.
        self.feedback("retention_monitor_finished", width_m=float(self.b.measured()[6]))
        self.guard = None
        self.stage = "release"
        self.move(self.b.measured()[:6], 0.07)
        self.hold(0.3)
        self.stage = "retreat"
        self.retreat_after_release(above, yaw, approach_q, guarded_release)
        self.hold(self.cfg.get("post_retreat_wait_s", 2.2))
        self.stage = "verify"
        return {
            "planned_center_m": plan["center"].tolist(),
            "planned_destination_m": place.tolist(),
            "target_height_estimate_m": plan["height"],
            "grasp_tilt_rad": plan["tilt"],
            "yaw_rad": yaw,
            "feedback_events": list(self.feedback_events),
        }

    def run(self, target, destination, tray_floor):
        plan = self.plan(target, destination, tray_floor)
        self.pick(plan)
        return self.place(plan)
