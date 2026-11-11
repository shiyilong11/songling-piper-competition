"""Sensor-only engineering baseline for the fixed exploration instructions."""

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from piper_sim.competition_sensor import PlacementCheck, ZoneEstimate
from piper_sim.demo_skill import TargetEstimate
from piper_sim.grasp_feedback import GraspFeedbackError, match_fresh_target
from piper_sim.l2_sensor_controller import L2SensorController
from piper_sim.observation import Observation
from piper_sim.scene_memory import SceneMemory, load_memory_config
from piper_sim.scene_observation import (
    CalibrationEstimate,
    InstanceMask,
    PlacementRegion,
    Provenance,
)

CARDS = {
    1: ("把红色方块放进红色目标区", (("cube_red", "red_region"),)),
    2: ("把最左边的柿子放进绿色目标区", (("persimmon", "green_region"),)),
    3: ("把左边的土豆放进右侧目标区", (("potato", "rightmost"),)),
    4: (
        "把桌上的红、绿、黄三个方块按颜色分别放进同色目标区",
        (
            ("cube_red", "red_region"),
            ("cube_green", "green_region"),
            ("cube_yellow", "yellow_region"),
        ),
    ),
    5: (
        "将红色方块与香蕉分别放入红色与黄色目标区",
        (("cube_red", "red_region"), ("banana", "yellow_region")),
    ),
}
REGION_IDS = {"red_region", "green_region", "yellow_region"}


def parse_round(instruction):
    lookup = {text: number for number, (text, _) in CARDS.items()}
    lines = instruction.replace("；", "\n").replace(";", "\n").splitlines()
    try:
        numbers = tuple(
            lookup["".join(line.split()).rstrip("。.!！")] for line in lines if line.strip()
        )
    except KeyError as exc:
        raise ValueError("请从五条正式指令中选择 1 至 3 条。") from exc
    if not 1 <= len(numbers) <= 3 or len(set(numbers)) != len(numbers):
        raise ValueError("每次提交 1 至 3 条不同的正式指令。")
    return numbers


def compile_round(numbers, snapshot, left_axis, *, gap_m=0.02):
    """Resolve the published task card against measured, model-labelled instances."""
    axis = np.asarray(left_axis, dtype=float)
    if axis.shape != (2,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-6:
        raise ValueError("需要裁判初始视角在机器人坐标中的有效左向。")
    axis /= np.linalg.norm(axis)
    objects = [
        obj
        for obj in snapshot["objects"]
        if obj["identity_status"] == "estimated" and obj["evidence_frame_ids"]
    ]
    regions = {
        row["region_id"]: row
        for row in snapshot["regions"]
        if row["region_id"] in REGION_IDS and row["evidence_frame_ids"]
    }
    result = []
    for number in numbers:
        instruction, roles = CARDS[number]
        actions, reason = [], None
        for label, destination in roles:
            matches = [obj for obj in objects if obj["label_hint"] == label]
            if number in (2, 3):
                matches.sort(
                    key=lambda row: np.asarray(row["last_observed_center_base_m"])[:2] @ axis,
                    reverse=True,
                )
                if len(matches) < 2 or (
                    (
                        np.asarray(matches[0]["last_observed_center_base_m"])[:2]
                        - np.asarray(matches[1]["last_observed_center_base_m"])[:2]
                    )
                    @ axis
                    < gap_m
                ):
                    reason = "需要补看同类目标或确认裁判视角下的左右关系。"
                    break
            elif len(matches) != 1:
                name = {
                    "cube_red": "红色方块",
                    "cube_green": "绿色方块",
                    "cube_yellow": "黄色方块",
                    "banana": "香蕉",
                }[label]
                reason = f"未能唯一确认{name}（已确认 {len(matches)} 个候选）。"
                break
            if destination == "rightmost":
                if regions.keys() != REGION_IDS:
                    reason = "需要观察三个完整目标区后再确定右侧区域。"
                    break
                projected = sorted(
                    (np.asarray(row["T_base_region"])[:2, 3] @ axis, key)
                    for key, row in regions.items()
                )
                if projected[1][0] - projected[0][0] < gap_m:
                    reason = "裁判视角下的右侧区域尚不能区分。"
                    break
                destination = projected[0][1]
            if destination not in regions:
                reason = "尚未观测到完整目标区。"
                break
            actions.append(
                {
                    "target_instance_id": matches[0]["instance_id"],
                    "category": label,
                    "destination_region_id": destination,
                }
            )
        result.append(
            {
                "number": number,
                "instruction": instruction,
                "required_actions": len(roles),
                "actions": actions if reason is None else [],
                "verified_actions": 0,
                "status": "planned" if reason is None else "needs_observation",
                "reason": reason,
            }
        )
    return result


def as_region(zone):
    pose = np.eye(4)
    pose[:3, 3] = np.r_[zone.center, zone.floor_m]
    edge = zone.corners[1] - zone.corners[0]
    pose[:2, 0] = edge / np.linalg.norm(edge)
    pose[:2, 1] = [-pose[1, 0], pose[0, 0]]
    return PlacementRegion(
        zone.color + "_region",
        pose,
        np.linalg.norm(np.roll(zone.corners, -1, axis=0) - zone.corners, axis=1)[:2] / 2,
        Provenance("sensor_estimate", "RGB-D complete square target-region estimate"),
        timestamp_s=zone.timestamp_s,
    )


@dataclass
class SurveyView:
    joints: np.ndarray
    targets: dict


@dataclass(frozen=True)
class TransferRequest:
    target_instance_id: str
    category: str
    observation: Observation
    target: TargetEstimate
    destination: ZoneEstimate
    attempt: int
    reserve_lateral_strip: bool


class RetryableEmptyTransfer(ValueError):
    """The grasp was empty and the arm already recovered for a new observation."""


class TransferExecutor(Protocol):
    def __call__(self, request: TransferRequest) -> np.ndarray:
        """Release the target and return its estimated base-frame centre in metres."""
        ...


def preserve_square_evidence(snapshot, views):
    """Prefer a measured, complete square face over a later partial crop."""
    squares = {}
    for frame_id, view in views.items():
        for key, (label, target) in view.targets.items():
            if label.startswith("cube_"):
                squares.setdefault(key, []).append((frame_id, label, target))
    for row in snapshot["objects"]:
        observations = squares.get(row["instance_id"], [])
        if not observations:
            continue
        if len({label for _, label, _ in observations}) != 1:
            row["identity_status"] = "ambiguous"
            continue
        _, label, target = observations[-1]
        row.update(
            label_hint=label,
            label_hint_source=Provenance("sensor_estimate", "complete RGB-D square face").as_dict(),
            last_observed_center_base_m=target.center.tolist(),
            evidence_frame_ids=[frame_id for frame_id, _, _ in observations],
        )


class ExplorationController(L2SensorController):
    """Interpret the task card, then plan and verify using public sensors."""

    def __init__(
        self, *args, left_axis, show_masks, publish,
        transfer_executor: TransferExecutor | None = None, **kwargs
    ):
        super().__init__(*args, method="yoloe_sam2", **kwargs)
        self.calibration = CalibrationEstimate(
            self.scene.calibration,
            Provenance("manual_config", "Configured wrist-camera hand-eye calibration"),
        )
        config = load_memory_config()
        config.update(
            left_axis_base=[*left_axis, 0.0],
            coordinate_source="configured simulated judge viewpoint",
            required_views=["left", "center", "right"],
            memory_max_age_s=65.0,
            association_max_age_s=60.0,
        )
        self.memory = SceneMemory(config)
        self.left_axis, self.show_masks, self.publish = left_axis, show_masks, publish
        self.transfer_executor = transfer_executor
        self.views = {}
        self.zone_evidence = {}
        self.zone_observation_poses = {}
        self.release_counts = Counter()
        self.unconfirmed_release_zones = set()
        self.initial_survey = None

    def observe(self):
        obs, blocks, zones = super().observe()
        for color in zones:
            self.zone_evidence[color] = f"sensors/{self.frames:04d}.jpg"
            poses = self.zone_observation_poses.setdefault(color, [])
            joints = obs.joint_position_rad
            if all(np.linalg.norm(joints - q) > 0.10 for q in poses):
                poses.append(joints.copy())
        cubes = []
        for color, block in blocks:
            top = np.percentile(block.points[:, 2], 98)
            face = block.points[block.points[:, 2] >= top - 0.004, :2].astype(np.float32)
            sides = cv2.minAreaRect(face)[1]
            area = cv2.contourArea(cv2.convexHull(face))
            if min(sides) >= 0.039 and max(sides) <= 0.061 and area >= 0.90 * np.prod(sides):
                cubes.append((color, block))
        return obs, cubes, zones

    def scan_view(self, name, joints):
        self.motion.stage = "scan"
        self.motion.move(np.asarray(joints), 0.07)
        self.motion.hold(0.25)
        obs, blocks, zones = self.observe()
        self.callback("understand")
        proposals = self.detect(obs)
        targets = {
            f"block_{i}": ("cube_" + color, target) for i, (color, target) in enumerate(blocks)
        }
        for proposal in proposals:
            if proposal.color not in self.cfg["l2_color_mapping"]:
                continue
            if any(
                np.linalg.norm(proposal.target.center - block.center) < 0.025 for _, block in blocks
            ):
                continue
            targets[proposal.key] = proposal.color, proposal.target
        detections = [
            InstanceMask(
                key,
                target.mask,
                obs.timestamp_s,
                1.0,
                Provenance(
                    "sensor_estimate",
                    "RGB-D square face and colour"
                    if label.startswith("cube_")
                    else "RGB instance mask with offline Qwen category estimate",
                ),
                label,
            )
            for key, (label, target) in targets.items()
        ]
        frame = self.memory.update(
            obs,
            self.calibration,
            detections=detections,
            view_id=name,
            regions=[as_region(zone) for zone in zones.values()],
            allow_geometric_proposals=False,
        )
        self.views[frame.frame_id] = SurveyView(
            obs.joint_position_rad.copy(),
            {frame.instances[k]: value for k, value in targets.items() if k in frame.instances},
        )
        self.show_masks(
            obs,
            [
                (label, None, target.mask)
                for label, target in targets.values()
                if label in self.requested_labels
            ],
        )

    def source_view(self, action):
        key = action["target_instance_id"]
        for view in reversed(self.views.values()):
            if key in view.targets and view.targets[key][0] == action["category"]:
                return view
        raise ValueError("没有已观测到的同类实例可供复查。")

    def reacquire(self, action):
        source = self.source_view(action)
        self.motion.stage = "observe"
        self.motion.move(source.joints, 0.07)
        self.motion.hold(0.25)
        reference = source.targets[action["target_instance_id"]][1]
        obs, blocks, _ = self.observe()
        color = action["category"]
        if color.startswith("cube_"):
            options = [target for label, target in blocks if label == color[5:]]
            target, _ = match_fresh_target(reference, options, self.motion_cfg["grasp_feedback"])
            axis = np.array([np.cos(target.yaw), np.sin(target.yaw), 0.0])
            target.grasp_geometry = {
                "source": "measured RGB-D square face and parallel edges",
                "major_axis_reliable": True,
                "metric_major_axis_base": axis.tolist(),
                "candidates": [
                    {"position_base_m": target.center.tolist(), "yaw_rad": yaw}
                    for yaw in (target.yaw, target.yaw + np.pi / 2)
                ],
            }
            return obs, target
        options = [
            p.target for p in self.detect(obs, focus=[(reference.center, 0.10)]) if p.color == color
        ]
        target, _ = match_fresh_target(reference, options, self.motion_cfg["grasp_feedback"])
        return obs, target

    def verify_release(self, action, expected, reference):
        destination = action["destination_region_id"].removesuffix("_region")
        poses = [self.zone_views[destination]]
        for q in reversed(self.zone_observation_poses[destination]):
            if all(np.linalg.norm(q - old) > 0.10 for old in poses):
                poses.append(q)
        for q in poses[:3]:
            try:
                return self.verify_release_from_view(action, expected, reference, q)
            except ValueError as exc:
                reason = str(exc)
                self.events.append({"event": "l3_verify_view_rejected", **action, "reason": reason})
        raise ValueError(reason)

    def verify_release_from_view(self, action, expected, reference, joints):
        destination = action["destination_region_id"].removesuffix("_region")
        label = action["category"]
        cube = label.startswith("cube_")
        self.motion.stage = "reobserve_place"
        self.motion.move(joints, 0.07)
        self.motion.hold(0.25)
        obs, blocks, _ = self.observe()
        if cube:
            candidates = []
            margin = self.motion_cfg["grasp_feedback"].get("image_border_margin_px", 4)
            for color, target in blocks:
                if color != label[5:]:
                    continue
                y, x = np.where(target.mask)
                h, w = target.mask.shape
                if (
                    min(x.min(), y.min()) >= margin
                    and x.max() < w - margin
                    and y.max() < h - margin
                ):
                    candidates.append(target)
        else:
            candidates = [
                p.target
                for p in self.detect(obs, focus=[(expected, 0.15)])
                if p.color == label
                and self.progress.compatible(
                    {
                        "color": label,
                        "destination": destination,
                        "appearance": reference.appearance_rgb,
                    },
                    p.color,
                    p.target,
                )
            ]
        nearby = sorted(
            (np.linalg.norm(target.center[:2] - expected[:2]), i)
            for i, target in enumerate(candidates)
        )
        if (
            not nearby
            or nearby[0][0] > 0.12
            or (len(nearby) > 1 and nearby[1][0] - nearby[0][0] < 0.025)
        ):
            raise ValueError("释放后未能唯一确认目标。")
        reference = candidates[nearby[0][1]]
        check = PlacementCheck(self.scene.zones[destination], cube=cube)
        self.motion.stage = "verify"
        for _ in range(8):
            if cube:
                current, blocks, _ = self.observe()
                targets = [target for color, target in blocks if color == label[5:]]
            else:
                current, targets = self.visible_geometry()
            try:
                target, _ = match_fresh_target(
                    reference, targets, self.motion_cfg["grasp_feedback"]
                )
            except GraspFeedbackError:
                target = None
            accepted = check.update(target, current.timestamp_s, current.gripper_width_m)
            self.events.append(
                {
                    "event": "l3_placement_observation",
                    **action,
                    "time_s": current.timestamp_s,
                    "accepted": accepted,
                    "stable_s": 0 if check.start is None else check.last - check.start,
                    "center_m": None if target is None else target.center.tolist(),
                    "observed_footprint_inside": target is not None
                    and self.scene.zones[destination].contains(target.points[:, :2], 0.012),
                }
            )
            if accepted:
                self.show_masks(current, [(label, destination, target.mask)])
                return target
            if target is not None:
                reference = target
            self.motion.hold(0.4)
        raise ValueError("未观测到完整入区且持续稳定两秒的证据。")

    def execute_ik_transfer(self, request, action):
        destination = request.destination.color
        target = request.target
        planning = {
            "category": request.category,
            "prefer_near_release": True,
            "reserve_lateral_strip": request.reserve_lateral_strip,
        }
        try:
            plan = self.make_plan(target, destination, **planning)
        except ValueError:
            self.events.append(
                {
                    "event": "l3_plan_rejected",
                    **action,
                    "attempt": request.attempt,
                    "diagnostics": self.last_plan_diagnostics,
                }
            )
            if np.linalg.norm(request.destination.center) <= 0.58:
                raise
            # Retry the outer zone with a lower approach and full path checks.
            self.motion.cfg["release_hover_above_table_m"] = 0.16
            try:
                plan = self.make_plan(target, destination, budget_s=10.0, **planning)
            finally:
                self.motion.cfg.pop("release_hover_above_table_m", None)
        _, candidates = self.visible_geometry()
        match_fresh_target(target, candidates, self.motion_cfg["grasp_feedback"])
        self.events.append(
            {
                "event": "l3_pick_place_plan",
                **action,
                "attempt": request.attempt,
                "pick_m": plan["center"].tolist(),
                "release_m": plan["place"].tolist(),
            }
        )
        action["plan"] = {
            "pick_m": plan["center"].tolist(),
            "release_m": plan["place"].tolist(),
        }
        action["attempts"] = request.attempt
        self.publish(self.cards)
        try:
            grip = self.motion.pick_closed_loop(plan)
        except GraspFeedbackError as exc:
            if not exc.retryable:
                raise
            self.motion.recover_empty()
            raise RetryableEmptyTransfer(str(exc)) from exc
        self.events.append(
            {
                "event": "l3_grip_estimate",
                **action,
                "evidence_status": grip["evidence_status"],
                "official_lift_verified": False,
            }
        )
        self.motion.place_closed_loop(plan)
        return plan.get("object_destination_estimate_m", plan["place"])

    def carry(self, action):
        color = action["category"]
        destination = action["destination_region_id"].removesuffix("_region")
        leave_banana_space = color != "banana" and any(
            other["category"] == "banana"
            and other["destination_region_id"] == action["destination_region_id"]
            and other.get("status") != "verified"
            for card in self.cards
            for other in card["actions"]
        )
        for attempt in (1, 2):
            obs, target = self.reacquire(action)
            self.show_masks(obs, [(color, destination, target.mask)])
            self.callback("planning")
            request = TransferRequest(
                action["target_instance_id"], color, obs, target, self.scene.zones[destination],
                attempt, leave_banana_space,
            )
            try:
                if self.transfer_executor is None:
                    expected = self.execute_ik_transfer(request, action)
                else:
                    action["attempts"] = attempt
                    self.publish(self.cards)
                    expected = self.transfer_executor(request)
            except RetryableEmptyTransfer as exc:
                if attempt == 2:
                    raise
                self.events.append({"event": "l3_pick_retry", **action, "reason": str(exc)})
                continue
            self.release_counts[destination] += 1
            try:
                expected = np.asarray(expected, dtype=float)
                if expected.shape != (3,) or not np.isfinite(expected).all():
                    raise ValueError("执行器未返回有效的释放位置估计。")
                return self.verify_release(action, expected, target), attempt
            except ValueError:
                self.unconfirmed_release_zones.add(destination)
                raise
        raise RuntimeError("未取得抓取结果。")

    def run_round(self, numbers, poses, zone_poses=()):
        completed = {}
        self.requested_labels = {label for number in numbers for label, _ in CARDS[number][1]}
        for name, joints in poses.items():
            self.scan_view(name, joints)
        # Region observations use measured bounds. The inherited search only
        # moves the wrist to collect evidence when a full card is still missing.
        if set(self.zone_views) != {"red", "green", "yellow"} and zone_poses:
            self.cfg["observation_q_rad"] = zone_poses
            try:
                self.scan()
            except ValueError as exc:
                self.events.append({"event": "l3_zone_survey_incomplete", "reason": str(exc)})
        self.initial_survey = self.memory.summary(float(self.robot.data.time))
        preserve_square_evidence(self.initial_survey, self.views)
        self.initial_survey["regions"] = [
            {**as_region(zone).as_dict(), "evidence_frame_ids": [self.zone_evidence[color]]}
            for color, zone in self.scene.zones.items()
        ]
        cards = compile_round(numbers, self.initial_survey, self.left_axis)
        self.cards = cards
        self.publish(cards)
        for card in cards:
            if not card["actions"]:
                continue
            card["status"] = "executing"
            self.publish(cards)
            for action in card["actions"]:
                key = action["target_instance_id"]
                destination = action["destination_region_id"].removesuffix("_region")
                try:
                    if key in completed:
                        previous, target = completed[key]
                        if previous != destination:
                            raise ValueError("同一目标的两个目的地冲突，需要裁判确认。")
                        target = self.verify_release(action, target.center, target)
                        action["reused_observed_placement"] = True
                    else:
                        if destination in self.unconfirmed_release_zones:
                            raise ValueError("目标区内上次释放仍未确认，暂缓继续投放。")
                        target, action["attempts"] = self.carry(action)
                    self.reserve_observed_footprint(
                        {"id": key, "color": action["category"], "destination": destination}, target
                    )
                    completed[key] = (destination, target)
                    action["status"] = "verified"
                    card["verified_actions"] += 1
                except ValueError as exc:
                    action.update(status="not_verified", reason=str(exc))
                self.publish(cards)
            card["status"] = (
                "verified"
                if card["verified_actions"] == card["required_actions"]
                else "not_verified"
            )
            self.publish(cards)
        self.recheck_shared_regions(cards, completed)
        return cards

    def recheck_shared_regions(self, cards, completed):
        actions_by_target = {}
        for card in cards:
            for action in card["actions"]:
                if action.get("status") == "verified":
                    actions_by_target.setdefault(action["target_instance_id"], []).append(action)
        for key, (destination, target) in completed.items():
            if self.release_counts[destination] < 2:
                continue
            actions = actions_by_target[key]
            try:
                self.verify_release(actions[0], target.center, target)
            except ValueError as exc:
                for action in actions:
                    action.update(status="not_verified", reason=f"终检：{exc}")
        for card in cards:
            if card["actions"]:
                card["verified_actions"] = sum(
                    a.get("status") == "verified" for a in card["actions"]
                )
                card["status"] = (
                    "verified"
                    if card["verified_actions"] == card["required_actions"]
                    else "not_verified"
                )
        self.publish(cards)
