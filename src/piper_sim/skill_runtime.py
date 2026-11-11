"""Nine explicit manipulation skills and a bounded recovery scheduler.

Sensor observations flow through perception and planning. Simulator ground truth
is accessible only through the separate evaluator binding/verification port.
"""

import re
import time
from dataclasses import dataclass, field

import numpy as np

from piper_sim.demo_skill import candidates_from_rgbd, select_pixel, select_rule
from piper_sim.grasp_feedback import GraspFeedbackError, match_fresh_target


def destination_text(instruction):
    """A conservative preflight for explicit colors and unqualified tray phrases.

    Other descriptions still go to visual interpretation; this is not a replacement NLU.
    Color words in the source object's description must not choose the destination.
    """
    tail = re.split(r"(?:放(?:置)?|投放|送|移|摆)(?:到|进|入|在)?", instruction, maxsplit=1)
    if len(tail) != 2:
        return None, False
    destination = tail[1]
    found = {
        color
        for word, color in (("红", "red"), ("蓝", "blue"), ("绿", "green"), ("黄", "yellow"))
        if re.search(word + r"色?(?:的)?(?:托盘|篮子|容器|盘子|目标区|区域)", destination)
    }
    if len(found) == 1:
        return found.pop(), False
    bare = re.fullmatch(
        r"\s*(?:托盘|篮子|容器|盘子|目标区|区域)(?:里面|里|中|内)?[。.!！\s]*", destination
    )
    return None, bool(bare) or len(found) > 1


@dataclass
class SkillResult:
    status: str
    data: dict = field(default_factory=dict)
    code: str | None = None
    message: str = ""
    summary: dict = field(default_factory=dict)


class ManipulationSkills:
    names = (
        "observe_scene",
        "detect_objects",
        "interpret_task",
        "resolve_target",
        "resolve_destination",
        "pick_object",
        "place_object",
        "verify_result",
        "recover_or_continue",
    )

    def __init__(self, port, motion):
        self.port, self.motion = port, motion
        self.trace = []
        self.pending_view_q = None
        self.recenter_attempts = 0
        self.observe_current_once = getattr(port, "start_from_current", False)
        self.views = [port.settings["observation_q_rad"]] + port.cfg.get(
            "observation_search_q_rad", []
        )

    def call(self, name, **arguments):
        if name not in self.names:
            raise ValueError("Unknown skill")
        self.port.checkpoint()
        self.port.update(active_skill=name)
        record = {"skill": name, "status": "running", "started_s": time.monotonic()}
        self.trace.append(record)
        self.port.update(skill_trace=[dict(row) for row in self.trace])
        try:
            result = getattr(self, name)(**arguments)
        except BaseException:
            record.update(
                status="interrupted", elapsed_s=round(time.monotonic() - record["started_s"], 3)
            )
            self.port.update(skill_trace=[dict(row) for row in self.trace])
            raise
        record.update(
            status=result.status,
            code=result.code,
            message=result.message,
            output=result.summary,
            elapsed_s=round(time.monotonic() - record["started_s"], 3),
        )
        self.port.update(skill_trace=[dict(row) for row in self.trace])
        return result

    def observe_scene(self, view_index):
        active = self.pending_view_q is not None
        current = self.observe_current_once and not active
        self.observe_current_once = False
        self.port.stage(
            "observe",
            "观察技能：将目标移向画面中心，重新采集腕部图像。"
            if active
            else "观察技能：保留现场，从当前机械臂姿态重新拍照。"
            if current
            else f"观察技能：移动到视角 {view_index + 1}/{len(self.views)}，采集新腕部图像。",
        )
        self.motion.stage = "observe"
        q = np.asarray(self.pending_view_q if active else self.views[view_index], dtype=float)
        self.pending_view_q = None
        if not current and np.max(np.abs(self.port.world.backend.measured()[:6] - q)) > 0.01:
            self.motion.move(q, 0.07)
        observation = self.port.capture()
        if current:
            # A new instruction gets the full search budget, even when the robot
            # happens to be at the previous instruction's final preset view.
            view_index = -1
        data = {
            "frame_id": self.port.sequence,
            "observation": observation,
            "view_index": view_index,
        }
        return SkillResult(
            "ok",
            data,
            summary={
                "frame_id": data["frame_id"],
                "view_index": view_index,
                "active_recenter": active,
                "from_current_pose": current,
            },
        )

    def detect_objects(self, observation):
        candidates = candidates_from_rgbd(
            observation["observation"],
            self.port.world.backend.camera,
            self.port.scene_cfg["table"]["top_m"],
            self.port.scene_cfg.get("perception_workspace"),
            self.port.scene_cfg["trays"],
        )
        # IDs are local to this frame. They are deliberately not simulator body IDs.
        data = {"frame_id": observation["frame_id"], "candidates": candidates}
        return SkillResult(
            "ok",
            data,
            summary={
                "frame_id": data["frame_id"],
                "candidate_count": len(candidates),
                "source": "wrist_rgbd_connected_regions",
            },
        )

    def interpret_task(self, request, observation=None, detections=None):
        self.port.stage("understand")
        explicit_destination, needs_choice = destination_text(request["instruction"])
        if request.get("classification"):
            explicit_destination, needs_choice = None, False
        if observation is None:
            if needs_choice:
                choices = list(self.port.scene_cfg["trays"])
                return SkillResult(
                    "needs_clarification",
                    {"field": "destination_color", "choices": choices},
                    code="destination_unspecified",
                    message="桌面有多个托盘，请先选择要放入哪个托盘。",
                    summary={"missing_field": "destination_color", "choices": choices},
                )
            return SkillResult(
                "ok", summary={"instruction_checked": True, "requires_visual_grounding": True}
            )
        if request["backend"] == "rule":
            try:
                target, destination = select_rule(
                    detections["candidates"], observation["observation"].rgb, request["instruction"]
                )
            except ValueError as exc:
                return SkillResult("retryable", code="target_uncertain", message=str(exc))
            response = {"backend": "CPU颜色规则", "inference_executed": False}
            prediction = {
                "status": "ready",
                "target_pixel_xy": target.pixel,
                "destination_color": destination,
            }
        else:
            from piper_sim.competition_demo import COLORS

            colors = "、".join(COLORS[c] for c in self.port.scene_cfg["trays"])
            context = (
                f"仿真场地配置：{colors}目标区存在，使用配置位置；只能选择这些目的地。"
                "只选择当前腕部图中能看到的目标。用户指令：" + request["instruction"]
            )
            response = self.port.infer(
                {"frame_id": observation["frame_id"], "instruction": context}
            )
            self.port.checkpoint()
            self.port.update(
                model_response=response, model_loaded=response.get("model_loaded", False)
            )
            if response.get("error"):
                return SkillResult(
                    "failed", code="model_unavailable", message="模型调用失败：" + response["error"]
                )
            if response.get("validation_error"):
                return SkillResult(
                    "retryable",
                    code="invalid_model_response",
                    message="模型输出格式不完整，重新观察后再试；原始输出保留在详情中。",
                )
            prediction = response["prediction"]
            if prediction["status"] != "ready":
                return SkillResult(
                    "retryable",
                    code=prediction["status"],
                    message="当前视角无法确定目标，需要换视角观察。",
                )
        self.port.update(model_response=response)
        if explicit_destination is not None:
            prediction = {**prediction, "destination_color": explicit_destination}
        return SkillResult(
            "ok",
            {**prediction, "frame_id": observation["frame_id"]},
            summary={**prediction, "frame_id": observation["frame_id"]},
        )

    def resolve_target(self, task, detections, observation):
        self.port.stage("localize")
        if (
            not task["frame_id"]
            == detections["frame_id"]
            == observation["frame_id"]
            == self.port.sequence
        ):
            return SkillResult(
                "failed", code="stale_frame", message="选点与深度图的帧号不一致，拒绝执行。"
            )
        try:
            if self.port.cfg.get("segmentation", {}).get("enabled", False):
                self.port.stage("segment", "SAM分割选中目标，结合当前帧深度定位。")
                target = self.port.refine_target(
                    observation["observation"], task["target_pixel_xy"]
                )
            else:
                target = select_pixel(detections["candidates"], task["target_pixel_xy"])
        except (RuntimeError, TimeoutError, OSError) as exc:
            return SkillResult("failed", code="segmentation_error", message=str(exc))
        except ValueError as exc:
            obs = observation["observation"]
            x, y = np.rint(task["target_pixel_xy"]).astype(int)
            h, w = obs.depth_m.shape
            depth = float(obs.depth_m[y, x]) if 0 <= x < w and 0 <= y < h else float("nan")
            missing = not np.isfinite(depth) or depth <= 0
            detail = {
                "pixel_xy": task["target_pixel_xy"],
                "depth_m": None if missing else depth,
                "candidate_count": len(detections["candidates"]),
            }
            self.port.show_perception(obs, detections["candidates"], task["target_pixel_xy"])
            return SkillResult(
                "retryable",
                code="missing_target_depth" if missing else "point_outside_object",
                message="选中像素没有有效深度，需要改变相机距离后重试。" if missing else str(exc),
                summary=detail,
            )
        y, x = np.where(target.mask)
        h, w = target.mask.shape
        margin = self.port.cfg.get("image_border_margin_px", 4)
        if x.min() < margin or y.min() < margin or x.max() >= w - margin or y.max() >= h - margin:
            self.port.show_perception(
                observation["observation"], detections["candidates"], task["target_pixel_xy"]
            )
            return SkillResult(
                "retryable",
                data={"observation": observation["observation"], "points": target.points},
                code="target_clipped",
                message="目标区域被画面边缘截断，先换视角获取完整几何。",
                summary={
                    "bbox_xyxy": [int(x.min()), int(y.min()), int(x.max()), int(y.max())],
                    "border_margin_px": margin,
                },
            )
        self.port.show_target(observation["observation"], target, task["destination_color"])
        return SkillResult(
            "ok",
            {"target": target, "frame_id": task["frame_id"]},
            summary={
                "center_m": target.center.tolist(),
                "height_m": target.height,
                "source": "sam2_mask_and_calibrated_wrist_depth"
                if self.port.cfg.get("segmentation", {}).get("enabled", False)
                else "calibrated_wrist_depth",
            },
        )

    def resolve_destination(self, task):
        color = task["destination_color"]
        if color not in self.port.scene_cfg["trays"]:
            return SkillResult("failed", code="unknown_destination", message="目的区域未配置。")
        data = {
            "color": color,
            "xy": self.port.scene_cfg["trays"][color].get(
                "placement_xy_m", self.port.scene_cfg["trays"][color]["center_xy_m"]
            ),
            "floor": self.port.scene_cfg["table"]["top_m"]
            + self.port.scene_cfg["tray_geometry"]["floor_thickness_m"],
        }
        return SkillResult("ok", data, summary={**data, "source": "configured_tray_geometry"})

    def pick_object(self, target, destination):
        # Inference and planning can outlive the observation. Recheck using a fresh
        # wrist RGB-D frame before moving out of the camera's observable range.
        self.port.stage("pregrasp_check", "抓前复核：重新拍照，检查目标是否移动或被遮挡。")
        observation = self.port.capture()
        candidates = candidates_from_rgbd(
            observation,
            self.port.world.backend.camera,
            self.port.scene_cfg["table"]["top_m"],
            self.port.scene_cfg.get("perception_workspace"),
            self.port.scene_cfg["trays"],
        )
        try:
            if self.port.cfg.get("segmentation", {}).get("enabled", False):
                from piper_sim.calibration import transform_points

                # Reproject previous geometry into the fresh RGB-D frame. This is
                # bounded geometric association, not a persistent SAM video ID.
                T = observation.T_base_end @ np.asarray(
                    self.port.world.backend.camera["hand_eye"]["T_end_camera"]
                )
                camera_point = transform_points(np.linalg.inv(T), target.center)
                if camera_point[2] <= 0:
                    raise ValueError("抓前目标已移出相机前方，需要重新观察。")
                k = observation.intrinsics
                pixel = [
                    k["fx"] * camera_point[0] / camera_point[2] + k["cx"],
                    k["fy"] * camera_point[1] / camera_point[2] + k["cy"],
                ]
                candidates = [self.port.refine_target(observation, pixel)]
            target, recheck = match_fresh_target(
                target,
                candidates,
                {
                    **self.port.cfg.get("grasp_feedback", {}),
                    "image_border_margin_px": self.port.cfg.get("image_border_margin_px", 4),
                },
            )
        except GraspFeedbackError as exc:
            return SkillResult("retryable", code=exc.code, message=str(exc), summary=exc.evidence)
        except ValueError as exc:
            return SkillResult("retryable", code="target_changed", message=str(exc))
        except (RuntimeError, TimeoutError, OSError) as exc:
            return SkillResult("failed", code="segmentation_error", message=str(exc))
        self.port.show_target(observation, target, destination["color"])
        self.port.stage("planning")
        try:
            plan = self.motion.plan(target, destination["xy"], destination["floor"])
        except RuntimeError as exc:
            return SkillResult("failed", code="unreachable", message=str(exc))
        try:
            holding = self.motion.pick(plan)
        except GraspFeedbackError as exc:
            return self.grasp_failure(exc)
        return SkillResult("ok", {"plan": plan}, summary={**holding, "pregrasp_recheck": recheck})

    def grasp_failure(self, exc):
        return SkillResult(
            "retryable" if exc.retryable else "failed",
            code=exc.code,
            message=str(exc),
            summary={**exc.evidence, "feedback_events": list(self.motion.feedback_events)},
        )

    def place_object(self, grasp):
        try:
            plan = self.motion.place(grasp["plan"])
        except GraspFeedbackError as exc:
            return self.grasp_failure(exc)
        return SkillResult("ok", {"plan": plan}, summary=plan)

    def verify_result(self, binding, destination):
        self.port.stage("verify")
        result = self.port.evaluate_selection(binding, destination["color"])
        return SkillResult(
            "ok" if result["success"] else "failed",
            result,
            code=None if result["success"] else "placement_unverified",
            message=result["message"],
            summary={
                "selected_object_physical_success": result.get(
                    "selected_object_physical_success", result["success"]
                ),
                "language_semantics_verified": result.get("language_semantics_verified", False),
                "source": "simulation_evaluator_only",
            },
        )

    def recover_or_continue(self, failure, view_index, grasp_attempt):
        self.port.stage("recover", failure.message)
        active_cfg = self.port.cfg.get("active_view", {})
        if (
            failure.status == "retryable"
            and failure.code == "target_clipped"
            and active_cfg.get("enabled", True)
            and self.recenter_attempts < active_cfg.get("max_attempts", 3)
        ):
            self.recenter_attempts += 1
            try:
                q, detail = self.motion.recenter(
                    failure.data["observation"], failure.data["points"]
                )
            except ValueError as exc:
                self.port.stage("recover", str(exc))
            else:
                self.pending_view_q = q
                decision = {
                    "decision": "recenter_camera",
                    "next_view": view_index,
                    "attempt": self.recenter_attempts,
                    **detail,
                }
                return SkillResult(
                    "ok",
                    decision,
                    message="根据目标深度区域主动居中，然后重新识别和定位。",
                    summary=decision,
                )
        grasp_failures = ("empty_grasp", "grip_lost", "target_changed")
        if failure.status == "retryable" and failure.code in grasp_failures:
            # Even the last failed attempt retracts before a later instruction
            # can open the gripper or sweep toward another observation pose.
            if failure.code in ("empty_grasp", "grip_lost") and self.motion is not None:
                try:
                    self.motion.recover_empty()
                except (RuntimeError, TimeoutError) as exc:
                    return SkillResult("failed", code="recovery_failed", message=str(exc))
            if grasp_attempt < self.port.cfg.get("max_grasp_attempts", 2):
                decision = {
                    "decision": "reobserve_and_retry_grasp",
                    "next_view": max(0, view_index),
                }
                return SkillResult(
                    "ok",
                    decision,
                    message="重新观察、识别和定位后再尝试，保留当前物体状态。",
                    summary=decision,
                )
        elif failure.code == "missing_target_depth" and view_index < self.port.cfg.get(
            "depth_recovery_view_index", len(self.views)
        ) < len(self.views):
            decision = {
                "decision": "increase_camera_distance",
                "next_view": self.port.cfg["depth_recovery_view_index"],
            }
            return SkillResult(
                "ok", decision, message="选点没有有效深度，转到退远观察位。", summary=decision
            )
        elif failure.status == "retryable" and view_index + 1 < len(self.views):
            decision = {"decision": "change_view", "next_view": view_index + 1}
            return SkillResult(
                "ok", decision, message="当前证据不足，调用观察技能切换视角。", summary=decision
            )
        message = (
            "可用视角已检查，仍无法确定目标。请调整指令或场景。"
            if failure.status == "retryable" and failure.code not in grasp_failures
            else failure.message
        )
        return SkillResult(
            "failed",
            code=failure.code,
            message=message,
            summary={"decision": "stop", "reason": failure.code},
        )


class SkillCoordinator:
    def __init__(self, skills):
        self.skills = skills

    def run(self, request):
        skills = self.skills
        preflight = skills.call("interpret_task", request=request)
        if preflight.status == "needs_clarification":
            return {
                "success": False,
                "status": "needs_clarification",
                "clarification": preflight.data,
                "message": preflight.message,
                "failure_code": preflight.code,
                "skill_trace": skills.trace,
            }
        view_index = grasp_attempt = 0
        while True:
            observed = skills.call("observe_scene", view_index=view_index)
            view_index = observed.data.get("view_index", view_index)
            detected = skills.call("detect_objects", observation=observed.data)
            interpreted = skills.call(
                "interpret_task",
                request=request,
                observation=observed.data,
                detections=detected.data,
            )
            failure = interpreted if interpreted.status != "ok" else None
            if failure is None:
                target = skills.call(
                    "resolve_target",
                    task=interpreted.data,
                    detections=detected.data,
                    observation=observed.data,
                )
                failure = target if target.status != "ok" else None
            if failure is None:
                destination = skills.call("resolve_destination", task=interpreted.data)
                failure = destination if destination.status != "ok" else None
            if failure is None:
                # The binding is an opaque evaluator token. It is not passed to any action skill.
                binding = skills.port.bind_evaluator(target.data["target"])
                grasp_attempt += 1
                grasp = skills.call(
                    "pick_object", target=target.data["target"], destination=destination.data
                )
                failure = grasp if grasp.status != "ok" else None
            if failure is None:
                placed = skills.call("place_object", grasp=grasp.data)
                failure = placed if placed.status != "ok" else None
            if failure is not None:
                recovery = skills.call(
                    "recover_or_continue",
                    failure=failure,
                    view_index=view_index,
                    grasp_attempt=grasp_attempt,
                )
                if recovery.status != "ok":
                    return {
                        "success": False,
                        "error": recovery.message,
                        "failure_code": recovery.code,
                        "skill_trace": skills.trace,
                    }
                view_index = recovery.data["next_view"]
                continue
            verified = skills.call("verify_result", binding=binding, destination=destination.data)
            if verified.status != "ok":
                skills.call(
                    "recover_or_continue",
                    failure=verified,
                    view_index=view_index,
                    grasp_attempt=grasp_attempt,
                )
            return {**verified.data, **placed.data, "skill_trace": skills.trace}
