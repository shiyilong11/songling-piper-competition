"""Instruction-first demo. Simulation and OpenGL stay on one owner thread."""

import argparse
import base64
import io
import json
import queue
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from wrist_qwen_panel import CameraPanel

from piper_sim.calibration import ROOT
from piper_sim.competition_controller import L1SensorController
from piper_sim.competition_world import CompetitionWorld
from piper_sim.demo_skill import PickPlaceSkill, candidates_from_rgbd, select_pixel
from piper_sim.field_dashboard import render_segmentation
from piper_sim.field_process import FieldProcess
from piper_sim.l2_sensor_controller import L2SensorController
from piper_sim.l3_competition import CARDS, ExplorationController, parse_round
from piper_sim.sam_adapter import Sam2Client
from piper_sim.sensor_robot import SensorRobot
from piper_sim.skill_runtime import ManipulationSkills, SkillCoordinator
from piper_sim.wrist_camera import wrist_scene_config


class Cancelled(RuntimeError):
    pass


class ExplorationWorld(CompetitionWorld):
    def export_metadata(self):
        # The exploration controller never needs the simulation evaluator's answer file.
        return None


class Demo(CameraPanel):
    def __init__(self, output, config, task="l3"):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.cfg = yaml.safe_load(config.read_text())
        self.settings, self.scene_cfg = wrist_scene_config(
            hand_eye_config=self.cfg.get("hand_eye_config")
        )
        self.worker = self.worker_log = self.world = None
        self.sam = None
        self.sequence = self.requests = 0
        self.commands = queue.Queue()
        self.lock = threading.RLock()
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self.last_render = 0
        self.state_data = {
            "busy": False,
            "stage": "idle",
            "paused": False,
            "logs": [],
            "result": None,
            "backend": self.cfg["default_backend"],
            "instruction": "",
            "rgb": None,
            "segmentation": None,
            "selected_rgb": None,
            "model_loaded": False,
        }
        initial_scene = {"l1": "blocks", "l2": "fruit"}.get(task, self.cfg["default_scene"])
        self.reset_scene(self.cfg["default_seed"], initial_scene, task)

    def update(self, **values):
        with self.lock:
            self.state_data.update(values)

    def state(self):
        with self.lock:
            return dict(self.state_data)

    def stage(self, name, detail=None):
        with self.lock:
            if name != self.state_data["stage"] or detail:
                self.state_data["logs"] = (
                    self.state_data["logs"]
                    + [{"stage": name, "detail": detail, "wall": time.strftime("%H:%M:%S")}]
                )[-100:]
            self.state_data["stage"] = name

    def reset_scene(self, seed, scene, task_mode="l3"):
        # Build first: a bad random packing must not destroy the current scene.
        import copy

        candidate_cfg = copy.deepcopy(self.scene_cfg)
        demo_cfg = self.cfg.get("demos", {})
        if task_mode == "l3":
            scene = "mixed"
            task = candidate_cfg["tasks"]["exploration"]
            task["objects"] = [
                "cube_red",
                "cube_green",
                "cube_yellow",
                "persimmon",
                "persimmon",
                "potato",
                "potato",
                "banana",
            ]
            task["spawn_override"] = {"x_m": [0.15, 0.35], "y_m": [-0.27, 0.27]}
            candidate_cfg["robot_initial_qpos"][:6] = demo_cfg["l3"]["survey_q_rad"]["center"]
        if scene == "fruit":
            candidate_cfg["spawn"].update(demo_cfg.get("fruit_spawn", {}))
        l2_initial_q = demo_cfg.get("l2_initial_q_rad")
        if task_mode == "l2" and l2_initial_q is not None:
            candidate_cfg["robot_initial_qpos"][:6] = l2_initial_q
        if "target_edge_m" in demo_cfg:
            for zone in candidate_cfg["trays"].values():
                zone["half_size_xy_m"] = [demo_cfg["target_edge_m"] / 2] * 2
        candidate_dir = self.output / "scenes" / str(time.time_ns())
        candidate_dir.mkdir(parents=True)
        config_path = candidate_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(candidate_cfg))
        world_type = ExplorationWorld if task_mode == "l3" else CompetitionWorld
        candidate = world_type(
            task_mode if task_mode in ("l1", "l2") else "exploration",
            config_path,
            candidate_dir,
            render=True,
        )
        try:
            candidate.reset(seed)
        except Exception:
            candidate.close()
            raise
        if self.world:
            if self.world.backend.renderer:
                # Rebind the old GL context before freeing its renderer.
                self.world.backend.renderer.render()
            self.world.close()
        self.world = candidate
        camera = candidate.cfg["camera"]
        self.seed, self.scene, self.task_mode = seed, scene, task_mode
        self.config_path = config_path
        self.update(
            seed=seed,
            scene=scene,
            result=None,
            selected_rgb=None,
            segmentation=None,
            target=None,
            model_response=None,
            needs_reset=False,
            wall_time_s=0,
            skill_trace=[],
            active_skill=None,
            task_mode=task_mode,
            l2_category_backend=demo_cfg.get("l2_category_backend", "rules"),
            camera={
                name: camera[name]
                for name in ("model", "width", "height", "intrinsics", "calibration_status")
            },
            round_progress=None,
            l3_progress=None,
            l3_cards={number: text for number, (text, _) in CARDS.items()},
            demo_rules=self.scene_cfg["tasks"].get(task_mode, {}).get("mapping", {}),
        )
        self.render()

    @staticmethod
    def encode(rgb):
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def render(self, *, include_depth=False):
        b = self.world.backend
        obs = b.observe(render_depth=include_depth)
        self.update(
            rgb=self.encode(obs.rgb),
            sim_time_s=round(float(b.data.time), 2),
            q=b.measured().tolist(),
            model_loaded=self.worker is not None and self.worker.poll() is None,
        )
        self.last_render = time.monotonic()
        return obs

    def show_task_masks(self, observation, candidates):
        segmentation = render_segmentation(observation, candidates) if candidates else None
        self.update(segmentation=segmentation)

    def checkpoint(self):
        while self.pause_flag.is_set() and not self.stop_flag.is_set():
            time.sleep(0.05)
        if self.stop_flag.is_set():
            raise Cancelled("执行已停止，现场与夹爪状态已保留。可以直接输入新指令。")

    def tick(self, stage):
        self.stage(stage)
        self.checkpoint()
        now = time.monotonic()
        if now - self.last_render >= 1 / self.cfg["render_hz"]:
            self.render()
        if self.task_mode in ("l1", "l2", "l3"):
            # Pace the round from simulation time so one slow UI frame does
            # not add its latency to every subsequent control period.
            deadline = self.started + self.world.elapsed_s / self.cfg["realtime_factor"]
        else:
            hz = self.world.backend.interface["control"]["policy_hz"]
            deadline = self.last_tick + 1 / (hz * self.cfg["realtime_factor"])
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        self.last_tick = time.monotonic()
        self.update(wall_time_s=round(time.monotonic() - self.started, 1))
        progress = self.state().get("round_progress")
        if progress:
            self.update(round_progress={**progress, "elapsed_s": round(self.world.elapsed_s, 2)})

    def advance(self, action):
        self.checkpoint()
        if self.world.elapsed_s >= self.scene_cfg["tasks"][self.world.task]["time_limit_s"] - 1e-8:
            raise TimeoutError("本次任务时间已用尽，现场已保留，可以输入新指令。")
        self.world.step(action, observe=False)

    def capture(self):
        self.sequence += 1
        self.frame = self.output / "frames" / f"{self.sequence:06d}"
        self.frame.mkdir(parents=True)
        obs = self.render(include_depth=True)
        Image.fromarray(obs.rgb).save(self.frame / "wrist_rgb.png")
        np.save(self.frame / "depth_m.npy", obs.depth_m)
        metadata = {
            "frame_id": self.sequence,
            "timestamp_s": obs.timestamp_s,
            "joint_position_rad": obs.joint_position_rad.tolist(),
            "T_base_end": obs.T_base_end.tolist(),
            "intrinsics": obs.intrinsics,
            "depth_valid_fraction": float(np.isfinite(obs.depth_m).mean()),
            "T_end_camera": self.world.backend.camera["hand_eye"]["T_end_camera"],
        }
        (self.frame / "observation.json").write_text(json.dumps(metadata, indent=2))
        return obs

    def show_perception(self, obs, candidates, pixel):
        image = Image.fromarray(obs.rgb)
        draw = ImageDraw.Draw(image)
        for i, candidate in enumerate(candidates):
            y, x = np.where(candidate.mask)
            bounds = (int(x.min()), int(y.min()), int(x.max()), int(y.max()))
            draw.rectangle(bounds, outline="yellow", width=2)
            draw.text((bounds[0], bounds[1]), str(i + 1), fill="white")
        x, y = pixel
        draw.line((x - 8, y - 8, x + 8, y + 8), fill="red", width=3)
        draw.line((x - 8, y + 8, x + 8, y - 8), fill="red", width=3)
        self.update(
            selected_rgb=self.encode(np.asarray(image)),
            target={"diagnostic": "红叉为模型选点，黄框为有效深度候选", "pixel_xy": pixel},
        )

    def refine_target(self, obs, pixel):
        """Same-frame semantic mask intersected with calibrated, valid depth."""
        config = self.cfg.get("segmentation", {})
        if not config.get("enabled", False):
            raise RuntimeError("SAM refinement called while segmentation is disabled")
        if self.sam is None:
            self.sam = Sam2Client(ROOT / config["config"])
        instance, metadata = self.sam.segment(
            obs.rgb,
            pixel,
            obs.timestamp_s,
            self.sequence,
            self.frame,
        )
        candidates = candidates_from_rgbd(
            obs,
            self.world.backend.camera,
            self.scene_cfg["table"]["top_m"],
            self.scene_cfg.get("perception_workspace"),
            self.scene_cfg["trays"],
            instance_mask=instance.mask,
        )
        target = select_pixel(candidates, pixel)
        # Preserve the complete visual boundary for clipping checks. Points above
        # come only from valid masked depth inside the configured workspace.
        target.mask = instance.mask
        self.update(segmentation=metadata)
        return target

    def show_target(self, obs, target, destination):
        overlay = obs.rgb.copy()
        overlay[target.mask] = (
            0.55 * overlay[target.mask] + 0.45 * np.array([0, 220, 180])
        ).astype(np.uint8)
        marked = Image.fromarray(overlay)
        draw = ImageDraw.Draw(marked)
        y, x = np.where(target.mask)
        draw.rectangle(
            (int(x.min()), int(y.min()), int(x.max()), int(y.max())), outline="yellow", width=3
        )
        draw.ellipse(
            (target.pixel[0] - 5, target.pixel[1] - 5, target.pixel[0] + 5, target.pixel[1] + 5),
            fill="cyan",
        )
        self.update(
            selected_rgb=self.encode(np.asarray(marked)),
            target={
                "center_m": target.center.tolist(),
                "height_m": target.height,
                "destination": destination,
                "point_refinement_distance_px": target.snap_distance_px,
            },
        )

    def bind_evaluator(self, target):
        # Truth isolated here: only match a selected sensor region for later evaluation.
        b = self.world.backend
        return min(
            self.world.objects,
            key=lambda obj: np.linalg.norm(b.data.body(obj["id"]).xpos[:2] - target.center[:2]),
        )["id"]

    def evaluate_selection(self, matched, destination):
        evaluation = self.world.evaluation()
        event = evaluation["events"][matched]
        physical_success = bool(
            event["grasped"]
            and destination in event["placed"]
            and event["stable_s"][destination]
            >= self.scene_cfg["provisional_judging"]["stable_hold_s"]
        )
        competition = self.task_mode in ("l1", "l2")
        category = next(obj["category"] for obj in self.world.objects if obj["id"] == matched)
        classified = (
            self.scene_cfg["tasks"][self.task_mode]["mapping"].get(category) == destination
            if competition
            else None
        )
        success = physical_success and (classified if competition else True)
        self.render()
        result = {
            "success": success,
            "evaluation": evaluation,
            "selected_object_physical_success": physical_success,
            "language_semantics_verified": competition,
            "classification_correct": classified,
            "message": "物体按演示分类规则放入对应托盘并保持稳定。"
            if success and competition
            else "所选物体已抓起并稳定放入指定托盘。请核对选物是否符合原指令。"
            if success
            else "物体已放置，但分类与演示规则不符。"
            if physical_success and competition
            else "动作已结束，但未通过抓取和稳定放置检查。",
        }
        return result

    def execute(self, request, new_attempt=True):
        if new_attempt:
            self.started = self.last_tick = time.monotonic()
            self.world.begin_attempt()
            self.update(round_progress=None)
        self.start_from_current = True
        self.update(
            result=None,
            selected_rgb=None,
            segmentation=None,
            target=None,
            model_response=None,
            wall_time_s=0,
            skill_trace=[],
            active_skill=None,
            needs_reset=False,
        )
        motion = PickPlaceSkill(
            self.world.backend,
            self.scene_cfg["table"]["top_m"],
            self.tick,
            self.cfg,
            stepper=self.advance,
        )
        skills = ManipulationSkills(self, motion)
        start_sequence = self.sequence
        run_id = str(time.time_ns())
        record_path = self.output / "runs" / (run_id + ".json")
        record_path.parent.mkdir(exist_ok=True)
        result = None
        try:
            result = SkillCoordinator(skills).run(request)
            self.render()
            self.update(result=result, needs_reset=False, active_skill=None)
            self.stage(
                "clarify"
                if result.get("status") == "needs_clarification"
                else "done"
                if result["success"]
                else "failed",
                result.get("message") or result.get("error"),
            )
        except Exception as exc:
            result = {"success": False, "error": str(exc), "exception": type(exc).__name__}
            raise
        finally:
            record = {
                "run_id": run_id,
                "instruction": request["instruction"],
                "backend": request["backend"],
                "scene": self.scene,
                "seed": self.seed,
                "task_mode": self.task_mode,
                "round_id": request.get("round_id"),
                "result": result,
                "skill_trace": skills.trace,
                "frame_id": self.sequence,
                "wall_time_s": round(time.monotonic() - self.started, 3),
            }
            record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            if result is not None and self.sequence > start_sequence:
                (self.frame / "execution.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2)
                )
        return result

    def submit(self, action, request):
        with self.lock:
            busy = self.state_data["busy"]
            if action in ("pause", "resume", "stop"):
                if not busy:
                    raise ValueError("当前没有正在执行的任务")
                if action == "pause":
                    self.pause_flag.set()
                elif action == "resume":
                    self.pause_flag.clear()
                else:
                    self.stop_flag.set()
                    self.pause_flag.clear()
                self.state_data["paused"] = self.pause_flag.is_set()
                return {"accepted": True}
            if action not in ("run", "reset", "release"):
                raise ValueError("未知操作")
            if busy:
                raise ValueError("当前任务尚未结束，请先停止。")
            if action == "run":
                mode = self.state_data.get("task_mode", "l3")
                if mode in ("l1", "l2"):
                    request = {
                        "instruction": mode.upper() + " 自动演示",
                        "backend": "rule" if mode == "l1" else "yoloe_sam2",
                    }
                instruction = request.get("instruction")
                if (
                    not isinstance(instruction, str)
                    or not instruction.strip()
                    or len(instruction) > 2000
                ):
                    raise ValueError("请输入 1–2000 字的指令")
                if mode == "l3":
                    request = {**request, "backend": "qwen", "cards": parse_round(instruction)}
                self.state_data.update(instruction=instruction, backend=request["backend"])
            if action == "reset":
                request = dict(request)
                seed = request.get("seed", secrets.randbelow(2**31))
                request["seed"] = seed
                request.setdefault("scene", self.state_data.get("scene", "fruit"))
                mode = request.get("task_mode", self.state_data.get("task_mode", "l3"))
                if mode not in ("l1", "l2", "l3"):
                    raise ValueError("请选择 L1、L2 或 L3 演示")
                request["task_mode"] = mode
                if mode in ("l1", "l2"):
                    request["scene"] = "blocks" if mode == "l1" else "fruit"
                if (
                    type(seed) is not int
                    or not 0 <= seed < 2**31
                    or request.get("scene") not in ("blocks", "fruit", "mixed")
                ):
                    raise ValueError("布局种子须为非负整数，场景须为 blocks 或 fruit")
            self.stop_flag.clear()
            self.pause_flag.clear()
            self.state_data.update(busy=True, paused=False)
            self.commands.put((action, request))
            return {"accepted": True}

    def run_sensor_round(self):
        mode = self.task_mode
        world, b = self.world, self.world.backend
        world.time_origin_s = float(b.data.time)
        self.started = self.last_tick = time.monotonic()
        limit = self.scene_cfg["tasks"][mode]["time_limit_s"]
        config = yaml.safe_load((ROOT / "config/competition/sensor_tasks.yaml").read_text())
        plane = yaml.safe_load((ROOT / "config/scene_observation.yaml").read_text())["plane"]
        config.update(l2_precise_packing=True, l2_classify_fruits=True)
        demo_cfg = self.cfg.get("demos", {})
        config["l2_category_backend"] = demo_cfg.get("l2_category_backend", "rules")
        if "target_edge_m" in demo_cfg:
            config["target_edge_m"] = demo_cfg["target_edge_m"]
        if "observation_q_rad" in demo_cfg:
            config["observation_q_rad"] = (
                demo_cfg["observation_q_rad"] + config["observation_q_rad"]
            )
        config["l1_mapping"] = {
            category.removeprefix("cube_"): destination
            for category, destination in self.scene_cfg["tasks"]["l1"]["mapping"].items()
        }
        config["l2_color_mapping"] = self.scene_cfg["tasks"]["l2"]["mapping"]
        config["l2_mapping_status"] = "user_confirmed_fruit_task_card"
        mapping = config["l1_mapping"] if mode == "l1" else config["l2_color_mapping"]
        motion = {
            **self.cfg,
            "max_task_s": limit,
            "preserve_gripper_on_hold": True,
            "encoder_bias_correction": True,
            "prefer_independent_orientation": True,
            "grasp_tilt_candidates_rad": [0.6, 0.9, 0.35, 0.15, -0.15, -0.35, -0.6, -0.9],
        }
        if mode == "l2":
            motion.update(
                release_clearance_m=0.008,
                allow_grip_change_at_release=True,
                follow_planned_transport_segment=True,
            )
        folder = self.output / "rounds" / str(time.time_ns())
        folder.mkdir(parents=True)
        process = FieldProcess(folder, mode, mapping, limit, mask_observer=self.show_task_masks)

        def advance(action):
            self.checkpoint()
            if world.elapsed_s >= limit - 1e-8:
                raise TimeoutError(f"{mode.upper()} 已达到 {limit} 秒仿真时限")
            renderer, b.renderer = b.renderer, None
            try:
                b.step(action)
            finally:
                b.renderer = renderer

        robot = SensorRobot(
            b.interface, lambda: (float(b.data.time), b.measured()), advance, b._set_target
        )
        attempt_count = 0

        def publish():
            category_client = controller.segmenter.category_client if mode == "l2" else None
            if category_client is not None:
                self.update(model_loaded=category_client.loaded)
            self.update(
                round_progress={
                    "mode": mode,
                    "target_count": len(mapping),
                    "completed": len(controller.completed),
                    "time_limit_s": limit,
                    "elapsed_s": round(world.elapsed_s, 2),
                    "attempt": attempt_count,
                    "results": [row.copy() for row in process.rows.values()],
                }
            )

        def callback(stage, *, color=None, attempt=None):
            nonlocal attempt_count
            if attempt is not None:
                attempt_count += 1
            process.stage(stage)
            self.tick(stage)
            publish()

        controller_type = L1SensorController if mode == "l1" else L2SensorController
        controller = controller_type(
            robot,
            b.observe,
            config,
            motion,
            b.camera,
            plane,
            folder,
            callback,
            process=process,
            **({"method": "yoloe_sam2"} if mode == "l2" else {}),
        )
        self.update(
            result=None,
            wall_time_s=0,
            skill_trace=[],
            selected_rgb=None,
            segmentation=None,
            target=None,
            model_response=None,
        )
        process.begin(lambda: world.elapsed_s)
        publish()
        report = {"placement_success": False, "failure": "本轮中断"}
        try:
            report = controller.run()
        finally:
            report["execution_wall_s"] = round(time.monotonic() - self.started, 2)
            report["simulation_elapsed_s"] = round(world.elapsed_s, 2)
            self.update(wall_time_s=report["execution_wall_s"])
            if not report["placement_success"]:
                controller.motion.hold_current()
            process.finish(report["placement_success"], report["failure"])
            publish()
            (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        success = report["placement_success"]
        message = f"{mode.upper()} 复查通过 {len(controller.completed)}/{len(mapping)} 个目标"
        self.update(
            result={"success": success, "message": message, "error": report["failure"]},
            active_skill=None,
            needs_reset=False,
        )
        self.stage("done" if success else "failed", message)

    def run_exploration_round(self, request):
        settings = self.cfg["demos"]["l3"]
        requested = len(request["cards"])
        world, backend = self.world, self.world.backend
        world.time_origin_s = float(backend.data.time)
        self.started = self.last_tick = time.monotonic()
        folder = self.output / "rounds" / str(time.time_ns())
        folder.mkdir(parents=True)
        plane = yaml.safe_load((ROOT / "config/scene_observation.yaml").read_text())["plane"]
        config = yaml.safe_load((ROOT / "config/competition/sensor_tasks.yaml").read_text())
        config.update(
            target_edge_m=0.20,
            l2_precise_packing=True,
            l2_classify_fruits=True,
            l2_category_backend="qwen",
            l1_mapping={color: color for color in ("red", "green", "yellow")},
            l2_color_mapping={
                label: label
                for label in (
                    "potato",
                    "persimmon",
                    "banana",
                    "pear",
                    "cucumber",
                    "bell_pepper",
                )
            },
        )
        motion_config = {
            **self.cfg,
            "max_task_s": 300,
            "preserve_gripper_on_hold": True,
            "encoder_bias_correction": True,
            "release_clearance_m": 0.008,
            "allow_grip_change_at_release": True,
            "follow_planned_transport_segment": True,
            "grasp_tilt_candidates_rad": [0.6, 0.9, 0.35, 0.15, -0.15, -0.35, -0.6, -0.9],
        }
        report = {
            "mode": "sensor_only_ik",
            "official_score": None,
            "official_lift_verified": False,
            "judge_left_axis_base_xy": settings["judge_left_axis_base_xy"],
        }
        self.update(
            result=None,
            l3_progress=None,
            selected_rgb=None,
            segmentation=None,
            target=None,
            model_response=None,
            skill_trace=[],
            round_progress=None,
            wall_time_s=0,
        )

        def advance(action):
            self.checkpoint()
            if world.elapsed_s >= 300 - 1e-8:
                raise TimeoutError("L3 已达到 300 秒仿真时限。")
            renderer, backend.renderer = backend.renderer, None
            try:
                backend.step(action)
            finally:
                backend.renderer = renderer

        def publish(cards):
            visible_cards = json.loads(json.dumps(cards, ensure_ascii=False))
            trace = []
            for card in visible_cards:
                status = {
                    "verified": "ok",
                    "executing": "running",
                    "needs_observation": "retryable",
                    "not_verified": "failed",
                    "planned": "ready",
                }[card["status"]]
                trace.append(
                    {
                        "skill": f"指令 {card['number']}",
                        "status": status,
                        "message": card["reason"]
                        or (
                            f"{card['instruction']} · 已复查 {card['verified_actions']}/{card['required_actions']} 件"
                        ),
                    }
                )
            self.update(l3_progress=visible_cards, skill_trace=trace)

        robot = SensorRobot(
            backend.interface,
            lambda: (float(backend.data.time), backend.measured()),
            advance,
            backend._set_target,
        )
        controller = ExplorationController(
            robot,
            backend.observe,
            config,
            motion_config,
            backend.camera,
            plane,
            folder,
            self.tick,
            left_axis=settings["judge_left_axis_base_xy"],
            publish=publish,
            show_masks=self.show_task_masks,
        )
        controller.cards = []
        failure = None
        try:
            controller.run_round(
                request["cards"],
                settings["survey_q_rad"],
                self.cfg.get("demos", {}).get("observation_q_rad", [])
                + config["observation_q_rad"],
            )
        except (RuntimeError, ValueError, OSError, TimeoutError) as exc:
            failure = str(exc)
        finally:
            controller.motion.hold_current()
            controller.segmenter.close()
            report.update(
                cards=controller.cards,
                survey=controller.initial_survey,
                events=controller.events,
                failure=failure,
                simulation_elapsed_s=round(world.elapsed_s, 3),
                wall_time_s=round(time.monotonic() - self.started, 3),
            )
            verified = sum(card["status"] == "verified" for card in controller.cards)
            report["verified_instructions"] = verified
            report["requested_instructions"] = requested
            (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        result = {
            "success": verified == requested and failure is None,
            "message": f"L3 传感器复查通过 {verified}/{requested} 条指令。",
            "error": failure,
        }
        self.update(result=result, wall_time_s=report["wall_time_s"])
        self.stage(
            "stopped" if self.stop_flag.is_set() else "done" if result["success"] else "failed",
            result["message"],
        )

    def loop(self):
        while True:
            action, request = self.commands.get()
            try:
                if action == "run":
                    if self.task_mode in ("l1", "l2"):
                        self.run_sensor_round()
                    else:
                        self.run_exploration_round(request)
                elif action == "reset":
                    self.reset_scene(request["seed"], request["scene"], request["task_mode"])
                    self.stage("idle")
                elif action == "release":
                    self.release()
                    self.update(model_loaded=False)
            except Exception as exc:  # noqa: BLE001 -- per-command UI error boundary
                self.update(
                    result={"success": False, "error": str(exc)},
                    needs_reset=False,
                    active_skill=None,
                )
                self.stage("stopped" if isinstance(exc, Cancelled) else "failed", str(exc))
            finally:
                self.render()
                self.update(busy=False, paused=False)

    def release(self):
        self._release_single()

    def close(self):
        if getattr(self, "sam", None) is not None:
            self.sam.close()
        super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/interactive_demo.yaml")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/interactive_demo" / str(time.time_ns())
    )
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--task", choices=("l1", "l2", "l3"), default="l3")
    args = parser.parse_args()
    demo = Demo(args.output, args.config, args.task)
    if args.port is not None:
        demo.cfg["port"] = args.port

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, data, status=200, content_type="application/json; charset=utf-8"):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self.respond(
                    (ROOT / "assets/ui/interactive_demo.html").read_bytes(),
                    content_type="text/html; charset=utf-8",
                )
            elif self.path == "/api/state":
                self.respond(demo.state())
            else:
                self.respond({"error": "Not found"}, 404)

        def do_POST(self):
            try:
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("JSON required")
                origin = self.headers.get("Origin")
                if origin and origin != "http://" + self.headers.get("Host", ""):
                    raise ValueError("跨站请求被拒绝")
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 20000:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                if not self.path.startswith("/api/") or not isinstance(request, dict):
                    raise ValueError("Invalid request")
                self.respond(demo.submit(self.path[5:], request), 202)
            except (ValueError, KeyError, TypeError) as exc:
                self.respond({"error": str(exc)}, 400)

    server = ThreadingHTTPServer((demo.cfg["host"], demo.cfg["port"]), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Interactive demo: http://{demo.cfg['host']}:{server.server_port}/", flush=True)
    try:
        demo.loop()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        demo.close()


if __name__ == "__main__":
    main()
