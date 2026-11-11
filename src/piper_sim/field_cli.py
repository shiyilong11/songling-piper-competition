"""Run L1 and L2 on the PiPER ROS 2 bench, or inspect the bench read-only."""

import argparse
import json
import shutil
import signal
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from piper_sim.calibration import ROOT
from piper_sim.competition_controller import L1SensorController
from piper_sim.control import load_interface
from piper_sim.field_calibration import FieldCalibration
from piper_sim.field_dashboard import FieldDashboard
from piper_sim.field_motion import TableClearance
from piper_sim.field_process import FieldProcess
from piper_sim.l2_sensor_controller import L2SensorController
from piper_sim.sensor_robot import SensorRobot

COLOR_LABELS = {"red", "green", "yellow", "orange_brown", "yellow_green"}
ZONES = {"red", "green", "yellow"}
FRUIT_SORT = {
    "banana": "yellow",
    "potato": "yellow",
    "cucumber": "green",
    "pear": "green",
    "bell_pepper": "red",
    "persimmon": "red",
}


def parse_sort(value, colors):
    mapping = {}
    for item in value.split(","):
        source, separator, destination = item.partition("=")
        source, destination = source.strip(), destination.strip()
        if not separator or source not in colors or destination not in ZONES or source in mapping:
            raise ValueError(f"Invalid colour task-card entry: {item}")
        mapping[source] = destination
    return mapping


def _load(path):
    return yaml.safe_load((ROOT / path).read_text())


def _stage_printer(process):
    previous = None

    def stage(name, **fields):
        nonlocal previous
        if process is not None:
            process.stage(name)
        if name != previous or fields:
            print(json.dumps({"stage": name, **fields}, ensure_ascii=False), flush=True)
            previous = name

    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "jog", "l1", "l2"))
    parser.add_argument(
        "--task", choices=("l1", "l2"), default="l1", help="Task to inspect in probe"
    )
    parser.add_argument("--calibration", type=Path, default=ROOT / "field_calibration.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--l1-sort", help="L1 task card if the three colour destinations differ")
    parser.add_argument(
        "--l2-model", choices=("sam2", "yoloe_sam2"), default="yoloe_sam2", help="L2 mask backend"
    )
    parser.add_argument("--l2-category-backend", choices=("qwen", "rules"))
    parser.add_argument("--dashboard-port", type=int, help="Local read-only browser dashboard")
    parser.add_argument("--preview-seconds", type=float, default=0.0, help="Keep a probe view open")
    parser.add_argument("--l2-sort", help="Override L2 sorting, e.g. banana=yellow,potato=yellow")
    parser.add_argument("--speed-percent", type=int, default=15)
    parser.add_argument("--joint", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--delta-rad", type=float, default=0.03)
    args, ros_args = parser.parse_known_args()
    if ros_args and ros_args[0] != "--ros-args":
        parser.error(f"Unrecognised arguments: {ros_args}")
    if not 1 <= args.speed_percent <= 100:
        parser.error("PiPER driver speed must be between 1 and 100 percent")
    if not np.isfinite(args.delta_rad) or abs(args.delta_rad) > 0.05:
        parser.error("Jog magnitude must be at most 0.05 rad")
    if args.dashboard_port is not None and not 0 <= args.dashboard_port <= 65535:
        parser.error("Invalid dashboard port")
    if (
        not np.isfinite(args.preview_seconds)
        or args.preview_seconds < 0
        or (args.preview_seconds and (args.mode != "probe" or args.dashboard_port is None))
    ):
        parser.error("--preview-seconds requires probe with --dashboard-port")
    task = args.task if args.mode == "probe" else args.mode
    try:
        calibration = FieldCalibration(args.calibration)
        l1_mapping = parse_sort(args.l1_sort, ZONES) if args.l1_sort else None
        if l1_mapping is not None and (
            set(l1_mapping) != ZONES or set(l1_mapping.values()) != ZONES
        ):
            raise ValueError(
                "L1 card must specify a separate destination for each of three colours"
            )
        mapping = (
            parse_sort(args.l2_sort, COLOR_LABELS | FRUIT_SORT.keys()) if args.l2_sort else None
        )
        if mapping and mapping.keys() & COLOR_LABELS and mapping.keys() & FRUIT_SORT.keys():
            raise ValueError("An L2 task card must use fruit names or colour labels consistently")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(f"Calibration or task card: {exc}")

    import rclpy
    from rclpy.signals import SignalHandlerOptions

    from piper_sim.field_ros import FieldROS

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    cv2.setNumThreads(1)
    cfg = _load("config/competition/sensor_tasks.yaml")
    motion_cfg = _load("config/interactive_demo.yaml")
    cfg["fixed_camera"] = calibration.fixed
    cfg["wall_clock"] = True
    cfg["target_edge_m"] = 0.20  # Public dimensions in the Nov 10 competition rules.
    cfg["l2_precise_packing"] = True
    if l1_mapping is not None:
        cfg["l1_mapping"] = l1_mapping
    if task == "l2":
        cfg["l2_color_mapping"] = mapping if mapping is not None else FRUIT_SORT
        cfg["l2_classify_fruits"] = not (cfg["l2_color_mapping"].keys() & COLOR_LABELS)
        cfg["l2_category_backend"] = args.l2_category_backend or motion_cfg.get("demos", {}).get(
            "l2_category_backend", "rules"
        )
        if cfg["l2_category_backend"] == "qwen" and not cfg["l2_classify_fruits"]:
            parser.error("Qwen classification uses the six fruit categories")
        cfg["l2_mapping_status"] = (
            "user_confirmed_fruit_task_card" if mapping is None else "operator_supplied_task_card"
        )
    motion_cfg["max_task_s"] = cfg["time_limits_s"].get(task, 30)
    motion_cfg["preserve_gripper_on_hold"] = True
    motion_cfg["extra_table_clearance_m"] = 0.004
    if task == "l2":
        motion_cfg["release_clearance_m"] = 0.008
    motion_cfg["grasp_tilt_candidates_rad"] = [0.6, 0.9, 0.35, 0.15, -0.15, -0.35, -0.6, -0.9]
    interface = load_interface()
    interface["ros2"]["driver_speed_percent"] = args.speed_percent
    output = args.output or ROOT / "logs" / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.mode}"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.calibration, output / "calibration_used.yaml")
    process = dashboard = None
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    session, controller = None, None
    result = {"mode": args.mode, "task": task, "success": False}
    if task == "l2":
        result["l2_model"] = args.l2_model
        result["l2_category_backend"] = cfg["l2_category_backend"]
        result["l2_destinations"] = cfg["l2_color_mapping"]
    try:
        if args.dashboard_port is not None:
            dashboard = FieldDashboard(
                args.dashboard_port, (calibration.depth_min, calibration.depth_max)
            )
            print(f"现场任务台  {dashboard.url}", flush=True)
        if task in {"l1", "l2"}:
            process = FieldProcess(
                output,
                task,
                cfg["l1_mapping"] if task == "l1" else cfg["l2_color_mapping"],
                motion_cfg["max_task_s"],
                probe=args.mode == "probe",
                dashboard=dashboard,
            )
            print(f"过程展示  {process.path.resolve()}", flush=True)
        session = FieldROS(interface, calibration)
        if dashboard is not None:
            dashboard.attach_camera(session.preview_pair)
        session.wait_ready()
        robot = SensorRobot(
            interface,
            session.read_state,
            lambda _: session.send(robot.previous_target),
            session.pause,
        )
        if calibration.fixed:
            pose = robot.measured()[:6].tolist()
            cfg["observation_q_rad"] = cfg["l2_observation_q_rad"] = [pose]
        controller_type = L2SensorController if task == "l2" else L1SensorController
        model_kwargs = {"method": args.l2_model} if task == "l2" else {}
        controller = controller_type(
            robot,
            lambda: session.capture(robot),
            cfg,
            motion_cfg,
            calibration.controller_config(),
            _load("config/scene_observation.yaml")["plane"],
            output,
            _stage_printer(process),
            process=process,
            **model_kwargs,
        )
        obs, blocks, zones = controller.observe()
        result["initial_observation"] = {
            "table_m": controller.scene.table,
            "zones": {name: zone.as_dict() for name, zone in zones.items()},
            "blocks": [
                {"colour": name, "center_m": target.center.tolist()} for name, target in blocks
            ],
            "aligned_depth_fraction": float(np.isfinite(obs.depth_m).mean()),
            "mount": "fixed" if calibration.fixed else "wrist",
        }
        guard = TableClearance(robot, calibration, controller.scene.table)
        guard(robot.measured(), robot.measured())
        robot.path_guard = guard
        if args.mode == "probe":
            driver, multiplier = session.driver_parameters()
            result.update(
                driver=driver, gripper_val_mutiple=multiplier, command_topic=session.command_topic
            )
            if task == "l2":
                targets, seg = controller.segmenter.detect(
                    obs,
                    controller.scene.calibration,
                    controller.scene.table,
                    controller.scene.zones,
                    output / "segmentation",
                    include_zones=True,
                )
                for target in seg["accepted_targets"]:
                    target["destination"] = cfg["l2_color_mapping"].get(target["color"])
                result["fruit_observation"] = seg
                result["fruit_count"] = len(targets)
                process.fruits(targets, allow_new=True, observation=obs)
            result["success"] = True
            if args.preview_seconds:
                time.sleep(args.preview_seconds)
        else:
            result["transport"] = session.begin(motion_cfg["max_task_s"], guard)
            if process is not None:
                process.begin(session.clock)
            robot.sync()
            controller.motion.started = robot.data.time
            if args.mode == "jog":
                before = robot.measured()
                goal = before.copy()
                goal[args.joint - 1] += args.delta_rad
                lo, hi = interface["joints"]["lower_rad"], interface["joints"]["upper_rad"]
                if np.any(goal[:6] < lo) or np.any(goal[:6] > hi):
                    raise ValueError("Jog would exceed the calibrated PiPER joint limits")
                controller.motion.stage = "jog"
                controller.motion.move(goal[:6], goal[6])
                controller.motion.hold(0.25)
                after = robot.measured()
                result["jog_feedback_rad"] = float(after[args.joint - 1] - before[args.joint - 1])
                result["success"] = bool(abs(after[args.joint - 1] - goal[args.joint - 1]) < 0.010)
            else:
                task_report = controller.run()
                task_report["transport"] = "ROS 2 Humble"
                task_report["round_clock"] = "monotonic wall time"
                task_report["limitation"] = (
                    "Gripper width estimates retention; no independent visual lift proof"
                )
                result["task_report"] = task_report
                result["success"] = bool(
                    task_report["placement_success"] and task_report["failure"] is None
                )
    except (KeyboardInterrupt, RuntimeError, TimeoutError, ValueError, OSError) as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        if session is not None:
            if args.mode != "probe":
                reason = (
                    result.get("failure")
                    or result.get("task_report", {}).get("failure")
                    or "session_finished"
                )
                result["termination"] = session.stop(reason)
            if isinstance(controller, L2SensorController):
                controller.segmenter.close()
        if process is not None:
            process.finish(
                result["success"],
                result.get("failure") or result.get("task_report", {}).get("failure"),
            )
            result["process"] = {
                "display": process.path.name,
                "numbered_log": process.log.name,
                "available": process.available,
            }
        if dashboard is not None:
            dashboard.close()
        if session is not None:
            session.close()
        if rclpy.ok():
            rclpy.shutdown()
        (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"{'完成' if result['success'] else '未通过'}  {output / 'report.json'}", flush=True)
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
