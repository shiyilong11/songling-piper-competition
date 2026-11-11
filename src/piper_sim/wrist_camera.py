"""Compose the wrist-camera fixture using the existing hand-eye parameter file."""

from pathlib import Path

import yaml

from piper_sim.calibration import ROOT, load_config


def wrist_scene_config(path=ROOT / "config/vlm/wrist_camera.yaml", *, hand_eye_config=None):
    settings = yaml.safe_load(Path(path).read_text())
    if hand_eye_config is not None:
        settings["hand_eye_config"] = hand_eye_config
    scene = yaml.safe_load((ROOT / settings["scene_config"]).read_text())
    calibration = load_config(ROOT / settings["hand_eye_config"])
    scene["camera_config"] = settings["hand_eye_config"]
    sensor, hand_eye = calibration["camera"], calibration["hand_eye"]
    scene["camera"] = {
        "mount": "wrist",
        "parent_body": hand_eye["parent_body"],
        "T_parent_optical": hand_eye["T_end_camera"],
        "width": sensor["width"],
        "height": sensor["height"],
        "intrinsics": sensor["intrinsics"],
        "depth_range_m": [sensor["depth"]["min_valid_m"], sensor["depth"]["max_valid_m"]],
        "calibration_source": settings["hand_eye_config"],
        "calibration_status": calibration["calibration"]["status"],
        "visual": calibration["simulation"]["camera_visual"],
    }
    width = settings["gripper_width_m"]
    scene["robot_initial_qpos"] = settings["observation_q_rad"] + [width / 2, -width / 2]
    return settings, scene
